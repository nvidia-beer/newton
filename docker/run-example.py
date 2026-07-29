#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Helper for run-examples.sh.

Loads per-example JSON configs from ``docker/config/`` and resolves them into
CLI arguments for ``python -m newton.examples <name>``.

Subcommands:
    list     List examples (one "<name>\\t<description>" line per JSON file).
    resolve  Print the resolved CLI arg string for one config file, optionally
             after an interactive edit pass or with --set KEY=VAL overrides.

Design notes:
- Prompts go to stderr, resolved args go to stdout. This lets the caller
  capture the args via ``$(...)`` in bash while interactive prompts remain
  visible to the user.
- JSON schema (all fields optional):
    {
        "description": "Short one-liner for the menu.",
        "example":     "module_name",   # defaults to the config filename stem
        "args":        { "kebab-case-key": <json value>, ... }
    }
- Value encoding for argparse (Newton examples use argparse with
  ``BooleanOptionalAction`` for flags and ``nargs=N`` for vectors):
    * bool  -> ``--key``       (True)  /  ``--no-key`` (False)
    * null  -> skipped (argparse default wins)
    * list  -> ``--key v1 v2 v3``  (single flag, N space-separated values)
    * dict  -> recurse with ``parent-child`` flag names; lets configs
              group related options into a section, e.g.
              ``"mujoco": {"iterations": 100}`` -> ``--mujoco-iterations 100``
    * other -> ``--key <str(value)>``
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def list_configs(config_dir: Path) -> list[tuple[str, str]]:
    """Return sorted ``(name, description)`` for each ``*.json`` in ``config_dir``."""
    entries: list[tuple[str, str]] = []
    for path in sorted(config_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"warning: bad JSON in {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"warning: {path} top-level is not an object; skipping", file=sys.stderr)
            continue
        entries.append((path.stem, str(data.get("description", ""))))
    return entries


def value_to_cli(key: str, value: object) -> list[str]:
    """Encode a single ``(key, value)`` pair as an argparse CLI fragment."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"--{key}"] if value else [f"--no-{key}"]
    if isinstance(value, dict):
        # Nested section: recurse with ``parent-child`` flag names so the JSON
        # can group related options (e.g. all MuJoCo solver knobs) without
        # changing the example's flat argparse interface.
        out: list[str] = []
        for sub_key, sub_value in value.items():
            out.extend(value_to_cli(f"{key}-{sub_key}", sub_value))
        return out
    if isinstance(value, (list, tuple)):
        # Two regimes:
        # * **Vector of scalars** — Newton examples typically use
        #   ``nargs=N`` for these (positions, half-extents, etc.); send
        #   them as one flag followed by N space-separated values.
        # * **List of dicts / nested lists** — argparse can't natively
        #   round-trip a list of mappings via space separation (the
        #   default Python ``str(dict)`` isn't JSON), so we JSON-encode
        #   the whole list into a single ``--key '<json>'`` argument.
        #   The example then parses the string with ``json.loads``.
        # * **Empty list** — JSON-encode so the example receives "[]"
        #   rather than a bare flag with no argument.
        if not value or any(isinstance(v, (dict, list, tuple)) for v in value):
            return [f"--{key}", json.dumps(list(value))]
        return [f"--{key}", *(str(v) for v in value)]
    return [f"--{key}", str(value)]


def flatten_args(args: dict) -> dict:
    """Flatten nested-dict sections into ``parent-child`` keys.

    Lets the interactive editor and ``--set`` overrides see one flat keyspace
    while the JSON keeps human-readable sections.
    """
    out: dict = {}
    for key, value in args.items():
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_args(value).items():
                out[f"{key}-{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


def parse_scalar(raw: str, default: object) -> object:
    """Coerce ``raw`` to the same type as ``default`` when reasonable."""
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "y", "yes", "t", "true", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    if isinstance(default, list):
        # Allow JSON list syntax or comma-separated
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return [p.strip() for p in stripped.split(",") if p.strip()]
    return raw


def edit_interactive(args: dict) -> dict:
    """Walk the user through each key; Enter keeps the default."""
    if not args:
        print("(no parameters to edit)", file=sys.stderr)
        return args
    print("Edit parameters (Enter to keep default):", file=sys.stderr)
    result: dict = {}
    for key, default in args.items():
        if isinstance(default, bool):
            hint = "yes/no"
            shown = "yes" if default else "no"
        elif isinstance(default, list):
            hint = "comma-sep or JSON"
            shown = json.dumps(default)
        else:
            hint = ""
            shown = str(default) if default is not None else "null"
        prompt = f"  --{key} [{shown}]"
        if hint:
            prompt += f" ({hint})"
        prompt += ": "
        print(prompt, end="", file=sys.stderr, flush=True)
        try:
            line = input()
        except EOFError:
            line = ""
        if line == "":
            result[key] = default
        else:
            result[key] = parse_scalar(line, default)
    return result


def apply_overrides(args: dict, raw_sets: list[str]) -> dict:
    out = dict(args)
    for item in raw_sets:
        if "=" not in item:
            print(f"error: --set expects KEY=VAL, got {item!r}", file=sys.stderr)
            sys.exit(2)
        key, raw = item.split("=", 1)
        default = args.get(key)
        # If key isn't in the config, try JSON-parse the value; else coerce by default's type
        if default is None and key not in args:
            try:
                out[key] = json.loads(raw)
            except json.JSONDecodeError:
                out[key] = raw
        else:
            out[key] = parse_scalar(raw, default)
    return out


def resolve(path: Path, edit: bool, overrides: list[str]) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a JSON object")
    args = flatten_args(dict(data.get("args") or {}))
    args = apply_overrides(args, overrides)
    if edit:
        args = edit_interactive(args)
    cli: list[str] = []
    for key, value in args.items():
        cli += value_to_cli(key, value)
    return cli


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available example configs.")
    p_list.add_argument("--config-dir", type=Path, required=True)

    p_resolve = sub.add_parser("resolve", help="Resolve a config into CLI args.")
    p_resolve.add_argument("--config", type=Path, required=True)
    p_resolve.add_argument("--edit", action="store_true", help="Prompt to edit each parameter.")
    p_resolve.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Override a parameter without the interactive prompt.",
    )

    ns = parser.parse_args()

    if ns.cmd == "list":
        if not ns.config_dir.is_dir():
            print(f"error: config dir not found: {ns.config_dir}", file=sys.stderr)
            return 2
        for name, desc in list_configs(ns.config_dir):
            print(f"{name}\t{desc}")
        return 0

    if ns.cmd == "resolve":
        if not ns.config.is_file():
            print(f"error: config file not found: {ns.config}", file=sys.stderr)
            return 2
        cli = resolve(ns.config, ns.edit, ns.set)
        print(" ".join(shlex.quote(tok) for tok in cli))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
