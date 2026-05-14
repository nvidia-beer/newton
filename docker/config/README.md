# Example Configs (JSON)

Each JSON file here is **one runnable example**. The filename (without
`.json`) is the example name passed to
`python -m newton.examples <name>`.

`run-examples.sh` loads every `*.json` in this folder at startup, renders a
menu, and resolves the selected config's `args` block into CLI flags via
`../run-example.py`. Append `e` to your menu choice (e.g. `3e`) or pass
`-e` / `--edit` to interactively override individual parameters before the
run; without it, the example launches straight with the JSON values.

## File schema

All fields optional. Unknown fields are ignored.

```json
{
    "description": "Short one-liner shown in the menu.",
    "args": {
        "viewer":     "gl",
        "num-frames": 100,
        "headless":   false
    }
}
```

### Key → CLI mapping

Keys are **kebab-case**, matching Newton's argparse flags.

| JSON value            | CLI fragment                                      |
| --------------------- | ------------------------------------------------- |
| `"gl"` (string)       | `--key gl`                                        |
| `100` (number)        | `--key 100`                                       |
| `true`                | `--key`                                           |
| `false`               | `--no-key` (Newton uses `BooleanOptionalAction`)  |
| `null`                | *omitted* — argparse's default wins               |
| `[1, 2, 3]`           | `--key 1 2 3`  (single flag, `nargs=N`)           |

User-supplied args (either `--set KEY=VAL` or raw args after `--`) win
over the JSON defaults (argparse's last-value rule).

## Adding a new example

1. Pick the example name (everything after `example_` in the filename).
2. Drop a matching JSON file here (or regenerate the whole folder with
   `/tmp/extract_example_defaults.py` if you prefer to pull defaults
   straight from the `add_argument` calls).
3. The menu picks it up automatically on the next run.

A bare-minimum file works:

```json
{ "description": "…" }
```
