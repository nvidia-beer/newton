# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ANCF launcher configuration checks."""

import contextlib
import io
import json
import shlex
import unittest
from unittest.mock import patch

from newton.tests.ancf_example_probe import ROOT, load_launcher


class TestANCFLauncherConfig(unittest.TestCase):
    def test_ancf_configs_use_the_solver_budget(self):
        tested = []
        for path in sorted((ROOT / "docker/config").glob("0*.json")):
            config = json.loads(path.read_text())
            if "substeps" not in config.get("args", {}):
                continue
            tested.append(path.name)
            with self.subTest(config=path.name):
                self.assertEqual(config["args"]["substeps"], 6)
        self.assertEqual(len(tested), 6, tested)

    def test_production_launcher_preserves_raw_arguments(self):
        helper = load_launcher()
        config = ROOT / "docker/config/07_vehicle_telemetry.json"
        stdout = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "run-example.py",
                    "resolve",
                    "--config",
                    str(config),
                    "--raw-args",
                    "--num-frames",
                    "12",
                    "--log",
                    "a path/replay.npz",
                ],
            ),
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(helper.main(), 0)
        cli = shlex.split(stdout.getvalue())
        self.assertEqual(cli[-4:], ["--num-frames", "12", "--log", "a path/replay.npz"])


if __name__ == "__main__":
    unittest.main()
