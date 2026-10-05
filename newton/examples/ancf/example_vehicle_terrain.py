# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""A USD vehicle with ANCF tires driving over a terrain asset.

The shared runtime is in ``_vehicle_terrain``; acceptance checks are in
``newton.tests.ancf_vehicle_checks``. Existing launcher options are unchanged.
"""

import newton.examples
from newton.examples.ancf._vehicle_terrain import VehicleTerrain


class Example(VehicleTerrain):
    """Runnable example using the shared vehicle runtime."""

    # The runner requires these hooks; test code is loaded only in test mode.
    def test_post_step(self) -> None:
        from newton.tests.ancf_vehicle_checks import check_vehicle_step  # noqa: PLC0415

        check_vehicle_step(self)

    def test_final(self) -> None:
        from newton.tests.ancf_vehicle_checks import check_terrain_final  # noqa: PLC0415

        check_terrain_final(self)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
