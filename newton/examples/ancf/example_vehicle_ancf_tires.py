# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""A USD vehicle on four inflatable ANCF tires on flat ground.

The shared runtime is in ``_vehicle_simulation``; acceptance checks are in
``newton.tests.ancf_vehicle_checks``. Existing launcher options are unchanged.
"""

import newton.examples
from newton.examples.ancf._vehicle_simulation import VehicleSimulation


class Example(VehicleSimulation):
    """Runnable example using the shared vehicle runtime."""

    # The runner requires these hooks; test code is loaded only in test mode.
    def test_post_step(self) -> None:
        from newton.tests.ancf_vehicle_checks import check_vehicle_step  # noqa: PLC0415

        check_vehicle_step(self)

    def test_final(self) -> None:
        from newton.tests.ancf_vehicle_checks import check_vehicle_final  # noqa: PLC0415

        check_vehicle_final(self)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
