# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Helpers shared by the ANCF terrain examples (``vehicle_track``, ``vehicle_field``).

Chassis-pose readers, the follow camera and the terrain-contact spike kernel; the terrain
loaders themselves live in :mod:`newton.examples.ancf._field_task` (field) and
``example_vehicle_track`` (track).
"""

from __future__ import annotations

import math

import warp as wp

FAR_BELOW = -1.0e3  # [m] parks the base example's flat analytic plane and the MJCF worldbody plane
SPAWN_CLEARANCE = 0.05  # [m] drop height of the tire bottoms above the terrain under the vehicle


def quat_yaw(q) -> float:
    """Yaw [rad] of a body transform row ``(px, py, pz, qx, qy, qz, qw)``."""
    x, y, z, w = float(q[3]), float(q[4]), float(q[5]), float(q[6])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_roll_pitch(q) -> tuple[float, float]:
    """Roll and pitch [rad] of a body transform row ``(px, py, pz, qx, qy, qz, qw)``."""
    x, y, z, w = float(q[3]), float(q[4]), float(q[5]), float(q[6])
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def set_follow_camera(viewer, pose) -> None:
    """Chase camera 9 m behind and 3.5 m above the chassis transform ``pose``, looking along its heading."""
    yaw = quat_yaw(pose)
    cam = wp.vec3(float(pose[0]) - 9.0 * math.cos(yaw), float(pose[1]) - 9.0 * math.sin(yaw), float(pose[2]) + 3.5)
    viewer.set_camera(pos=cam, pitch=-18.0, yaw=math.degrees(yaw))


@wp.kernel
def terrain_contact_spikes(
    node_f: wp.array[wp.vec3],  # per-node terrain force, Z-up
    particle_q: wp.array[wp.vec3],  # node positions already in Z-up
    scale: float,
    line_s: wp.array[wp.vec3],
    line_e: wp.array[wp.vec3],
):
    """Spike per node in contact: length = |f| * scale along the force direction."""
    i = wp.tid()
    p = particle_q[i]
    line_s[i] = p
    line_e[i] = p + node_f[i] * scale
