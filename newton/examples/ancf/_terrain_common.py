# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Helpers shared by the ANCF terrain examples (``vehicle_terrain``, ``vehicle_telemetry``) and the
Isaac Lab vehicle tasks.

Heightmap sampling, rock clearance under a posed vehicle, chassis-pose readers, vehicle-asset
loading, the follow camera, the terrain-contact spike kernel and the terrain mesh; the
terrain loader itself (grid + reference track) is :class:`newton.examples.ancf._field_task.FieldTerrain`,
which exposes the ``cell`` / ``hx`` / ``hy`` / ``origin`` attributes the sampling functions below read.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton.examples.ancf._vehicle_usd import VehicleUSD, asset_path

FAR_BELOW = -1.0e3  # [m] parks the base example's flat analytic plane and the MJCF worldbody plane
SPAWN_CLEARANCE = 0.05  # [m] drop height of the tire bottoms above the terrain under the vehicle


# ── Heightmap sampling (grid: any object with .cell/.hx/.hy/.origin, e.g. FieldTerrain) ──


def triangulated_sample(grid, values: np.ndarray, x: float, y: float) -> float:
    """Height on the same cell diagonal as :func:`grid_triangles` and MuJoCo."""
    nrow, ncol = values.shape
    fx = min(max((x - grid.origin[0] + grid.hx) / grid.cell, 0.0), ncol - 1 - 1e-6)
    fy = min(max((y - grid.origin[1] + grid.hy) / grid.cell, 0.0), nrow - 1 - 1e-6)
    c, r = int(fx), int(fy)
    tx, ty = fx - c, fy - r
    a00, a10 = float(values[r, c]), float(values[r, c + 1])
    a01, a11 = float(values[r + 1, c]), float(values[r + 1, c + 1])
    if tx >= ty:
        return a00 + tx * (a10 - a00) + ty * (a11 - a10)
    return a00 + tx * (a11 - a01) + ty * (a01 - a00)


def max_height_in_discs(grid, heights: np.ndarray, centres_xy: np.ndarray, radius: float) -> float:
    """Highest grid point under any of the discs ``(K, 2)`` [world m] of ``radius`` [m]: what a
    dropped vehicle's tires must clear."""
    best = -math.inf
    k = int(math.ceil(radius / grid.cell)) + 1
    nrow, ncol = heights.shape
    for cx, cy in np.asarray(centres_xy, dtype=np.float64) - np.array(grid.origin):
        c0 = int(round((cx + grid.hx) / grid.cell))
        r0 = int(round((cy + grid.hy) / grid.cell))
        cs = slice(max(c0 - k, 0), min(c0 + k + 1, ncol))
        rs = slice(max(r0 - k, 0), min(r0 + k + 1, nrow))
        if cs.start >= cs.stop or rs.start >= rs.stop:
            continue
        xs = -grid.hx + np.arange(cs.start, cs.stop) * grid.cell
        ys = -grid.hy + np.arange(rs.start, rs.stop) * grid.cell
        xx, yy = np.meshgrid(xs, ys)
        win = heights[rs, cs]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        if inside.any():
            best = max(best, float(win[inside].max()))
    return best if math.isfinite(best) else 0.0


def max_height_in_rect(grid, heights: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    """Highest grid point inside the world rectangle ``[x0, x1] x [y0, y1]`` [m]."""
    nrow, ncol = heights.shape
    c0 = max(int(math.floor((x0 - grid.origin[0] + grid.hx) / grid.cell)), 0)
    c1 = min(int(math.ceil((x1 - grid.origin[0] + grid.hx) / grid.cell)) + 1, ncol)
    r0 = max(int(math.floor((y0 - grid.origin[1] + grid.hy) / grid.cell)), 0)
    r1 = min(int(math.ceil((y1 - grid.origin[1] + grid.hy) / grid.cell)) + 1, nrow)
    if c0 >= c1 or r0 >= r1:
        return 0.0
    return float(heights[r0:r1, c0:c1].max())


def pose_xy(pose: tuple[float, float, float], pts: np.ndarray) -> np.ndarray:
    """Vehicle-frame plan-view points (K, 2) placed at ``pose = (x, y, yaw)`` [m, m, rad]."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]], axis=1)


def rock_clearance_height(
    grid, heights: np.ndarray, vehicle: VehicleUSD, pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> float:
    """Highest rock under the vehicle's tire discs, or under its chassis/suspension footprint
    minus belly clearance (rocks between the wheels can stand higher than the ones under the
    tires), with the vehicle at ``pose = (x, y, yaw)`` — add :data:`SPAWN_CLEARANCE` for the
    height to actually spawn/drop at."""
    spindle_xy = pose_xy(pose, vehicle.spindle_positions_zu()[:, :2])
    tires = max_height_in_discs(grid, heights, spindle_xy, float(vehicle.r_roll) + grid.cell)
    x0, x1, y0, y1 = vehicle.footprint
    corners = pose_xy(pose, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])  # bounding rect of the yawed footprint
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    body = max_height_in_rect(grid, heights, lo[0] - grid.cell, hi[0] + grid.cell, lo[1] - grid.cell, hi[1] + grid.cell)
    return max(tires, body - vehicle.belly)


# ── Vehicle asset / chassis telemetry ───────────────────────────────────────────


def require_vehicle_asset(args) -> VehicleUSD:
    """Load ``--vehicle-asset`` before the base class builds the model (needed for its footprint
    to compute a terrain-aware spawn height); the base class re-uses ``self.vehicle`` if already set."""
    vehicle_asset = getattr(args, "vehicle_asset", None)
    if not vehicle_asset:
        raise ValueError("--vehicle-asset is required (a vehicle USD baked by newton-tire-tool)")
    return VehicleUSD(asset_path(vehicle_asset))


def alloc_chassis_buffers(device: str) -> tuple[wp.array, wp.array]:
    """``(body_q, body_qd)`` single-body readback buffers for :func:`read_chassis_q`/:func:`read_chassis_qd`."""
    return wp.zeros(1, dtype=wp.transform, device=device), wp.zeros(1, dtype=wp.spatial_vector, device=device)


def read_chassis_q(state, chassis_body: int, q_buf: wp.array) -> np.ndarray:
    """Chassis transform row ``(px, py, pz, qx, qy, qz, qw)``, host-side."""
    wp.copy(q_buf, state.body_q, src_offset=chassis_body, count=1)
    return q_buf.numpy()[0]


def read_chassis_qd(state, chassis_body: int, qd_buf: wp.array) -> np.ndarray:
    """Chassis spatial velocity row ``(vx, vy, vz, wx, wy, wz)`` (Newton: linear first), host-side."""
    wp.copy(qd_buf, state.body_qd, src_offset=chassis_body, count=1)
    return qd_buf.numpy()[0]


def wheel_speed_rate(mu: float, r_roll: float, gravity: float, frame_dt: float, floor: float) -> float:
    """Max wheel-speed ramp [rad/s per frame]: the traction limit ``mu g / r_roll`` [rad/s^2]
    converted to a per-frame step, floored at ``floor`` (a faster ramp only spins the wheels)."""
    return max(mu * gravity / r_roll * frame_dt, floor)


def park_mjcf_plane(car: newton.ModelBuilder) -> None:
    """Move the MJCF worldbody's flat ground plane (the base example resets it to z = 0) out of
    the way — the terrain heightfield is the floor here, not the analytic plane."""
    for s in range(car.shape_count):
        if car.shape_body[s] == -1 and car.shape_type[s] == newton.GeoType.PLANE:
            xf = car.shape_transform[s]
            car.shape_transform[s] = wp.transform(wp.vec3(xf.p[0], xf.p[1], FAR_BELOW), xf.q)


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


# ── Terrain mesh for the viewer (vertices per grid node) ─────────────


@wp.kernel
def terrain_mesh_points(
    h: wp.array2d[float], w: float, hx: float, hy: float, cell: float, ox: float, oy: float, pts: wp.array[wp.vec3]
):
    """Grid node (r, c) -> world vertex at stage w, grid centre at (ox, oy)."""
    r, c = wp.tid()
    ncol = h.shape[1]
    pts[r * ncol + c] = wp.vec3(ox - hx + float(c) * cell, oy - hy + float(r) * cell, w * h[r, c])


def grid_triangles(nrow: int, ncol: int, cell_mask: np.ndarray | None = None) -> np.ndarray:
    """Two triangles per grid cell over vertices ``r * ncol + c``, flattened int32.

    ``cell_mask`` (``(nrow - 1, ncol - 1)``, truthy = keep) drops the cells it excludes.
    """
    r, c = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
    i0 = (r * ncol + c).ravel()
    i1 = i0 + 1
    i2 = i0 + ncol
    i3 = i2 + 1
    if cell_mask is not None:
        keep = np.asarray(cell_mask, dtype=bool).ravel()
        i0, i1, i2, i3 = i0[keep], i1[keep], i2[keep], i3[keep]
    return (
        np.concatenate([np.stack([i0, i1, i3], axis=1), np.stack([i0, i3, i2], axis=1)], axis=0)
        .flatten()
        .astype(np.int32)
    )
