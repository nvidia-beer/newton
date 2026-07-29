# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SE(3) screw-motion tracker: co-move a particle cloud with a rigid body.

Applies the incremental screw motion ``p' = R_rel · p + t_rel`` where::

    q_rel = q_new ⊗ conj(q_old)
    t_rel = T_new − rot(q_rel, T_old)

Two update paths:

* :meth:`~SE3Tracker.update` — NumPy in-place (CPU arrays / testing).
* :meth:`~SE3Tracker.update_device` — Warp kernel in-place, no D↔H copy.
"""

from __future__ import annotations

import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _se3_transform_kernel(
    pts: wp.array[wp.vec3],
    q_rel: wp.quat,
    t_rel: wp.vec3,
):
    i = wp.tid()
    pts[i] = wp.quat_rotate(q_rel, pts[i]) + t_rel


# ---------------------------------------------------------------------------
# SE3Tracker
# ---------------------------------------------------------------------------


class SE3Tracker:
    """Track an SE(3) pose and rigidly co-move a particle cloud.

    Args:
        pos:  Initial body position [3].
        quat: Initial body quaternion ``[qw, qx, qy, qz]``.
        tol:  Motion threshold below which updates are skipped.
    """

    def __init__(self, pos: np.ndarray, quat: np.ndarray, tol: float = 1e-6) -> None:
        self._pos = np.asarray(pos, dtype=np.float64).copy()
        self._quat = np.asarray(quat, dtype=np.float64).copy()
        self.tol = float(tol)

    def update(self, pts: np.ndarray, new_pos: np.ndarray, new_quat: np.ndarray) -> bool:
        """Apply SE(3) delta to ``pts`` in-place (NumPy path).

        Args:
            pts:      ``(N, 3)`` float32 array modified in-place.
            new_pos:  New body position ``[3]``.
            new_quat: New body quaternion ``[qw, qx, qy, qz]``.

        Returns:
            ``True`` if the transform was applied, ``False`` if skipped.
        """
        T_new = np.asarray(new_pos, dtype=np.float64)
        q_new = np.asarray(new_quat, dtype=np.float64)
        q_rel, t_rel, changed = self._delta(T_new, q_new)
        if changed:
            p64 = pts.astype(np.float64)
            pts[:] = (_rotate_batch(q_rel, p64) + t_rel).astype(pts.dtype)
        self._pos = T_new
        self._quat = q_new
        return changed

    def update_device(
        self,
        pts: wp.array[wp.vec3],
        new_pos: np.ndarray,
        new_quat: np.ndarray,
        device: str,
    ) -> bool:
        """Apply SE(3) delta to a Warp device array in-place (no D↔H copy).

        Args:
            pts:      ``[N]`` ``vec3`` Warp array on ``device``, modified in-place.
            new_pos:  New body position ``[3]``.
            new_quat: New body quaternion ``[qw, qx, qy, qz]``.
            device:   Warp device string.

        Returns:
            ``True`` if the transform was applied, ``False`` if skipped.
        """
        T_new = np.asarray(new_pos, dtype=np.float64)
        q_new = np.asarray(new_quat, dtype=np.float64)
        q_rel, t_rel, changed = self._delta(T_new, q_new)
        if changed:
            qw, qx, qy, qz = q_rel  # our [qw,qx,qy,qz] → Warp (x,y,z,w)
            wp.launch(
                _se3_transform_kernel,
                dim=pts.shape[0],
                inputs=[
                    pts,
                    wp.quat(float(qx), float(qy), float(qz), float(qw)),
                    wp.vec3(float(t_rel[0]), float(t_rel[1]), float(t_rel[2])),
                ],
                device=device,
            )
        self._pos = T_new
        self._quat = q_new
        return changed

    def _delta(self, T_new: np.ndarray, q_new: np.ndarray):
        q_rel = _quat_mul(q_new, _quat_conj(self._quat))
        t_rel = T_new - _rotate(q_rel, self._pos)
        changed = np.linalg.norm(t_rel) > self.tol or abs(q_rel[0]) < 1.0 - self.tol
        return q_rel, t_rel, changed


# ---------------------------------------------------------------------------
# Public math helpers
# ---------------------------------------------------------------------------


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Rotation vector (axis × angle) [rad] → unit quaternion ``[qw, qx, qy, qz]``."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.asarray(v, dtype=np.float64) / angle
    s = np.sin(angle * 0.5)
    return np.array([np.cos(angle * 0.5), s * axis[0], s * axis[1], s * axis[2]])


def quat_compose(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p ⊗ q`` for ``[qw, qx, qy, qz]`` quaternions."""
    return _quat_mul(p, q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` [3] by unit quaternion ``q = [qw, qx, qy, qz]``."""
    return _rotate(q, v)


# ---------------------------------------------------------------------------
# Private math
# ---------------------------------------------------------------------------


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    vv = np.asarray(v, dtype=np.float64)
    u = np.array([qx, qy, qz])
    return vv + 2.0 * qw * np.cross(u, vv) + 2.0 * np.cross(u, np.cross(u, vv))


def _rotate_batch(q: np.ndarray, pts: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    v = np.array([qx, qy, qz])
    c1 = np.cross(v, pts)
    return pts + 2.0 * qw * c1 + 2.0 * np.cross(v, c1)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles [rad] → unit quaternion ``[qw, qx, qy, qz]``."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz], dtype=np.float64)
