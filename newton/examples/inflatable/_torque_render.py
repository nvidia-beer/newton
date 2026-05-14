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

"""Torque-spring overlay for the inflatable chambered examples.

Wraps :meth:`newton.viewer.Viewer.log_lines` with a pair of persistent
``wp.vec3`` buffers that a gather kernel fills from the current
``particle_q`` each frame, so the overlay follows the deformed mesh.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp


@wp.kernel
def _gather_line_endpoints(
    particle_q: wp.array(dtype=wp.vec3),
    starts_idx: wp.array(dtype=int),
    ends_idx: wp.array(dtype=int),
    # output
    starts_out: wp.array(dtype=wp.vec3),
    ends_out: wp.array(dtype=wp.vec3),
):
    """Gather a subset of particles into start/end line-endpoint arrays."""
    tid = wp.tid()
    starts_out[tid] = particle_q[starts_idx[tid]]
    ends_out[tid] = particle_q[ends_idx[tid]]


class TorqueSpringLines:
    """Per-frame line overlay for bend-resisting (torque) springs.

    Constructed once at example init with the spring endpoint pairs and
    the per-spring rest-direction array. Springs with zero rest
    direction (no bend torque) are filtered out. Call :meth:`log` in
    each ``render()`` pass.

    Attributes:
        count: Number of active torque springs (zero-length overlay if
            ``count == 0``; :meth:`log` becomes a no-op).
    """

    def __init__(
        self,
        spring_pairs: Sequence[tuple[int, int]],
        spring_rest_direction: np.ndarray,
        device: wp.Device | str,
    ) -> None:
        srd = np.asarray(spring_rest_direction, dtype=np.float32)
        if srd.ndim != 2 or srd.shape[1] != 3 or srd.shape[0] != len(spring_pairs):
            raise ValueError(
                f"spring_rest_direction shape {srd.shape} != (len(spring_pairs)={len(spring_pairs)}, 3)"
            )
        active = np.where(np.linalg.norm(srd, axis=1) > 0.5)[0]
        self.count = int(active.size)
        if self.count == 0:
            self._starts_idx = None
            self._ends_idx = None
            self._starts = None
            self._ends = None
            self._device = device
            return
        starts_np = np.fromiter(
            (int(spring_pairs[k][0]) for k in active), dtype=np.int32, count=self.count
        )
        ends_np = np.fromiter(
            (int(spring_pairs[k][1]) for k in active), dtype=np.int32, count=self.count
        )
        self._device = device
        self._starts_idx = wp.array(starts_np, dtype=wp.int32, device=device)
        self._ends_idx = wp.array(ends_np, dtype=wp.int32, device=device)
        self._starts = wp.zeros(self.count, dtype=wp.vec3, device=device)
        self._ends = wp.zeros(self.count, dtype=wp.vec3, device=device)

    def log(
        self,
        viewer,
        particle_q: wp.array,
        *,
        name: str = "torque_springs",
        color: tuple[float, float, float] = (1.0, 0.5, 0.0),
        width: float = 0.003,
        hidden: bool = False,
    ) -> None:
        """Refresh endpoints from ``particle_q`` and submit to the viewer.

        A no-op when there are no active torque springs. When ``hidden``
        is True the gather kernel is skipped so the checkbox can be
        toggled off at zero GPU cost.
        """
        if self.count == 0:
            return
        if not hidden:
            wp.launch(
                _gather_line_endpoints,
                dim=self.count,
                inputs=[particle_q, self._starts_idx, self._ends_idx],
                outputs=[self._starts, self._ends],
                device=self._device,
            )
        viewer.log_lines(name, self._starts, self._ends, color, width=width, hidden=hidden)
