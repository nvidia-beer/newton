# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Bend solver: springs with a non-zero rest direction resist bending away from it.
"""

import numpy as np
import warp as wp

from newton._src.solvers.inflatable import SolverInflatable
from newton._src.sim import Model, State

from .kernels_bend import eval_springs_linear_and_torque


class SolverBend(SolverInflatable):
    """
    Inflatable solver + torque on selected springs. Pass spring_rest_direction (N,3):
    only springs with non-zero rest direction get torque (e.g. one axis-aligned set).
    """

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        max_volume_ratio: float = 3.0,
        preconditioner_type: str = "id",
        solver_type: str = "bicgstab",
        torque_stiffness: float = 100.0,
        torque_damping: float = 2.0,
        spring_rest_direction: "np.ndarray | None" = None,
        use_constraint_contacts: bool = False,
        contact_relaxation: float = 0.5,
        contact_max_velocity: float = 20.0,
        contact_max_correction: float = 0.02,
        contact_iterations: int = 2,
    ):
        super().__init__(
            model=model,
            dt=dt,
            mass=mass,
            max_volume_ratio=max_volume_ratio,
            preconditioner_type=preconditioner_type,
            solver_type=solver_type,
            use_constraint_contacts=use_constraint_contacts,
            contact_relaxation=contact_relaxation,
            contact_max_velocity=contact_max_velocity,
            contact_max_correction=contact_max_correction,
            contact_iterations=contact_iterations,
        )
        self.torque_stiffness = float(torque_stiffness)
        self.torque_damping = float(torque_damping)

        dr = np.asarray(
            spring_rest_direction if spring_rest_direction is not None else np.zeros((model.spring_count, 3), dtype=np.float32),
            dtype=np.float32,
        )
        if dr.shape != (model.spring_count, 3):
            raise ValueError(f"spring_rest_direction shape {dr.shape} != (spring_count={model.spring_count}, 3)")
        self.spring_rest_direction = wp.array(dr, dtype=wp.vec3, device=model.device)

    def eval_spring_forces(self, model: Model, state: State):
        if model.spring_count == 0:
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        f = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=eval_springs_linear_and_torque,
            dim=model.spring_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
                self.spring_rest_direction,
                wp.float32(self.torque_stiffness),
                wp.float32(self.torque_damping),
            ],
            outputs=[f],
            device=model.device,
        )
        return f
