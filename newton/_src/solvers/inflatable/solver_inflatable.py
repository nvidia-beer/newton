# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""
Inflatable soft body solver with pressure control.

Extends SolverSoft with inflation capabilities via FEM rest configuration scaling.
The inflation mechanism works by scaling the rest configuration (rest lengths, rest poses),
and the FEM material naturally deforms to reach its new rest state.

This approach is adapted from 2D balloon inflation (triangle FEM) to 3D (tetrahedra FEM).

INFLATION MECHANISM:
===================
Instead of applying external pressure forces, we scale the FEM rest configuration:
1. Store original rest configuration (tet_poses, spring_rest_length)
2. For pressure p (volume ratio), scale linear dimensions by cbrt(p) (cube root for 3D)
3. Scale rest poses (Dm_inv) by 1/cbrt(p)
4. Scale spring rest lengths by cbrt(p)
5. The FEM naturally drives the mesh toward the new rest state

This is stable and integrates well with implicit solvers.
"""

import numpy as np
import warp as wp

from newton._src.solvers.deformable import SolverDeformable
from .kernels_bend import eval_springs_linear_and_torque
from .kernels_inflatable import (
    scale_spring_rest_lengths_kernel,
    scale_tet_poses_kernel,
    scale_tet_poses_per_chamber_kernel,
    scale_spring_rest_lengths_per_chamber_kernel,
    compute_volume_kernel,
)

from newton._src.sim import Contacts, Control, Model, State


class SolverInflatable(SolverDeformable):
    """
    Inflatable soft body solver with pressure control via rest configuration scaling.
    
    Extends SolverDeformable with the ability to inflate/deflate soft bodies by scaling
    their FEM rest configuration. The material naturally deforms toward the new
    rest state, providing stable inflation behavior.
    
    Modes
    -----
    - **Single pressure**: Call ``set_pressure(p)`` for the whole body.
    - **Multi-chamber**: Call ``set_chamber_mask(tet_mask, spring_mask, num_chambers)`` once,
      then ``set_chamber_pressures([p0, p1, ...])`` to set per-chamber pressures. Use for
      bending actuators (e.g. two chambers side-by-side with different pressures).
    
    Parameters
    ----------
    model : Model
        The Newton physics model
    dt : float
        Physics timestep (default: 1/60)
    mass : float
        Particle mass (default: 1.0)
    max_volume_ratio : float
        Maximum inflation ratio (default: 3.0, meaning 3x original volume)
    preconditioner_type : str
        Preconditioner type: "id", "diag", or "diag_abs" (default: "id")
    solver_type : str
        Linear solver: "bicgstab", "cg", "gmres", or "cr" (default: "bicgstab")
    
    Example
    -------
    >>> solver = SolverInflatable(model, dt=1/60, max_volume_ratio=2.0)
    >>> solver.set_pressure(1.5)  # Inflate to 1.5x original volume
    >>> solver.step(state_in, state_out, control, contacts, dt)
    """

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        max_volume_ratio: float = 3.0,
        preconditioner_type: str = "id",
        solver_type: str = "bicgstab",
        use_constraint_contacts: bool = False,
        contact_relaxation: float = 0.5,
        contact_max_velocity: float = 20.0,
        contact_max_correction: float = 0.02,
        contact_iterations: int = 2,
        linear_solver_maxiter: int = 50,
        handle_self_contact: bool = False,
        self_contact_radius: float = 0.02,
        self_contact_stiffness: float = 1.0e5,
        self_contact_force_cap: float = 2.0,
        self_contact_edge_edge: bool = True,
        torque_stiffness: float = 0.0,
        torque_damping: float = 0.0,
        spring_rest_direction: "np.ndarray | None" = None,
        ground_plane: "tuple[float, float, float, float] | None" = None,
        ground_ke: float = 1.0e5,
        ground_kd: float = 1.0e2,
        ground_kf: float = 1.0e3,
        ground_mu: float = 0.5,
        extra_matrix_blocks: int | None = None,
    ):
        super().__init__(
            model=model,
            dt=dt,
            mass=mass,
            preconditioner_type=preconditioner_type,
            solver_type=solver_type,
            use_constraint_contacts=use_constraint_contacts,
            contact_relaxation=contact_relaxation,
            contact_max_velocity=contact_max_velocity,
            contact_max_correction=contact_max_correction,
            contact_iterations=contact_iterations,
            linear_solver_maxiter=linear_solver_maxiter,
            ground_plane=ground_plane,
            ground_ke=ground_ke,
            ground_kd=ground_kd,
            ground_kf=ground_kf,
            ground_mu=ground_mu,
            handle_self_contact=handle_self_contact,
            self_contact_radius=self_contact_radius,
            self_contact_stiffness=self_contact_stiffness,
            self_contact_force_cap=self_contact_force_cap,
            self_contact_edge_edge=self_contact_edge_edge,
            extra_matrix_blocks=extra_matrix_blocks,
        )
        
        self.max_volume_ratio = max_volume_ratio
        self.current_pressure = 1.0  # Current pressure (rest config ratio)
        self.target_pressure = 1.0   # Target pressure for PID control
        # Per-chamber: optional masks and pressures (chambers are spatially separate regions)
        self.tet_chamber_mask = None  # wp.array(dtype=int), length tet_count
        self.spring_chamber_mask = None  # wp.array(dtype=int), length spring_count
        self.num_chambers = 0
        self._chamber_pressures_array = None  # wp.array(dtype=float), length num_chambers
        
        # Store original rest configuration for scaling
        self._store_original_rest_config(model)
        
        # Allocate array for volume computation
        if model.tet_count > 0:
            self.tet_volumes = wp.zeros(model.tet_count, dtype=wp.float32, device=model.device)
        else:
            self.tet_volumes = None
        
        # Compute and store initial volume
        self._initial_volume = None

        # Optional bend/torque: springs with non-zero rest direction resist bending
        self.torque_stiffness = float(torque_stiffness)
        self.torque_damping = float(torque_damping)
        dr = np.asarray(
            spring_rest_direction
            if spring_rest_direction is not None
            else np.zeros((model.spring_count, 3), dtype=np.float32),
            dtype=np.float32,
        )
        if dr.shape != (model.spring_count, 3):
            raise ValueError(
                f"spring_rest_direction shape {dr.shape} != (spring_count={model.spring_count}, 3)"
            )
        self.spring_rest_direction = wp.array(dr, dtype=wp.vec3, device=model.device)

    def eval_spring_forces(self, model: Model, state: State):
        """Spring forces; with torque when torque_stiffness > 0 and rest directions set."""
        if model.spring_count == 0:
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if self.torque_stiffness > 0.0:
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
        return super().eval_spring_forces(model, state)
    
    def _store_original_rest_config(self, model: Model):
        """Store the original rest configuration for later scaling."""
        # Store original tetrahedra rest poses
        if model.tet_count > 0:
            self.original_tet_poses = wp.clone(model.tet_poses)
        else:
            self.original_tet_poses = None
        
        # Store original spring rest lengths
        if model.spring_count > 0:
            self.original_spring_rest_length = wp.clone(model.spring_rest_length)
        else:
            self.original_spring_rest_length = None
    
    def compute_volume(self, state: State) -> float:
        """
        Compute the current volume of the soft body.
        
        Sums up the volumes of all tetrahedra based on current particle positions.
        
        Parameters
        ----------
        state : State
            Current simulation state
            
        Returns
        -------
        float
            Total volume of all tetrahedra
        """
        model = self.model
        
        if model.tet_count == 0 or self.tet_volumes is None:
            return 0.0
        
        wp.launch(
            kernel=compute_volume_kernel,
            dim=model.tet_count,
            inputs=[
                state.particle_q,
                model.tet_indices,
            ],
            outputs=[self.tet_volumes],
            device=model.device,
        )
        
        return float(np.sum(self.tet_volumes.numpy()))
    
    def get_initial_volume(self, state: State) -> float:
        """
        Get the initial volume (computed once and cached).
        
        Parameters
        ----------
        state : State
            Simulation state (used to compute initial volume on first call)
            
        Returns
        -------
        float
            Initial volume of the soft body
        """
        if self._initial_volume is None:
            vol = self.compute_volume(state)
            # Only cache if sensible (avoids caching 0 or garbage from uninitialized state)
            if np.isfinite(vol) and vol > 1.0e-12:
                self._initial_volume = vol
            else:
                return 1.0  # fallback so ratio = current/1.0 and can be clamped
        return self._initial_volume
    
    def get_volume_ratio(self, state: State) -> float:
        """
        Get the current volume ratio (current / initial).
        
        Parameters
        ----------
        state : State
            Current simulation state
            
        Returns
        -------
        float
            Current volume / initial volume (clamped to sane range)
        """
        initial = self.get_initial_volume(state)
        # Guard against zero or denormal initial (uninitialized or bad state)
        if initial <= 1.0e-12:
            return 1.0
        current = self.compute_volume(state)
        if not np.isfinite(current) or current <= 0.0:
            return 1.0
        ratio = current / initial
        if not np.isfinite(ratio):
            return 1.0
        # Clamp to sane range to avoid garbage (e.g. 1e18 from bad initial)
        ratio = float(np.clip(ratio, 0.0, max(10.0, self.max_volume_ratio * 2.0)))
        return ratio
    
    def set_chamber_mask(
        self,
        tet_chamber_mask: "wp.array",
        spring_chamber_mask: "wp.array | None" = None,
        num_chambers: int = 0,
    ):
        """
        Set per-chamber masks so different regions can have different pressures.
        Chambers are spatially separate (e.g. slices along height Z).
        Call set_chamber_pressures([p0, p1, ...]) to apply pressures.
        """
        self.tet_chamber_mask = tet_chamber_mask
        self.spring_chamber_mask = spring_chamber_mask
        self.num_chambers = int(num_chambers)
        if self.num_chambers > 0:
            self._chamber_pressures_array = wp.array(
                np.full(self.num_chambers, 1.0, dtype=np.float32),
                dtype=wp.float32,
                device=self.model.device,
            )

    def set_chamber_pressures(self, pressure_list: list[float]):
        """
        Set pressure (volume ratio) per chamber. Use when tet_chamber_mask is set.
        pressure_list[i] is the pressure for chamber i (clamped to [1.0, max_volume_ratio]).
        """
        if self.tet_chamber_mask is None or self.num_chambers <= 0:
            return
        model = self.model
        pressures = np.array(
            [float(np.clip(p, 1.0, self.max_volume_ratio)) for p in pressure_list],
            dtype=np.float32,
        )
        if len(pressures) < self.num_chambers:
            pressures = np.resize(pressures, self.num_chambers)
            pressures[len(pressure_list) :] = 1.0
        pressures = pressures[: self.num_chambers]
        self._chamber_pressures_array.assign(pressures)
        # Update spring rest lengths: per-chamber if mask set, else single scale from mean pressure
        if model.spring_count > 0 and self.original_spring_rest_length is not None:
            if self.spring_chamber_mask is not None:
                wp.launch(
                    kernel=scale_spring_rest_lengths_per_chamber_kernel,
                    dim=model.spring_count,
                    inputs=[
                        self.original_spring_rest_length,
                        self.spring_chamber_mask,
                        self._chamber_pressures_array,
                        self.num_chambers,
                    ],
                    outputs=[model.spring_rest_length],
                    device=model.device,
                )
            else:
                # No spring mask: use mean pressure for isotropic spring scaling
                avg_p = float(np.cbrt(np.mean(pressures)))
                wp.launch(
                    kernel=scale_spring_rest_lengths_kernel,
                    dim=model.spring_count,
                    inputs=[self.original_spring_rest_length, avg_p],
                    outputs=[model.spring_rest_length],
                    device=model.device,
                )
        # Update tet rest poses: per-chamber pressure (isotropic)
        if model.tet_count > 0 and self.original_tet_poses is not None:
            wp.launch(
                kernel=scale_tet_poses_per_chamber_kernel,
                dim=model.tet_count,
                inputs=[
                    self.original_tet_poses,
                    self.tet_chamber_mask,
                    self._chamber_pressures_array,
                    self.num_chambers,
                ],
                outputs=[model.tet_poses],
                device=model.device,
            )

    def set_pressure(self, pressure: float):
        """
        Set the inflation pressure (volume ratio target) for single-pressure mode.
        When tet_chamber_mask is set, use set_chamber_pressures([p0, p1, ...]) instead.
        
        Parameters
        ----------
        pressure : float
            Target volume ratio (clamped to [1.0, max_volume_ratio])
        """
        # If chambers are set, apply same pressure to all chambers and return
        if self.tet_chamber_mask is not None and self.num_chambers > 0:
            self.set_chamber_pressures([pressure] * self.num_chambers)
            self.current_pressure = pressure
            return
        model = self.model
        
        pressure = float(np.clip(pressure, 1.0, self.max_volume_ratio))
        
        # Skip if pressure hasn't changed
        if abs(pressure - self.current_pressure) < 1e-6:
            return
        
        self.current_pressure = pressure
        
        # For volume scaling, use cbrt(pressure) for linear dimensions (3D)
        linear_scale = np.cbrt(pressure)
        
        # Scale spring rest lengths (always isotropic)
        if model.spring_count > 0 and self.original_spring_rest_length is not None:
            wp.launch(
                kernel=scale_spring_rest_lengths_kernel,
                dim=model.spring_count,
                inputs=[
                    self.original_spring_rest_length,
                    linear_scale,
                ],
                outputs=[model.spring_rest_length],
                device=model.device,
            )
        
        # Scale tetrahedra rest poses (Dm_inv) - isotropic
        if model.tet_count > 0 and self.original_tet_poses is not None:
            wp.launch(
                kernel=scale_tet_poses_kernel,
                dim=model.tet_count,
                inputs=[
                    self.original_tet_poses,
                    linear_scale,
                ],
                outputs=[model.tet_poses],
                device=model.device,
            )
    
    def set_target_pressure(self, target: float):
        """
        Set the target pressure for gradual inflation.
        
        Use apply_pressure_control() to gradually approach the target.
        
        Parameters
        ----------
        target : float
            Target pressure (volume ratio)
        """
        self.target_pressure = float(np.clip(target, 1.0, self.max_volume_ratio))
    
    def apply_pressure_control(self, rate: float = 0.1):
        """
        Apply pressure control to gradually approach target pressure.
        
        Smoothly interpolates current pressure toward target pressure.
        
        Parameters
        ----------
        rate : float
            Interpolation rate (0-1). Higher = faster convergence.
        """
        new_pressure = self.current_pressure + rate * (self.target_pressure - self.current_pressure)
        self.set_pressure(new_pressure)
    
    def get_inflation_info(self, state: State) -> dict:
        """
        Get comprehensive inflation information.
        
        Parameters
        ----------
        state : State
            Current simulation state
            
        Returns
        -------
        dict
            Dictionary containing:
            - initial_volume: Initial volume
            - current_volume: Current volume
            - max_volume: Maximum allowed volume
            - current_ratio: Current volume ratio
            - target_ratio: Target volume ratio (pressure)
            - max_ratio: Maximum volume ratio
            - pressure: Current pressure setting
        """
        initial = self.get_initial_volume(state)
        current = self.compute_volume(state)
        if not np.isfinite(current):
            current_ratio = float("nan")
        elif initial > 1.0e-12 and np.isfinite(initial):
            current_ratio = float(current / initial)
            if not np.isfinite(current_ratio):
                current_ratio = float("nan")
        else:
            current_ratio = 1.0
        
        return {
            'initial_volume': initial,
            'current_volume': current,
            'max_volume': initial * self.max_volume_ratio,
            'current_ratio': current_ratio,
            'target_ratio': self.target_pressure,
            'max_ratio': self.max_volume_ratio,
            'pressure': self.current_pressure,
        }
    
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """
        Simulate the model for a given time step.
        
        Same as SolverSoft.step(), but with inflation support.
        Call set_pressure() or apply_pressure_control() before stepping
        to change the inflation state.
        
        Parameters
        ----------
        state_in : State
            The input state
        state_out : State
            The output state
        control : Control
            The control input
        contacts : Contacts
            The contact information (generated by model.collide())
        dt : float
            The time step
            
        Returns
        -------
        State
            The output state
        """
        # Store initial volume on first step
        if self._initial_volume is None:
            self._initial_volume = self.compute_volume(state_in)
        
        # Call parent step
        return super().step(state_in, state_out, control, contacts, dt)
