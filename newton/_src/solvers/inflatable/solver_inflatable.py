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

from newton._src.solvers.soft import SolverSoft
from .kernels_inflatable import (
    scale_spring_rest_lengths_kernel,
    scale_tet_poses_kernel,
    compute_volume_kernel,
)

from newton._src.sim import Contacts, Control, Model, State


class SolverInflatable(SolverSoft):
    """
    Inflatable soft body solver with pressure control via rest configuration scaling.
    
    Extends SolverSoft with the ability to inflate/deflate soft bodies by scaling
    their FEM rest configuration. The material naturally deforms toward the new
    rest state, providing stable inflation behavior.
    
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
    ):
        super().__init__(
            model=model,
            dt=dt,
            mass=mass,
            preconditioner_type=preconditioner_type,
            solver_type=solver_type,
        )
        
        self.max_volume_ratio = max_volume_ratio
        self.current_pressure = 1.0  # Current pressure (rest config ratio)
        self.target_pressure = 1.0   # Target pressure for PID control
        
        # Store original rest configuration for scaling
        self._store_original_rest_config(model)
        
        # Allocate array for volume computation
        if model.tet_count > 0:
            self.tet_volumes = wp.zeros(model.tet_count, dtype=wp.float32, device=model.device)
        else:
            self.tet_volumes = None
        
        # Compute and store initial volume
        self._initial_volume = None
    
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
            self._initial_volume = self.compute_volume(state)
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
            Current volume / initial volume
        """
        initial = self.get_initial_volume(state)
        if initial <= 0.0:
            return 1.0
        return self.compute_volume(state) / initial
    
    def set_pressure(self, pressure: float):
        """
        Set the inflation pressure (volume ratio target).
        
        The pressure value represents the target volume ratio:
        - pressure=1.0: original size (no inflation)
        - pressure=2.0: target volume is 2x original
        
        The FEM rest configuration is scaled so the material naturally
        deforms toward the target volume.
        
        Parameters
        ----------
        pressure : float
            Target volume ratio (clamped to [1.0, max_volume_ratio])
        """
        model = self.model
        
        pressure = float(np.clip(pressure, 1.0, self.max_volume_ratio))
        
        # Skip if pressure hasn't changed
        if abs(pressure - self.current_pressure) < 1e-6:
            return
        
        self.current_pressure = pressure
        
        # For volume scaling, use cbrt(pressure) for linear dimensions (3D)
        # Volume scales as length^3, so length scales as volume^(1/3)
        linear_scale = np.cbrt(pressure)
        
        # Scale spring rest lengths
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
        
        # Scale tetrahedra rest poses (Dm_inv)
        # The rest pose is the inverse of the rest shape matrix Dm.
        # To scale the rest shape by 's', we scale Dm by 's', so Dm_inv scales by '1/s'.
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
        
        return {
            'initial_volume': initial,
            'current_volume': current,
            'max_volume': initial * self.max_volume_ratio,
            'current_ratio': current / initial if initial > 0 else 1.0,
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
