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
Inflatable Soft Body Example - Sphere

Demonstrates inflation of a 3D tetrahedral soft body sphere using the SolverInflatable.
The inflation mechanism works by scaling the FEM rest configuration,
causing the material to naturally deform toward its new rest state.

This is adapted from the 2D balloon inflation demo in soft_robotics/warp
to work with 3D tetrahedra in Newton.

Features:
- Inflatable sphere using TetraSphere generator
- Pressure control via rest configuration scaling
- Cycle through different inflation levels
- Real-time volume ratio tracking

Usage:
    python -m newton.examples.inflatable.example_inflatable_sphere
    python -m newton.examples.inflatable.example_inflatable_sphere --max_pressure 2.5 --cycle_speed 0.02
"""

import warp as wp
import numpy as np
import argparse
import math

import newton
from newton.solvers import SolverInflatable, TetraSphere


class Example:
    """
    Inflatable soft body demonstration.
    
    Creates a tetrahedral sphere mesh and inflates/deflates it
    using the SolverInflatable which scales the FEM rest configuration.
    """
    
    def __init__(
        self,
        viewer,
        radius: float = 0.3,
        subdivisions: int = 2,
        interior_layers: int = 2,
        initial_height: float = 0.5,
        mass: float = 1.0,
        k_mu: float = 1.0e5,         # Shear modulus (softer for visible deformation)
        k_lambda: float = 1.0e5,     # Bulk modulus
        k_damp: float = 1.0,         # Damping
        spring_ke: float = 5.0e4,    # Spring stiffness
        spring_kd: float = 1.0,      # Spring damping
        gravity: float = 9.81,
        max_pressure: float = 5.0,   # Maximum inflation (volume ratio)
        cycle_speed: float = 0.01,   # How fast to cycle inflation
        substeps: int = 5,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.radius = radius
        self.initial_height = initial_height
        self.mass = mass
        self.max_pressure = max_pressure
        self.cycle_speed = cycle_speed
        
        self.viewer = viewer
        
        # Generate FEM sphere mesh
        print(f"\n🎈 Generating tetrahedral sphere mesh...", flush=True)
        sphere = TetraSphere(
            radius=radius,
            subdivisions=subdivisions,
            interior_layers=interior_layers,
            verbose=True
        )
        mesh_data = sphere.get_mesh_data()
        
        vertices = mesh_data['vertices']
        indices = mesh_data['indices']
        tetrahedra = mesh_data['tetrahedra']
        
        print(f"   Mesh: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra", flush=True)
        
        # Build Newton model
        builder = newton.ModelBuilder()
        
        # Add bouncy ground plane at Z=0
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,   # High stiffness for bounce
                kd=1e3,   # Some damping
                kf=1e4,   # Friction stiffness
                mu=0.5    # Friction coefficient
            )
        )
        
        # Add soft mesh - positioned above ground (Z is up); builder adds edge springs
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, initial_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            scale=1.0,
            density=mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )
        
        self.model = builder.finalize()
        
        print(f"\nModel created:", flush=True)
        print(f"  Particles: {self.model.particle_count}", flush=True)
        print(f"  Springs: {self.model.spring_count}", flush=True)
        print(f"  Tetrahedra: {self.model.tet_count}", flush=True)
        
        # Set gravity (Z is up, gravity pulls down)
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        
        # Contact parameters
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 500.0
        self.model.soft_contact_kf = 5.0e4
        self.model.soft_contact_mu = 0.9
        
        # Particle constraint parameters
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        
        # Particle rendering radius
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, 0.008),
            dtype=wp.float32,
            device=self.model.device
        )
        
        # Create inflatable solver
        self.solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            mass=mass,
            max_volume_ratio=max_pressure,
            solver_type="bicgstab"
        )
        
        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        
        # Initialize FK
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0
        )
        
        # Set up viewer
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True
        
        # Inflation state - manual control only
        self.current_pressure = 1.0
        self.pressure_step = 0.1  # Pressure adjustment per key press
        
        # Register keyboard controls if viewer supports it
        if self.viewer:
            # Try renderer first (ViewerGL), then viewer directly
            if hasattr(self.viewer, 'renderer') and hasattr(self.viewer.renderer, 'register_key_press'):
                self.viewer.renderer.register_key_press(self._on_key_press)
                print(f"   [Keyboard controls registered on renderer]", flush=True)
            elif hasattr(self.viewer, 'register_key_press'):
                self.viewer.register_key_press(self._on_key_press)
                print(f"   [Keyboard controls registered on viewer]", flush=True)
        
        print(f"\n🎈 Inflatable Soft Body Ready!", flush=True)
        print(f"   Radius: {radius}m", flush=True)
        print(f"   Max inflation: {max_pressure}x volume", flush=True)
        print(f"   Stiffness: μ={k_mu:.0e}, λ={k_lambda:.0e}", flush=True)
        print(f"\n   Keyboard Controls:", flush=True)
        print(f"   [I] or [=]     - Increase pressure (inflate)", flush=True)
        print(f"   [K] or [-]     - Decrease pressure (deflate)", flush=True)
        print(f"   [O]            - Reset to Original rest size", flush=True)
    
    def _on_key_press(self, symbol, modifiers):
        """Handle keyboard input for pressure control."""
        # Key codes
        KEY_I = 105      # Inflate
        KEY_K = 107      # decrease (below I)
        KEY_O = 111      # Original/reset
        KEY_EQUAL = 61   # + key (=/+)
        KEY_MINUS = 45   # - key
        
        if symbol in (KEY_I, KEY_EQUAL):
            self.current_pressure = min(self.max_pressure, self.current_pressure + self.pressure_step)
            print(f"   [Pressure: {self.current_pressure:.2f}x]", flush=True)
        elif symbol in (KEY_K, KEY_MINUS):
            self.current_pressure = max(0.5, self.current_pressure - self.pressure_step)
            print(f"   [Pressure: {self.current_pressure:.2f}x]", flush=True)
        elif symbol == KEY_O:
            self.current_pressure = 1.0
            print(f"   [Reset to rest size]", flush=True)
    
    def _check_keys(self):
        """Poll keyboard state for pressure control (backup if callbacks don't work)."""
        if not self.viewer or not hasattr(self.viewer, 'renderer'):
            return
        renderer = self.viewer.renderer
        if not hasattr(renderer, 'is_key_down'):
            return
        
        # Key codes
        KEY_I = 105
        KEY_K = 107
        KEY_O = 111
        KEY_EQUAL = 61
        KEY_MINUS = 45
        
        # Check keys (with rate limiting via frame count)
        if not hasattr(self, '_key_cooldown'):
            self._key_cooldown = 0
        
        if self._key_cooldown > 0:
            self._key_cooldown -= 1
            return
            
        if renderer.is_key_down(KEY_I) or renderer.is_key_down(KEY_EQUAL):
            self.current_pressure = min(self.max_pressure, self.current_pressure + self.pressure_step)
            print(f"   [Pressure: {self.current_pressure:.2f}x]", flush=True)
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_K) or renderer.is_key_down(KEY_MINUS):
            self.current_pressure = max(0.5, self.current_pressure - self.pressure_step)
            print(f"   [Pressure: {self.current_pressure:.2f}x]", flush=True)
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_O):
            self.current_pressure = 1.0
            print(f"   [Reset to rest size]", flush=True)
            self._key_cooldown = 10
    
    def step(self):
        """Run one frame of simulation with inflation."""
        # Check keyboard for pressure control
        self._check_keys()
        
        # Apply current pressure (controlled by keyboard)
        self.solver.set_pressure(self.current_pressure)
        
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            
            # Collision detection
            self.contacts = self.model.collide(state=self.state_0)
            
            # Physics step
            self.solver.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt
            )
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
    
    def render(self):
        """Render current frame."""
        if self.viewer is None:
            return
            
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
    
    def run(self, num_frames: int = 1800):
        """Run simulation loop."""
        print(f"\n🎈 Starting inflation demo...", flush=True)
        print(f"   Cycling pressure from 1.0x to {self.max_pressure:.1f}x", flush=True)
        
        for frame in range(num_frames):
            self.step()
            self.render()
            
            # Track volume ratio
            volume_ratio = self.solver.get_volume_ratio(self.state_0)
            
            # Print periodic updates
            if frame % 60 == 0:
                positions = self.state_0.particle_q.numpy()
                center = positions.mean(axis=0)
                height = center[2]
                
                print(f"   Frame {frame}: "
                      f"pressure={self.current_pressure:.2f}x, "
                      f"volume_ratio={volume_ratio:.2f}x, "
                      f"height={height:.2f}m", flush=True)
        
        # Final statistics
        info = self.solver.get_inflation_info(self.state_0)
        print(f"\n🎈 Simulation complete!", flush=True)
        print(f"   Initial volume: {info['initial_volume']:.4f}", flush=True)
        print(f"   Final volume: {info['current_volume']:.4f}", flush=True)
        print(f"   Final ratio: {info['current_ratio']:.2f}x", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Inflatable Soft Body Sphere Demo')
    
    # Mesh parameters
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Sphere radius (default: 0.3)')
    parser.add_argument('--subdivisions', type=int, default=2,
                        help='Icosphere subdivisions (default: 2)')
    parser.add_argument('--interior_layers', type=int, default=2,
                        help='Interior radial layers (default: 2)')
    
    # Physics parameters
    parser.add_argument('--initial_height', type=float, default=0.5,
                        help='Initial height (default: 0.5)')
    parser.add_argument('--mass', type=float, default=1.0,
                        help='Mass (default: 1.0)')
    parser.add_argument('--k_mu', type=float, default=1.0e5,
                        help='Shear modulus (default: 1.0e5)')
    parser.add_argument('--k_lambda', type=float, default=1.0e5,
                        help='Bulk modulus (default: 1.0e5)')
    parser.add_argument('--k_damp', type=float, default=1.0,
                        help='Damping (default: 1.0)')
    parser.add_argument('--spring_ke', type=float, default=5.0e4,
                        help='Spring stiffness (default: 5.0e4)')
    parser.add_argument('--spring_kd', type=float, default=1.0,
                        help='Spring damping (default: 1.0)')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='Gravity (default: 9.81)')
    
    # Inflation parameters
    parser.add_argument('--max_pressure', type=float, default=5.0,
                        help='Maximum inflation ratio (default: 2.0)')
    parser.add_argument('--cycle_speed', type=float, default=0.02,
                        help='Inflation cycle speed (default: 0.02)')
    
    # Simulation parameters
    parser.add_argument('--substeps', type=int, default=5,
                        help='Substeps per frame (default: 5)')
    parser.add_argument('--num_frames', type=int, default=1800,
                        help='Number of frames (default: 1800 = 30 seconds at 60fps)')
    parser.add_argument('--device', type=str, default=None,
                        help='Compute device')
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization')
    
    args = parser.parse_args()
    
    wp.init()
    
    with wp.ScopedDevice(args.device):
        # Create viewer
        if args.headless:
            viewer = None
        else:
            try:
                # Try OpenGL viewer first
                viewer = newton.viewer.ViewerGL(
                    width=1920,
                    height=1080,
                )
            except Exception as e:
                print(f"Could not create OpenGL viewer: {e}")
                try:
                    # Fall back to Rerun
                    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
                except Exception as e2:
                    print(f"Could not create Rerun viewer: {e2}")
                    print("Running headless...")
                    viewer = None
        
        example = Example(
            viewer=viewer,
            radius=args.radius,
            subdivisions=args.subdivisions,
            interior_layers=args.interior_layers,
            initial_height=args.initial_height,
            mass=args.mass,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            spring_ke=args.spring_ke,
            spring_kd=args.spring_kd,
            gravity=args.gravity,
            max_pressure=args.max_pressure,
            cycle_speed=args.cycle_speed,
            substeps=args.substeps,
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()
