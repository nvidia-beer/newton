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
Inflatable Soft Body Box with Rigid Body Interaction

Demonstrates an inflatable soft body box pushing a rigid box upward as it inflates,
and the box settling back down as the soft body deflates.

This combines:
- Inflation via FEM rest configuration scaling (SolverInflatable)
- Rigid body physics with XPBD or MuJoCo solver
- Soft-rigid contact interaction

The demo shows a soft box on the ground with a rigid box resting on top.
As the soft box inflates/deflates cyclically, the rigid box rises and falls.

Usage:
    python -m newton.examples.inflatable.example_inflatable_rigid [--solver xpbd|mujoco]
    python -m newton.examples.inflatable.example_inflatable_rigid --solver mujoco --max_pressure 2.5
    
Options:
    --solver xpbd      : Use XPBD solver (default, unified model)
    --solver mujoco    : Use MuJoCo solver (hybrid approach with unified model)
"""

import warp as wp
import numpy as np
import argparse
import math

import newton
from newton.solvers import SolverInflatable, SolverXPBD, TetraBox


# Colors
COLOR_GREEN = (0.2, 0.9, 0.3)    # Rigid box (XPBD)
COLOR_RED = (0.9, 0.2, 0.2)      # Rigid box (MuJoCo)
COLOR_BLUE = (0.3, 0.5, 1.0)     # Soft box (inflatable)


class Example:
    """
    Inflatable soft body box with rigid body interaction.
    
    A soft box inflates/deflates while a rigid box sits on top,
    demonstrating the force transfer from inflation to rigid body dynamics.
    Supports both XPBD and MuJoCo solvers for rigid body physics.
    """
    
    def __init__(
        self,
        viewer,
        solver_type: str = "xpbd",  # "xpbd" or "mujoco"
        size=(0.3, 0.3, 0.3),  # Box size (width, height, depth)
        subdivisions=(3, 3, 3),  # Subdivisions per axis
        soft_mass: float = 1.0,
        rigid_width: float = 3.0,     # Rigid box width/length (5x soft body diameter)
        rigid_mass: float = 0.01,     # Light plate (10 grams, was 1g - better for MuJoCo)
        particle_radius: float = 0.015, # Soft body particle collision radius (was 0.03, reduced for MuJoCo)
        k_mu: float = 1.0e5,          # Shear modulus (same as inflatable.py)
        k_lambda: float = 1.0e5,      # Bulk modulus (same as inflatable.py)
        k_damp: float = 5.0,          # Slightly higher damping for contact stability
        spring_ke: float = 5.0e4,     # Spring stiffness (same as inflatable.py)
        spring_kd: float = 5.0,       # Slightly higher spring damping for contact
        gravity: float = 9.81,
        max_pressure: float = 5.0,    # Max 5x volume
        cycle_speed: float = 0.01,    # Cycle speed
        substeps: int = 16,           # More substeps for contact stability
        use_mujoco_cpu: bool = False,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        if isinstance(size, (int, float)):
            self.size = (float(size), float(size), float(size))
        else:
            self.size = tuple(float(s) for s in size)
        self.soft_mass = soft_mass
        self.rigid_mass = rigid_mass
        self.particle_radius = particle_radius
        self.max_pressure = max_pressure
        self.cycle_speed = cycle_speed
        self.solver_type = solver_type.lower()
        
        if self.solver_type not in ["xpbd", "mujoco"]:
            raise ValueError(f"Invalid solver_type: {solver_type}. Choose 'xpbd' or 'mujoco'.")
        
        self.viewer = viewer
        
        # Generate FEM box mesh
        print(f"\n📦 Generating tetrahedral box mesh...", flush=True)
        box = TetraBox(
            size=self.size,
            subdivisions=subdivisions,
            verbose=True
        )
        mesh_data = box.get_mesh_data()
        
        vertices = mesh_data['vertices']
        indices = mesh_data['indices']
        tetrahedra = mesh_data['tetrahedra']
        
        print(f"   Mesh: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra", flush=True)
        
        # Build Newton model
        builder = newton.ModelBuilder()
        
        # Add bouncy ground plane at Z=0
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,   # High stiffness
                kd=1e3,   # Some damping
                kf=1e4,   # Friction stiffness
                mu=0.5    # Friction coefficient
            )
        )
        
        # ===== Compute soft body positioning from actual mesh geometry =====
        vertices_np = np.array(vertices)
        mesh_min_z = vertices_np[:, 2].min()  # Bottom of box
        mesh_max_z = vertices_np[:, 2].max()  # Top of box
        mesh_height = mesh_max_z - mesh_min_z
        
        # Position soft body so its bottom rests on ground (z=0)
        # soft_offset shifts mesh so mesh_min_z + soft_offset = 0
        soft_offset = -mesh_min_z + particle_radius  # Small gap for particle collision radius
        soft_box_top = mesh_max_z + soft_offset  # Top of soft body in world coords
        
        # Flat box dimensions: height = width / 50 (thin plate)
        rigid_height = rigid_width / 50.0  # Thin plate
        rigid_half_width = rigid_width / 2.0
        rigid_half_height = rigid_height / 2.0
        
        # Position rigid box above soft box
        # Box center is at soft_box_top + half_height + gap
        # Start with box just barely touching the soft body for better MuJoCo initialization
        gap = particle_radius * 0.5 if self.solver_type == "mujoco" else particle_radius
        rigid_z = soft_box_top + rigid_half_height + gap
        
        print(f"\n📐 Geometry-based positioning:", flush=True)
        print(f"   Mesh Z range: [{mesh_min_z:.3f}, {mesh_max_z:.3f}] (height={mesh_height:.3f}m)", flush=True)
        print(f"   Soft body offset: {soft_offset:.3f}m (bottom at z={particle_radius:.3f}m)", flush=True)
        print(f"   Soft box top at z={soft_box_top:.3f}m", flush=True)
        print(f"   Rigid box: {rigid_width}m x {rigid_width}m x {rigid_height:.4f}m (flat plate)", flush=True)
        print(f"   Rigid box center at z={rigid_z:.3f}m (gap={gap:.3f}m)", flush=True)
        
        # Create rigid body (method differs between XPBD and MuJoCo)
        color_name = "RED" if self.solver_type == "mujoco" else "GREEN"
        print(f"\n📦 Adding rigid box ({color_name})...", flush=True)
        
        if self.solver_type == "mujoco":
            # MuJoCo uses add_link
            self.rigid_body_id = builder.add_link(mass=rigid_mass)
            joint_id = builder.add_joint_free(
                child=self.rigid_body_id,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, rigid_z), wp.quat_identity())
            )
        else:
            # XPBD uses add_body
            self.rigid_body_id = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, rigid_z), wp.quat_identity())
            )
            joint_id = builder.add_joint_free(self.rigid_body_id)
        
        builder.add_articulation([joint_id], key="rigid_box")
        
        # Add flat box shape (width x width x height)
        box_volume = rigid_width * rigid_width * rigid_height
        self.rigid_shape_id = builder.add_shape_box(
            body=self.rigid_body_id,
            hx=rigid_half_width,
            hy=rigid_half_width,
            hz=rigid_half_height,
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,
                kd=100.0,
                kf=1e4,
                mu=0.5,
                density=rigid_mass / box_volume,
            )
        )
        print(f"   Rigid mass: {rigid_mass}kg, volume: {box_volume:.6f}m³", flush=True)
        
        # Store for tracking
        self.initial_rigid_z = rigid_z
        
        # ===== Add soft inflatable box =====
        print(f"\n📦 Adding inflatable soft box (BLUE)...", flush=True)
        
        # Offset vertices using computed soft_offset (places bottom at ground + particle_radius)
        positioned_vertices = [(v[0], v[1], v[2] + soft_offset) for v in vertices]
        
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=positioned_vertices,
            indices=indices,
            scale=1.0,
            density=soft_mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )
        
        # Finalize model
        self.model = builder.finalize()
        
        print(f"\nModel created:", flush=True)
        print(f"  Bodies: {self.model.body_count}", flush=True)
        print(f"  Particles: {self.model.particle_count}", flush=True)
        print(f"  Springs: {self.model.spring_count}", flush=True)
        print(f"  Tetrahedra: {self.model.tet_count}", flush=True)
        
        # Set gravity (Z is up, gravity pulls down)
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        
        # Soft contact parameters (controls force from soft body onto rigid body)
        # Higher damping is critical for MuJoCo rigid-soft interaction
        self.model.soft_contact_ke = 5e4      # Increased stiffness for better contact
        self.model.soft_contact_kd = 2000.0   # Much higher damping (was 500.0) - critical for MuJoCo
        self.model.soft_contact_kf = 1.0e4    # Friction stiffness
        self.model.soft_contact_mu = 0.8      # Higher friction for better interaction
        
        # Particle constraint parameters
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        
        # Particle collision and rendering radius
        # IMPORTANT: This radius is used for collision detection with rigid bodies!
        # Must be large enough to prevent interpenetration
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, self.particle_radius),
            dtype=wp.float32,
            device=self.model.device
        )
        print(f"   Particle collision radius: {self.particle_radius}m", flush=True)
        
        # ===== Create solvers =====
        print(f"\n--- Creating Solvers ---", flush=True)
        
        # Inflatable solver for the soft body
        print(f"Creating SolverInflatable...", flush=True)
        self.soft_solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            mass=soft_mass,
            max_volume_ratio=max_pressure,
            solver_type="bicgstab",
            linear_solver_maxiter=150,  # larger mesh needs more iterations for convergence
            use_constraint_contacts=True,  # XPBD-style contact correction for stable rigid-soft and ground contact
        )
        
        # Rigid body solver
        if self.solver_type == "mujoco":
            # Import MuJoCo solver only when needed
            try:
                from newton.solvers import SolverMuJoCo
                print(f"Creating SolverMuJoCo...", flush=True)
                self.rigid_solver = SolverMuJoCo(
                    self.model,
                    use_mujoco_cpu=use_mujoco_cpu,
                    use_mujoco_contacts=False,  # Use Newton's contact system for rigid-soft interaction
                )
            except ImportError as e:
                print("\n" + "=" * 70)
                print("ERROR: MuJoCo dependencies not installed")
                print("=" * 70)
                print(f"\n{e}")
                print("\nTo use MuJoCo solver, install:")
                print("  pip install mujoco mujoco_warp")
                raise
        else:
            print(f"Creating SolverXPBD...", flush=True)
            self.rigid_solver = SolverXPBD(self.model)
        
        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_soft = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        
        # Initialize FK
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0
        )
        
        # Debug: verify rigid body has proper mass and initial position
        if self.model.body_count > 0:
            body_mass = self.model.body_mass.numpy()
            body_inv_mass = self.model.body_inv_mass.numpy()
            body_q = self.state_0.body_q.numpy()
            print(f"\n[DEBUG] Rigid body mass: {body_mass[0]:.4f}kg, inv_mass: {body_inv_mass[0]:.4f}", flush=True)
            print(f"[DEBUG] Gravity: {self.model.gravity.numpy()[0]}", flush=True)
            print(f"[DEBUG] Initial body_q: {body_q[0]}", flush=True)
        
        # Set up viewer
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True
            # Color the rigid box based on solver
            rigid_color = COLOR_RED if self.solver_type == "mujoco" else COLOR_GREEN
            self.viewer.update_shape_colors({
                self.rigid_shape_id: rigid_color,
            })
        
        # Inflation state - manual control only
        self.current_pressure = 1.0
        self.pressure_step = 0.1  # Pressure adjustment per key press
        
        # Tracking
        self.initial_rigid_height = rigid_z
        self.max_rigid_height = rigid_z
        
        # Register keyboard controls if viewer supports it
        if self.viewer:
            # Try renderer first (ViewerGL), then viewer directly
            if hasattr(self.viewer, 'renderer') and hasattr(self.viewer.renderer, 'register_key_press'):
                self.viewer.renderer.register_key_press(self._on_key_press)
                print(f"   [Keyboard controls registered on renderer]", flush=True)
            elif hasattr(self.viewer, 'register_key_press'):
                self.viewer.register_key_press(self._on_key_press)
                print(f"   [Keyboard controls registered on viewer]", flush=True)
        
        # Calculate expected expansion
        linear_scale_at_max = np.cbrt(max_pressure)
        
        print(f"\n📦 Inflatable + Rigid Interaction Ready!", flush=True)
        if self.solver_type == "mujoco":
            print(f"   Rigid Solver: MuJoCo {'(CPU)' if use_mujoco_cpu else '(GPU)'}", flush=True)
            print(f"   Soft Solver: Newton SolverInflatable (Implicit FEM)", flush=True)
            print(f"   RED box will rise/fall as BLUE box inflates/deflates", flush=True)
        else:
            print(f"   Solver: XPBD for rigid, SolverInflatable for soft", flush=True)
            print(f"   GREEN box will rise/fall as BLUE box inflates/deflates", flush=True)
        print(f"   Max inflation: {max_pressure}x volume", flush=True)
        print(f"   Linear scale at max: {linear_scale_at_max:.2f}x", flush=True)
        print(f"   Box size: {self.size[0]:.2f}×{self.size[1]:.2f}×{self.size[2]:.2f}m", flush=True)
        print(f"   Stiffness: k_mu={k_mu:.0e}, k_lambda={k_lambda:.0e}", flush=True)
        print(f"\n   Keyboard Controls:", flush=True)
        print(f"   [I] or [=]     - Increase pressure (inflate)", flush=True)
        print(f"   [K] or [-]     - Decrease pressure (deflate)", flush=True)
        print(f"   [O]            - Reset to Original rest size", flush=True)
        print(f"   [F]            - Toggle wireframe mode", flush=True)
    
    def _on_key_press(self, symbol, modifiers):
        """Handle keyboard input for pressure control."""
        # Key codes
        KEY_I = 105      # Inflate
        KEY_K = 107      # decrease (below I)
        KEY_O = 111      # Original/reset
        KEY_F = 102      # wireFrame toggle
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
        elif symbol == KEY_F:
            # Toggle wireframe mode
            if self.viewer and hasattr(self.viewer, 'renderer'):
                self.viewer.renderer.draw_wireframe = not self.viewer.renderer.draw_wireframe
                mode = "ON" if self.viewer.renderer.draw_wireframe else "OFF"
                print(f"   [Wireframe: {mode}]", flush=True)
    
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
        KEY_F = 102
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
            self._key_cooldown = 10  # Prevent too rapid changes
        elif renderer.is_key_down(KEY_K) or renderer.is_key_down(KEY_MINUS):
            self.current_pressure = max(0.5, self.current_pressure - self.pressure_step)
            print(f"   [Pressure: {self.current_pressure:.2f}x]", flush=True)
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_O):
            self.current_pressure = 1.0
            print(f"   [Reset to rest size]", flush=True)
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_F):
            renderer.draw_wireframe = not renderer.draw_wireframe
            mode = "ON" if renderer.draw_wireframe else "OFF"
            print(f"   [Wireframe: {mode}]", flush=True)
            self._key_cooldown = 10
    
    def _apply_soft_contact_forces_to_bodies(self, state, contacts):
        """
        Apply soft contact reaction forces to rigid bodies.
        
        Critical for MuJoCo: Soft contacts apply forces to particles, but MuJoCo
        rigid bodies need the equal/opposite reaction forces explicitly applied.
        """
        import warp as wp
        import numpy as np
        
        contact_count = int(contacts.soft_contact_count.numpy()[0])
        if contact_count == 0:
            return
        
        # Get contact data
        particle_idx = contacts.soft_contact_particle.numpy()[:contact_count]
        shape_idx = contacts.soft_contact_shape.numpy()[:contact_count]
        body_pos = contacts.soft_contact_body_pos.numpy()[:contact_count]
        body_vel = contacts.soft_contact_body_vel.numpy()[:contact_count]
        normals = contacts.soft_contact_normal.numpy()[:contact_count]
        
        # Get particle states
        particle_q = state.particle_q.numpy()
        particle_qd = state.particle_qd.numpy()
        particle_radius = self.model.particle_radius.numpy()
        
        # Contact parameters - INCREASED for MuJoCo
        ke = self.model.soft_contact_ke * 5.0  # 5x multiplier for MuJoCo reaction forces
        kd = self.model.soft_contact_kd
        kf = self.model.soft_contact_kf
        mu = self.model.soft_contact_mu
        
        # Map shapes to bodies
        shape_body = self.model.shape_body.numpy()
        
        # Accumulate forces per body
        body_forces = {}
        body_torques = {}
        
        for i in range(contact_count):
            pid = particle_idx[i]
            sid = shape_idx[i]
            body_id = shape_body[sid]
            
            if body_id < 0:
                continue  # Ground or invalid
            
            # Get particle state
            x = particle_q[pid]
            v = particle_qd[pid]
            radius = particle_radius[pid]
            n = normals[i]
            b_pos = body_pos[i]
            b_vel = body_vel[i]
            
            # Compute penetration
            d = np.dot(x - b_pos, n)
            penetration = radius - d
            
            if penetration > 0.0:
                # Relative velocity
                rel_v = v - b_vel
                vn = np.dot(rel_v, n)
                vt = rel_v - vn * n
                vt_norm = np.linalg.norm(vt)
                
                # Normal force (spring + damping)
                fn = ke * penetration - kd * vn
                
                # Friction force
                if vt_norm > 1e-6:
                    ft_dir = vt / vt_norm
                    ft_mag = min(mu * abs(fn), kf * vt_norm)
                    ft = -ft_mag * ft_dir
                else:
                    ft = np.zeros(3)
                
                # Total force on particle (from body)
                f_particle = fn * n + ft
                
                # Reaction force on body (Newton's 3rd law)
                f_body = -f_particle
                
                # Torque on body (r × F)
                r = x - b_pos  # Contact point relative to body
                torque = np.cross(r, f_body)
                
                # Accumulate
                if body_id not in body_forces:
                    body_forces[body_id] = np.zeros(3)
                    body_torques[body_id] = np.zeros(3)
                
                body_forces[body_id] += f_body
                body_torques[body_id] += torque
        
        # Apply accumulated forces to body_f
        if body_forces:
            body_f = state.body_f.numpy()
            for body_id, force in body_forces.items():
                body_f[body_id, 0:3] += force  # Linear force
                body_f[body_id, 3:6] += body_torques[body_id]  # Angular torque
            
            # Copy back to device
            state.body_f = wp.array(body_f, dtype=wp.spatial_vector, device=self.model.device)
            
            # Debug output
            if hasattr(self, '_debug_step_count') and self._debug_step_count <= 30:
                total_force = np.linalg.norm(list(body_forces.values())[0])
                print(f"   [DEBUG MuJoCo] Applied reaction force to body: {total_force:.2f}N (contacts={contact_count})", flush=True)
    
    def step(self):
        """Run one frame of simulation with inflation and rigid-soft interaction."""
        # Check keyboard for pressure control
        self._check_keys()
        
        # Apply current pressure (controlled by keyboard)
        self.soft_solver.set_pressure(self.current_pressure)
        
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            
            # Unified collision detection (detects rigid-rigid, rigid-soft, soft-ground, etc.)
            self.contacts = self.model.collide(state=self.state_0)
            
            # Debug: Check soft contact count and rigid body position (only for MuJoCo, first few substeps)
            if self.solver_type == "mujoco" and hasattr(self, '_debug_step_count'):
                self._debug_step_count += 1
                if self._debug_step_count <= 50:  # More debug output
                    if self.contacts and hasattr(self.contacts, 'soft_contact_count'):
                        count = int(self.contacts.soft_contact_count.numpy()[0])
                        # Get rigid body position BEFORE stepping
                        body_q = self.state_0.body_q.numpy()
                        if len(body_q) > 0:
                            rigid_pos = body_q[self.rigid_body_id]
                            rigid_z = rigid_pos[2]
                            body_vel = self.state_0.body_qd.numpy()[self.rigid_body_id]
                            rigid_vz = body_vel[2]
                            if self._debug_step_count % 10 == 0:  # Every 10 substeps
                                print(f"   [DEBUG MuJoCo substep {self._debug_step_count}] z={rigid_z:.4f}m, vz={rigid_vz:.4f}m/s, contacts={count}", flush=True)
            elif not hasattr(self, '_debug_step_count'):
                self._debug_step_count = 0
            
            # Soft solver updates particles (SolverInflatable handles inflation + FEM)
            self.soft_solver.step(
                self.state_0, self.state_soft, self.control, self.contacts, self.sim_dt
            )
            
            # Rigid solver updates bodies
            self.rigid_solver.step(
                self.state_0, self.state_rigid, self.control, self.contacts, self.sim_dt
            )
            
            # Combine results: particles from soft, bodies from rigid
            wp.copy(self.state_1.particle_q, self.state_soft.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_soft.particle_qd)
            wp.copy(self.state_1.body_q, self.state_rigid.body_q)
            wp.copy(self.state_1.body_qd, self.state_rigid.body_qd)
            wp.copy(self.state_1.joint_q, self.state_rigid.joint_q)
            wp.copy(self.state_1.joint_qd, self.state_rigid.joint_qd)
            
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
    
    def run(self, num_frames: int = 600):
        """Run simulation loop."""
        print(f"\n📦 Starting inflation + rigid interaction demo...", flush=True)
        color_name = "RED" if self.solver_type == "mujoco" else "GREEN"
        print(f"   Watch the {color_name} box rise and fall!", flush=True)
        
        for frame in range(num_frames):
            self.step()
            self.render()
            
            # Track box height
            body_q = self.state_0.body_q.numpy()
            if len(body_q) > 0:
                rigid_pos = body_q[self.rigid_body_id]
                rigid_height = rigid_pos[2]  # Z position (translation part of transform)
                self.max_rigid_height = max(self.max_rigid_height, rigid_height)
            
            # Track volume ratio
            volume_ratio = self.soft_solver.get_volume_ratio(self.state_0)
            
            # Print periodic updates (every 30 frames = 0.5 seconds)
            if frame % 30 == 0:
                # Debug: Check if soft contacts are being generated
                if self.contacts and hasattr(self.contacts, 'soft_contact_count'):
                    contact_count = int(self.contacts.soft_contact_count.numpy()[0])
                    if frame < 90:  # Only print first few times
                        print(f"   [DEBUG] Soft contacts: {contact_count}", flush=True)
                particle_positions = self.state_0.particle_q.numpy()
                sphere_center = particle_positions.mean(axis=0)
                sphere_height = sphere_center[2]
                
                linear_scale = np.cbrt(self.current_pressure)
                
                print(f"   Frame {frame}: "
                      f"pressure={self.current_pressure:.2f}x (scale={linear_scale:.2f}), "
                      f"volume={volume_ratio:.2f}x, "
                      f"rigid_z={rigid_height:.3f}m", flush=True)
        
        # Final statistics
        info = self.soft_solver.get_inflation_info(self.state_0)
        print(f"\n📦 Simulation complete!", flush=True)
        print(f"   Initial rigid height: {self.initial_rigid_height:.3f}m", flush=True)
        print(f"   Max rigid height: {self.max_rigid_height:.3f}m", flush=True)
        print(f"   Height gain: {self.max_rigid_height - self.initial_rigid_height:.3f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Inflatable + Rigid Interaction Demo',
        epilog='Choose between XPBD (default) or MuJoCo solver for rigid body dynamics'
    )
    
    # Solver selection
    parser.add_argument('--solver', type=str, default='xpbd', choices=['xpbd', 'mujoco'],
                        help='Rigid body solver: "xpbd" (default) or "mujoco"')
    parser.add_argument('--use-mujoco-cpu', action='store_true',
                        help='Use MuJoCo CPU backend (MuJoCo only)')
    
    # Mesh parameters
    parser.add_argument('--size', type=float, nargs=3, default=[0.3, 0.3, 0.3],
                        help='Box size (width, height, depth) (default: 0.3 0.3 0.3)')
    parser.add_argument('--subdivisions', type=int, nargs=3, default=[3, 3, 3],
                        help='Subdivisions per axis (default: 3 3 3)')
    
    # Physics parameters
    parser.add_argument('--soft_mass', type=float, default=1.0,
                        help='Soft body mass (default: 1.0)')
    parser.add_argument('--rigid_width', type=float, default=3.0,
                        help='Rigid box width/length (default: 3.0 = 5x soft diameter, height=width/5)')
    parser.add_argument('--rigid_mass', type=float, default=0.5,
                        help='Rigid box mass (default: 0.5)')
    parser.add_argument('--particle_radius', type=float, default=0.03,
                        help='Soft body particle collision radius (default: 0.03)')
    parser.add_argument('--k_mu', type=float, default=5.0e4,
                        help='Shear modulus (default: 5.0e4)')
    parser.add_argument('--k_lambda', type=float, default=5.0e4,
                        help='Bulk modulus (default: 5.0e4)')
    parser.add_argument('--k_damp', type=float, default=50.0,
                        help='Damping (default: 50.0)')
    parser.add_argument('--spring_ke', type=float, default=2.0e4,
                        help='Spring stiffness (default: 2.0e4)')
    parser.add_argument('--spring_kd', type=float, default=20.0,
                        help='Spring damping (default: 20.0)')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='Gravity (default: 9.81)')
    
    # Inflation parameters
    parser.add_argument('--max_pressure', type=float, default=5.0,
                        help='Maximum inflation ratio (default: 3.0)')
    parser.add_argument('--cycle_speed', type=float, default=0.005,
                        help='Inflation cycle speed (default: 0.005)')
    
    # Simulation parameters
    parser.add_argument('--substeps', type=int, default=32,
                        help='Substeps per frame (default: 32)')
    parser.add_argument('--num_frames', type=int, default=600,
                        help='Number of frames (default: 600)')
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
                viewer = newton.viewer.ViewerGL(
                    width=1920,
                    height=1080,
                )
            except Exception as e:
                print(f"Could not create OpenGL viewer: {e}")
                try:
                    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
                except Exception as e2:
                    print(f"Could not create Rerun viewer: {e2}")
                    print("Running headless...")
                    viewer = None
        
        example = Example(
            viewer=viewer,
            solver_type=args.solver,
            size=args.size,
            subdivisions=args.subdivisions,
            soft_mass=args.soft_mass,
            rigid_width=args.rigid_width,
            rigid_mass=args.rigid_mass,
            particle_radius=args.particle_radius,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            spring_ke=args.spring_ke,
            spring_kd=args.spring_kd,
            gravity=args.gravity,
            max_pressure=args.max_pressure,
            cycle_speed=args.cycle_speed,
            substeps=args.substeps,
            use_mujoco_cpu=args.use_mujoco_cpu,
        )
        example.run(num_frames=args.num_frames)
        
        if viewer:
            viewer.close()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        import sys
        if "mujoco" in str(e).lower():
            print("\n" + "=" * 70)
            print("ERROR: MuJoCo dependencies not installed")
            print("=" * 70)
            print(f"\n{e}")
            print("\nTo use MuJoCo solver, install:")
            print("  pip install mujoco mujoco_warp")
            sys.exit(1)
        else:
            raise
