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
Inflatable Table - Four Independently Inflatable Soft Bodies

Demonstrates four inflatable soft bodies at each corner of a rigid plate,
each with INDEPENDENT pressure control. Press 1-4 to select a ball,
then I/K to inflate/deflate that specific ball.

This combines:
- Per-ball inflation via FEM rest configuration scaling
- Rigid body physics with XPBD solver
- Multiple soft-rigid contact interactions

Usage:
    python -m newton.examples.inflatable.example_inflatable_table
    python -m newton.examples.inflatable.example_inflatable_table --max_pressure 3.0
"""

import warp as wp
import numpy as np
import argparse

import newton
from newton.solvers import SolverInflatable, SolverXPBD, TetraSphere


# Warp kernels for per-range scaling
@wp.kernel
def scale_spring_range_kernel(
    original_rest_lengths: wp.array(dtype=wp.float32),
    scale: wp.float32,
    start_idx: wp.int32,
    count: wp.int32,
    scaled_rest_lengths: wp.array(dtype=wp.float32),
):
    """Scale spring rest lengths for a specific range."""
    tid = wp.tid()
    if tid < count:
        idx = start_idx + tid
        scaled_rest_lengths[idx] = original_rest_lengths[idx] * scale


@wp.kernel
def scale_tet_range_kernel(
    original_poses: wp.array(dtype=wp.mat33),
    scale: wp.float32,
    start_idx: wp.int32,
    count: wp.int32,
    scaled_poses: wp.array(dtype=wp.mat33),
):
    """Scale tetrahedra rest poses for a specific range."""
    tid = wp.tid()
    if tid < count:
        idx = start_idx + tid
        inv_scale = 1.0 / scale
        orig = original_poses[idx]
        scaled_poses[idx] = wp.mat33(
            orig[0, 0] * inv_scale, orig[0, 1] * inv_scale, orig[0, 2] * inv_scale,
            orig[1, 0] * inv_scale, orig[1, 1] * inv_scale, orig[1, 2] * inv_scale,
            orig[2, 0] * inv_scale, orig[2, 1] * inv_scale, orig[2, 2] * inv_scale
        )


# Colors (RGB tuples)
COLOR_GREY_TRANSPARENT = (0.6, 0.6, 0.6)  # Rigid plate wireframe (grey)
COLOR_PARTICLE_SELECTED = (0.3, 0.6, 1.0) # Light blue particles for selected ball


class Example:
    """
    Inflatable table with four INDEPENDENTLY inflatable soft bodies at corners.
    
    Each ball can be inflated/deflated separately using keyboard controls.
    """
    
    def __init__(
        self,
        viewer,
        radius: float = 0.25,
        subdivisions: int = 2,
        interior_layers: int = 2,
        soft_mass: float = 1.0,
        rigid_width: float = 3.0,
        rigid_mass: float = 0.004,
        particle_radius: float = 0.03,
        k_mu: float = 1.0e5,
        k_lambda: float = 1.0e5,
        k_damp: float = 5.0,
        spring_ke: float = 5.0e4,
        spring_kd: float = 5.0,
        gravity: float = 9.81,
        max_pressure: float = 5.0,
        substeps: int = 16,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.radius = radius
        self.max_pressure = max_pressure
        self.particle_radius = particle_radius
        
        self.viewer = viewer
        
        # Per-ball state - initialize BEFORE any method calls
        self.selected_ball = 0  # Currently selected ball (0-3)
        self.ball_pressures = [1.0, 1.0, 1.0, 1.0]  # Per-ball pressure
        self.pressure_step = 0.1
        
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
        
        # Add ground plane
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5, kd=1e3, kf=1e4, mu=0.5
            )
        )
        
        # Compute positioning
        vertices_np = np.array(vertices)
        mesh_min_z = vertices_np[:, 2].min()
        mesh_max_z = vertices_np[:, 2].max()
        
        soft_offset = -mesh_min_z + particle_radius
        soft_sphere_top = mesh_max_z + soft_offset
        
        rigid_height = rigid_width / 50.0
        rigid_half_width = rigid_width / 2.0
        rigid_half_height = rigid_height / 2.0
        rigid_z = soft_sphere_top + rigid_half_height + particle_radius
        
        # Store for wireframe rendering
        self.rigid_half_width = rigid_half_width
        self.rigid_half_height = rigid_half_height
        
        # Corner positions
        corner_offset = rigid_half_width - radius * 2.0
        corner_positions = [
            ( corner_offset,  corner_offset),  # Ball 1
            ( corner_offset, -corner_offset),  # Ball 2
            (-corner_offset,  corner_offset),  # Ball 3
            (-corner_offset, -corner_offset),  # Ball 4
        ]
        
        print(f"\n📐 Positioning:", flush=True)
        print(f"   Plate: {rigid_width}m x {rigid_width}m at z={rigid_z:.3f}m", flush=True)
        
        # Add rigid plate
        self.rigid_body_id = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, rigid_z), wp.quat_identity())
        )
        joint_id = builder.add_joint_free(self.rigid_body_id)
        builder.add_articulation([joint_id], key="rigid_plate")
        
        box_volume = rigid_width * rigid_width * rigid_height
        self.rigid_shape_id = builder.add_shape_box(
            body=self.rigid_body_id,
            hx=rigid_half_width,
            hy=rigid_half_width,
            hz=rigid_half_height,
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5, kd=100.0, kf=1e4, mu=0.5,
                density=rigid_mass / box_volume,
            )
        )
        
        self.initial_rigid_z = rigid_z
        
        # Add four soft spheres - track ranges for each
        print(f"\n🎈 Adding 4 inflatable soft spheres...", flush=True)
        
        self.soft_start_particles = []
        self.soft_particle_counts = []
        self.soft_start_tets = []
        self.soft_tet_counts = []
        self.soft_start_springs = []
        self.soft_spring_counts = []
        
        for i, (cx, cy) in enumerate(corner_positions):
            color_name = ["Red", "Green", "Blue", "Yellow"][i]
            print(f"   Ball {i+1} ({color_name}): ({cx:.2f}, {cy:.2f})", flush=True)
            
            positioned_vertices = [(v[0] + cx, v[1] + cy, v[2] + soft_offset) for v in vertices]
            
            start_particle = builder.particle_count
            start_tet = builder.tet_count
            start_spring = builder.spring_count
            
            self.soft_start_particles.append(start_particle)
            self.soft_start_tets.append(start_tet)
            self.soft_start_springs.append(start_spring)
            
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
            
            self.soft_particle_counts.append(builder.particle_count - start_particle)
            self.soft_tet_counts.append(builder.tet_count - start_tet)
            
            # Add springs
            num_tets = len(tetrahedra)
            added_springs = set()
            vertex_positions = np.array(positioned_vertices)
            
            for t in range(num_tets):
                tet_indices = [indices[t * 4 + k] for k in range(4)]
                edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                for ei, ej in edges:
                    i_local, j_local = tet_indices[ei], tet_indices[ej]
                    if i_local > j_local:
                        i_local, j_local = j_local, i_local
                    
                    spring_key = (i_local, j_local)
                    if spring_key not in added_springs:
                        added_springs.add(spring_key)
                        
                        p0 = vertex_positions[i_local]
                        p1 = vertex_positions[j_local]
                        rest_length = float(np.linalg.norm(p1 - p0))
                        
                        builder.add_spring(
                            start_particle + i_local,
                            start_particle + j_local,
                            spring_ke,
                            spring_kd,
                            rest_length
                        )
            
            self.soft_spring_counts.append(builder.spring_count - start_spring)
        
        # Finalize model
        self.model = builder.finalize()
        
        print(f"\nModel: {self.model.particle_count} particles, {self.model.tet_count} tets, {self.model.spring_count} springs", flush=True)
        
        # Store ORIGINAL rest configurations for per-ball scaling
        self.original_tet_poses = wp.clone(self.model.tet_poses)
        self.original_spring_rest_length = wp.clone(self.model.spring_rest_length)
        
        # Set physics parameters
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 500.0
        self.model.soft_contact_kf = 1.0e4
        self.model.soft_contact_mu = 0.5
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, particle_radius),
            dtype=wp.float32,
            device=self.model.device
        )
        
        # Create particle colors
        self._init_particle_colors()
        
        # Create solvers
        print(f"\n--- Creating Solvers ---", flush=True)
        self.soft_solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            mass=soft_mass,
            max_volume_ratio=max_pressure,
            solver_type="bicgstab"
        )
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
        
        # Set up viewer
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True
            self.viewer.show_triangles = False  # Hide default triangles, we render custom
            # Set plate to grey with transparency (alpha in material)
            self.viewer.update_shape_colors({
                self.rigid_shape_id: COLOR_GREY_TRANSPARENT,
            })
        
        # Tracking
        self.initial_rigid_height = rigid_z
        self.max_rigid_height = rigid_z
        
        # Register keyboard controls
        if self.viewer:
            if hasattr(self.viewer, 'renderer') and hasattr(self.viewer.renderer, 'register_key_press'):
                self.viewer.renderer.register_key_press(self._on_key_press)
        
        print(f"\n🎈 Inflatable Table Ready!", flush=True)
        print(f"   Solid mesh for soft bodies, particles for selected ball", flush=True)
        print(f"   Grey wireframe plate (transparent effect)", flush=True)
        print(f"\n   Keyboard Controls:", flush=True)
        print(f"   [1-4]          - Select ball (shows its particles)", flush=True)
        print(f"   [I] or [=]     - Inflate SELECTED ball", flush=True)
        print(f"   [K] or [-]     - Deflate SELECTED ball", flush=True)
        print(f"   [A]            - Inflate ALL balls", flush=True)
        print(f"   [Z]            - Deflate ALL balls", flush=True)
        print(f"   [O]            - Reset ALL to original size", flush=True)
        print(f"   [F]            - Toggle wireframe mode", flush=True)
    
    def _init_particle_colors(self):
        """Initialize particle colors - only selected ball shows particles."""
        self._update_particle_colors()
    
    def _update_particle_colors(self):
        """Update particle colors - only show particles for selected ball."""
        colors_np = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        
        # Only color the selected ball's particles
        start = self.soft_start_particles[self.selected_ball]
        count = self.soft_particle_counts[self.selected_ball]
        colors_np[start:start+count] = COLOR_PARTICLE_SELECTED
        
        self.particle_colors = wp.array(colors_np, dtype=wp.vec3, device=self.model.device)
        
        # Store which particles to show (only selected ball)
        self.particle_radii = np.zeros(self.model.particle_count, dtype=np.float32)
        self.particle_radii[start:start+count] = self.particle_radius
        self.particle_radii_array = wp.array(self.particle_radii, dtype=wp.float32, device=self.model.device)
    
    def _apply_ball_pressure(self, ball_idx: int, pressure: float):
        """Apply pressure to a specific ball by scaling its rest configuration."""
        linear_scale = float(np.cbrt(pressure))
        
        # Scale tetrahedra for this ball
        start_tet = self.soft_start_tets[ball_idx]
        tet_count = self.soft_tet_counts[ball_idx]
        
        if tet_count > 0:
            wp.launch(
                kernel=scale_tet_range_kernel,
                dim=tet_count,
                inputs=[
                    self.original_tet_poses,
                    linear_scale,
                    start_tet,
                    tet_count,
                ],
                outputs=[self.model.tet_poses],
                device=self.model.device,
            )
        
        # Scale springs for this ball
        start_spring = self.soft_start_springs[ball_idx]
        spring_count = self.soft_spring_counts[ball_idx]
        
        if spring_count > 0:
            wp.launch(
                kernel=scale_spring_range_kernel,
                dim=spring_count,
                inputs=[
                    self.original_spring_rest_length,
                    linear_scale,
                    start_spring,
                    spring_count,
                ],
                outputs=[self.model.spring_rest_length],
                device=self.model.device,
            )
    
    def _apply_all_pressures(self):
        """Apply current pressure settings to all balls."""
        for ball_idx in range(4):
            self._apply_ball_pressure(ball_idx, self.ball_pressures[ball_idx])
    
    def _on_key_press(self, symbol, modifiers):
        """Handle keyboard input for per-ball pressure control."""
        KEY_1, KEY_2, KEY_3, KEY_4 = 49, 50, 51, 52
        KEY_A, KEY_Z = 97, 122
        KEY_I, KEY_K, KEY_O, KEY_F = 105, 107, 111, 102
        KEY_EQUAL, KEY_MINUS = 61, 45
        
        # Ball selection
        if symbol == KEY_1:
            self.selected_ball = 0
            self._update_particle_colors()
            print(f"   [Selected: Ball 1 (Red) - pressure: {self.ball_pressures[0]:.2f}x]", flush=True)
        elif symbol == KEY_2:
            self.selected_ball = 1
            self._update_particle_colors()
            print(f"   [Selected: Ball 2 (Green) - pressure: {self.ball_pressures[1]:.2f}x]", flush=True)
        elif symbol == KEY_3:
            self.selected_ball = 2
            self._update_particle_colors()
            print(f"   [Selected: Ball 3 (Blue) - pressure: {self.ball_pressures[2]:.2f}x]", flush=True)
        elif symbol == KEY_4:
            self.selected_ball = 3
            self._update_particle_colors()
            print(f"   [Selected: Ball 4 (Yellow) - pressure: {self.ball_pressures[3]:.2f}x]", flush=True)
        
        # Inflate/deflate SELECTED ball
        elif symbol in (KEY_I, KEY_EQUAL):
            ball = self.selected_ball
            self.ball_pressures[ball] = min(self.max_pressure, self.ball_pressures[ball] + self.pressure_step)
            self._apply_ball_pressure(ball, self.ball_pressures[ball])
            name = ["Red", "Green", "Blue", "Yellow"][ball]
            print(f"   [Ball {ball+1} ({name}): {self.ball_pressures[ball]:.2f}x]", flush=True)
        elif symbol in (KEY_K, KEY_MINUS):
            ball = self.selected_ball
            self.ball_pressures[ball] = max(0.5, self.ball_pressures[ball] - self.pressure_step)
            self._apply_ball_pressure(ball, self.ball_pressures[ball])
            name = ["Red", "Green", "Blue", "Yellow"][ball]
            print(f"   [Ball {ball+1} ({name}): {self.ball_pressures[ball]:.2f}x]", flush=True)
        
        # Inflate/deflate ALL balls
        elif symbol == KEY_A:
            for i in range(4):
                self.ball_pressures[i] = min(self.max_pressure, self.ball_pressures[i] + self.pressure_step)
            self._apply_all_pressures()
            print(f"   [ALL balls: {self.ball_pressures}]", flush=True)
        elif symbol == KEY_Z:
            for i in range(4):
                self.ball_pressures[i] = max(0.5, self.ball_pressures[i] - self.pressure_step)
            self._apply_all_pressures()
            print(f"   [ALL balls: {self.ball_pressures}]", flush=True)
        
        # Reset ALL
        elif symbol == KEY_O:
            self.ball_pressures = [1.0, 1.0, 1.0, 1.0]
            self._apply_all_pressures()
            print(f"   [Reset ALL to 1.0x]", flush=True)
        
        # Wireframe toggle
        elif symbol == KEY_F:
            if self.viewer and hasattr(self.viewer, 'renderer'):
                self.viewer.renderer.draw_wireframe = not self.viewer.renderer.draw_wireframe
                mode = "ON" if self.viewer.renderer.draw_wireframe else "OFF"
                print(f"   [Wireframe: {mode}]", flush=True)
    
    def _check_keys(self):
        """Poll keyboard state (backup if callbacks don't work)."""
        if not self.viewer or not hasattr(self.viewer, 'renderer'):
            return
        renderer = self.viewer.renderer
        if not hasattr(renderer, 'is_key_down'):
            return
        
        if not hasattr(self, '_key_cooldown'):
            self._key_cooldown = 0
        
        if self._key_cooldown > 0:
            self._key_cooldown -= 1
            return
        
        KEY_I, KEY_K = 105, 107
        KEY_EQUAL, KEY_MINUS = 61, 45
        
        if renderer.is_key_down(KEY_I) or renderer.is_key_down(KEY_EQUAL):
            ball = self.selected_ball
            self.ball_pressures[ball] = min(self.max_pressure, self.ball_pressures[ball] + self.pressure_step)
            self._apply_ball_pressure(ball, self.ball_pressures[ball])
            print(f"   [Ball {ball+1}: {self.ball_pressures[ball]:.2f}x]", flush=True)
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_K) or renderer.is_key_down(KEY_MINUS):
            ball = self.selected_ball
            self.ball_pressures[ball] = max(0.5, self.ball_pressures[ball] - self.pressure_step)
            self._apply_ball_pressure(ball, self.ball_pressures[ball])
            print(f"   [Ball {ball+1}: {self.ball_pressures[ball]:.2f}x]", flush=True)
            self._key_cooldown = 10
    
    def step(self):
        """Run one frame of simulation."""
        self._check_keys()
        
        # Note: We apply per-ball pressures directly to model arrays,
        # so we don't use soft_solver.set_pressure() anymore
        
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            
            self.contacts = self.model.collide(
                state=self.state_0,
                soft_contact_margin=0.1
            )
            
            self.soft_solver.step(
                state_in=self.state_0,
                state_out=self.state_soft,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt
            )
            
            self.rigid_solver.step(
                state_in=self.state_0,
                state_out=self.state_rigid,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt
            )
            
            wp.copy(self.state_1.particle_q, self.state_soft.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_soft.particle_qd)
            wp.copy(self.state_1.body_q, self.state_rigid.body_q)
            wp.copy(self.state_1.body_qd, self.state_rigid.body_qd)
            wp.copy(self.state_1.joint_q, self.state_rigid.joint_q)
            wp.copy(self.state_1.joint_qd, self.state_rigid.joint_qd)
            
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
    
    def render(self):
        """Render current frame with solid soft body mesh and transparent plate wireframe."""
        if self.viewer is None:
            return
            
        self.viewer.begin_frame(self.sim_time)
        
        # Render plate as grey wireframe (transparent effect)
        self._render_plate_wireframe()
        
        # Render SOLID soft body mesh (triangles)
        # Note: GL viewer doesn't support per-vertex colors for meshes
        if self.model.tri_count:
            self.viewer.log_mesh(
                "/model/triangles",
                self.state_0.particle_q,
                self.model.tri_indices.flatten(),
                hidden=False,
                backface_culling=False,
            )
        
        # Render particles ONLY for selected ball
        if self.model.particle_count and hasattr(self, 'particle_radii_array'):
            self.viewer.log_points(
                name="/model/particles",
                points=self.state_0.particle_q,
                radii=self.particle_radii_array,  # Zero radius hides non-selected
                colors=self.particle_colors,
                hidden=False,
            )
        
        if self.contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)
        
        self.viewer.model_changed = False
        self.viewer.end_frame()
    
    def _render_plate_wireframe(self):
        """Render the plate as a grey wireframe for transparency effect."""
        body_q = self.state_0.body_q.numpy()
        if len(body_q) == 0:
            return
        
        # Get plate transform
        plate_transform = body_q[self.rigid_body_id]
        plate_pos = np.array([plate_transform[0], plate_transform[1], plate_transform[2]])
        plate_rot = np.array([plate_transform[3], plate_transform[4], plate_transform[5], plate_transform[6]])
        
        # Box half-extents
        hx = self.rigid_half_width
        hy = self.rigid_half_width
        hz = self.rigid_half_height
        
        # Box corners in local space
        corners_local = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],  # Bottom
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],      # Top
        ])
        
        # Rotate corners by quaternion (w, x, y, z format from transform)
        def quat_rotate(q, v):
            """Rotate vector v by quaternion q (x, y, z, w format)."""
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]
            # Quaternion rotation formula
            t = 2.0 * np.cross([qx, qy, qz], v)
            return v + qw * t + np.cross([qx, qy, qz], t)
        
        corners_world = np.array([quat_rotate(plate_rot, c) + plate_pos for c in corners_local])
        
        # Box edges (12 edges)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
        ]
        
        starts = [corners_world[e[0]] for e in edges]
        ends = [corners_world[e[1]] for e in edges]
        color = COLOR_GREY_TRANSPARENT
        
        starts_arr = wp.array(starts, dtype=wp.vec3, device=self.model.device)
        ends_arr = wp.array(ends, dtype=wp.vec3, device=self.model.device)
        colors_arr = wp.array([color] * len(edges), dtype=wp.vec3, device=self.model.device)
        
        self.viewer.log_lines("/model/plate_wireframe", starts_arr, ends_arr, colors_arr, width=0.01)
    
    
    def run(self, num_frames: int = 600):
        """Run simulation loop."""
        print(f"\n🎈 Starting inflatable table demo...", flush=True)
        print(f"   Solid mesh + particles for selected ball, wireframe plate!", flush=True)
        print(f"   Press 1-4 to select balls, I/K to inflate/deflate!", flush=True)
        
        for frame in range(num_frames):
            self.step()
            self.render()
            
            body_q = self.state_0.body_q.numpy()
            if len(body_q) > 0:
                rigid_height = body_q[self.rigid_body_id][2]
                self.max_rigid_height = max(self.max_rigid_height, rigid_height)
            
            if frame % 60 == 0:
                name = ["Red", "Green", "Blue", "Yellow"][self.selected_ball]
                print(f"   Frame {frame}: selected=Ball {self.selected_ball+1} ({name}), "
                      f"pressures={[f'{p:.1f}' for p in self.ball_pressures]}, "
                      f"plate_z={rigid_height:.3f}m", flush=True)
        
        print(f"\n🎈 Simulation complete!", flush=True)
        print(f"   Max plate height: {self.max_rigid_height:.3f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Inflatable Table - Per-Ball Inflation')
    
    parser.add_argument('--radius', type=float, default=0.25)
    parser.add_argument('--subdivisions', type=int, default=2)
    parser.add_argument('--interior_layers', type=int, default=2)
    parser.add_argument('--soft_mass', type=float, default=1.0)
    parser.add_argument('--rigid_width', type=float, default=3.0)
    parser.add_argument('--rigid_mass', type=float, default=0.004)
    parser.add_argument('--particle_radius', type=float, default=0.03)
    parser.add_argument('--k_mu', type=float, default=5.0e4)
    parser.add_argument('--k_lambda', type=float, default=5.0e4)
    parser.add_argument('--k_damp', type=float, default=50.0)
    parser.add_argument('--spring_ke', type=float, default=2.0e4)
    parser.add_argument('--spring_kd', type=float, default=20.0)
    parser.add_argument('--gravity', type=float, default=9.81)
    parser.add_argument('--max_pressure', type=float, default=5.0)
    parser.add_argument('--substeps', type=int, default=32)
    parser.add_argument('--num_frames', type=int, default=600)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    
    args = parser.parse_args()
    
    wp.init()
    
    with wp.ScopedDevice(args.device):
        if args.headless:
            viewer = None
        else:
            try:
                viewer = newton.viewer.ViewerGL(width=1024, height=768)
            except Exception as e:
                print(f"Could not create viewer: {e}")
                viewer = None
        
        example = Example(
            viewer=viewer,
            radius=args.radius,
            subdivisions=args.subdivisions,
            interior_layers=args.interior_layers,
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
            substeps=args.substeps,
        )
        example.run(num_frames=args.num_frames)
        
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
