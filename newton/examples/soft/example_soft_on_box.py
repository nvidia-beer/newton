# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Soft Object on Rigid Box Example

Demonstrates a soft (FEM) object sitting stably on top of a rigid XPBD box.
This showcases constraint-based contact handling between soft bodies and rigid shapes.

The example shows:
- A rigid box (XPBD solver)
- A soft sphere resting on top of the box
- Stable contact without penetration
- Constraint-based contact corrections

Usage:
    python -m newton.examples.soft.example_soft_on_box
"""

import warp as wp
import numpy as np
import argparse

import newton
from newton.solvers import SolverXPBD, SolverSoft, TetraSphere


# Colors
COLOR_BOX = (0.8, 0.6, 0.2)      # Rigid box (brown/orange)
COLOR_SOFT = (0.3, 0.5, 1.0)     # Soft sphere (blue)


class SoftOnBoxExample:
    """Soft object sitting on top of a rigid XPBD box."""
    
    def __init__(
        self,
        viewer,
        box_size: tuple = (2.0, 2.0, 0.1),  # (width, length, height) - thinner plate
        box_pos: tuple = (0.0, 0.0, 0.3),  # Box center position: set so plate stays above ground (0.3 = half_height + amplitude/2)
        soft_radius: float = 0.4,
        soft_height: float = 1.2,  # Height above ground for soft object center
        soft_subdivisions: int = 2,
        soft_interior_layers: int = 2,
        soft_mass: float = 1.0,
        box_mass: float = 10.0,  # Heavy box (stable)
        k_mu: float = 5e4,
        k_lambda: float = 5e4,
        k_damp: float = 2.0,
        gravity: float = 9.81,
        substeps: int = 16,
        use_constraint_contacts: bool = True,  # Use constraint-based contacts
        plate_amplitude: float = 0.5,  # Vertical movement amplitude (meters)
        plate_frequency: float = 0.5,  # Movement frequency (Hz)
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.plate_amplitude = plate_amplitude
        self.plate_frequency = plate_frequency
        self.plate_body_id = None  # Will be set when body is created
        self.plate_base_z = box_pos[2]  # Base Z position
        
        # Create kernel for updating plate position
        @wp.kernel
        def update_plate_position(
            body_q: wp.array(dtype=wp.transform),
            body_qd: wp.array(dtype=wp.spatial_vector),
            body_id: int,
            target_z: float,
            target_velocity_z: float,
        ):
            # Get current transform
            current_transform = body_q[body_id]
            current_pos = wp.transform_get_translation(current_transform)
            current_rot = wp.transform_get_rotation(current_transform)
            
            # Update position (keep X, Y the same, update Z)
            new_pos = wp.vec3(current_pos[0], current_pos[1], target_z)
            body_q[body_id] = wp.transform(new_pos, current_rot)
            
            # Update velocity (keep angular velocity, update linear Z)
            current_vel = body_qd[body_id]
            linear_vel = wp.spatial_top(current_vel)
            angular_vel = wp.spatial_bottom(current_vel)
            
            new_vel = wp.spatial_vector(
                wp.vec3(linear_vel[0], linear_vel[1], target_velocity_z),
                angular_vel
            )
            body_qd[body_id] = new_vel
        
        self.update_plate_position_kernel = update_plate_position
        
        print("=" * 70)
        print(f"  Soft Object on Rigid Box Example")
        print("=" * 70)
        print(f"\n  BROWN box: Rigid (XPBD)")
        print(f"  BLUE sphere: Soft/FEM")
        print(f"\n  Constraint-based contacts: {'Enabled' if use_constraint_contacts else 'Disabled'}")
        
        # Generate FEM mesh for soft sphere
        print(f"\nGenerating tetrahedral mesh...")
        self.sphere = TetraSphere(
            radius=soft_radius,
            subdivisions=soft_subdivisions,
            interior_layers=soft_interior_layers,
            verbose=True,
        )
        mesh_data = self.sphere.get_mesh_data()
        
        # Build unified model
        print(f"\n--- Building Unified Model ---")
        builder = newton.ModelBuilder()
        
        # Ground plane
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=5e5, kd=1e3, kf=1e4, mu=0.5)
        )
        
        # Dynamic plate (XPBD) - moves up and down
        box_width, box_length, box_height = box_size
        box_x, box_y, box_z = box_pos
        
        print(f"\nAdding dynamic plate:")
        print(f"  Size: {box_size}")
        print(f"  Base position: {box_pos}")
        print(f"  Mass: {box_mass} kg")
        print(f"  Movement: amplitude={plate_amplitude}m, frequency={plate_frequency}Hz")
        
        # Create dynamic plate body (XPBD) - will be moved directly
        box_body = builder.add_body(
            xform=wp.transform(wp.vec3(box_x, box_y, box_z), wp.quat_identity()),
            mass=box_mass,
        )
        
        # Add free joint for XPBD
        joint_id = builder.add_joint_free(box_body)
        builder.add_articulation([joint_id], key="dynamic_plate_articulation")
        
        # Store body index for direct position control
        self.plate_body_id = box_body
        
        # Add box shape
        box_volume = box_width * box_length * box_height
        builder.add_shape_box(
            body=box_body,
            hx=box_width / 2.0,
            hy=box_length / 2.0,
            hz=box_height / 2.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,
                kd=1e3,
                kf=1e4,
                mu=0.5,
                density=box_mass / box_volume,  # Set density for proper mass distribution
            ),
            key="rigid_box",
        )
        
        # Soft sphere (FEM)
        print(f"\nAdding soft sphere:")
        print(f"  Radius: {soft_radius}")
        print(f"  Center height: {soft_height}")
        print(f"  Mass: {soft_mass} kg")
        
        # Use add_soft_mesh() like example_bouncing_ball.py - this automatically adds triangles for rendering
        vertices = mesh_data["vertices"]
        indices = mesh_data["indices"]
        
        # Position sphere on top of box (center at box_x, box_y, soft_height)
        builder.add_soft_mesh(
            pos=wp.vec3(box_x, box_y, soft_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            scale=1.0,
            density=soft_mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )
        
        # Track particle start for springs
        particle_start = builder.particle_count - len(vertices)
        
        # Add springs between mesh vertices for stability (same as example_bouncing_ball.py)
        spring_ke = k_mu * 0.5
        spring_kd = k_damp * 0.5
        num_tets = len(indices) // 4
        
        vertex_positions = np.array(vertices)
        added_springs = set()
        
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
                        particle_start + i_local,
                        particle_start + j_local,
                        spring_ke,
                        spring_kd,
                        rest_length
                    )
        
        print(f"Added {len(added_springs)} unique springs")
        
        # Build model
        print(f"\nBuilding model...")
        self.model = builder.finalize(device="cuda")
        
        # Set contact parameters
        self.model.soft_contact_ke = 5e5
        self.model.soft_contact_kd = 1e3
        self.model.soft_contact_kf = 1e4
        self.model.soft_contact_mu = 0.5
        self.model.particle_adhesion = 0.0
        
        # Set particle collision radius (smaller = less collision margin)
        # This affects both collision detection and rendering
        particle_radius = 0.01  # Smaller than default (0.015 in bouncing_ball)
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, particle_radius),
            dtype=wp.float32,
            device=self.model.device
        )
        
        print(f"  Particles: {self.model.particle_count}")
        print(f"  Particle collision radius: {particle_radius}m")
        print(f"  Springs: {self.model.spring_count}")
        print(f"  Tetrahedra: {self.model.tet_count}")
        print(f"  Bodies: {self.model.body_count}")
        
        # Create solvers
        print(f"\n--- Creating Solvers ---")
        if use_constraint_contacts:
            print(f"  Using SolverSoft with constraint-based contacts")
            self.soft_solver = SolverSoft(
                model=self.model,
                dt=self.sim_dt,
                mass=soft_mass,
                solver_type="bicgstab",
                use_constraint_contacts=True,  # Enable constraint-based contacts!
                contact_relaxation=0.9,
            )
        else:
            print(f"  Using SolverSoft with force-based contacts")
            self.soft_solver = SolverSoft(
                model=self.model,
                dt=self.sim_dt,
                mass=soft_mass,
                solver_type="bicgstab",
                use_constraint_contacts=False,
            )
        
        # XPBD for rigid bodies
        print(f"  Using SolverXPBD for rigid bodies")
        self.xpbd_solver = SolverXPBD(
            self.model,
            iterations=2,
            soft_contact_relaxation=0.9,
        )
        
        # State buffers for dual solver
        self.state_soft = self.model.state()
        self.state_rigid = self.model.state()
        
        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # Initialize state
        print(f"\n--- Initializing State ---")
        wp.copy(self.state_0.particle_q, self.model.particle_q)
        wp.copy(self.state_0.particle_qd, self.model.particle_qd)
        wp.copy(self.state_0.body_q, self.model.body_q)
        wp.copy(self.state_0.body_qd, self.model.body_qd)
        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        
        # Evaluate forward kinematics for rigid bodies
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        # Set up viewer
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True  # Enable soft body visualization
            self.viewer.show_triangles = True  # Enable mesh surface rendering
        
        print(f"\n✓ Setup complete!")
        print(f"\n  The soft sphere should settle on top of the rigid box.")
        print(f"  Watch for stable contact without penetration.")
    
    def step(self):
        """Step simulation forward."""
        # Clear forces
        self.state_0.clear_forces()
        
        # Collision detection
        contacts = self.model.collide(self.state_0)
        
        # Step 1: SolverSoft integrates particles (with internal forces)
        self.soft_solver.step(
            self.state_0, self.state_soft, self.control, contacts, self.sim_dt
        )
        
        # Step 2: XPBD integrates rigid bodies
        self.xpbd_solver.step(
            self.state_0, self.state_rigid, self.control, contacts, self.sim_dt
        )
        
        # Combine results: particles from SolverSoft, bodies from XPBD
        wp.copy(self.state_1.particle_q, self.state_soft.particle_q)
        wp.copy(self.state_1.particle_qd, self.state_soft.particle_qd)
        wp.copy(self.state_1.body_q, self.state_rigid.body_q)
        wp.copy(self.state_1.body_qd, self.state_rigid.body_qd)
        wp.copy(self.state_1.joint_q, self.state_rigid.joint_q)
        wp.copy(self.state_1.joint_qd, self.state_rigid.joint_qd)
        
        # Update plate position: sinusoidal motion up and down (after solver step)
        if self.plate_body_id is not None:
            # Calculate target Z position: sinusoidal motion around base position
            # Ensure plate never goes below ground (bottom of plate >= 0.0)
            plate_half_height = 0.05  # Half of plate height (0.1m / 2)
            min_center_z = plate_half_height  # Minimum center Z to keep bottom of plate at ground level
            target_z_unclamped = self.plate_base_z + self.plate_amplitude * np.sin(2.0 * np.pi * self.plate_frequency * self.sim_time)
            target_z = max(min_center_z, target_z_unclamped)  # Clamp to stay above ground
            
            # Calculate velocity (set to 0 if clamped to avoid sudden stops)
            if target_z_unclamped >= min_center_z:
                target_velocity_z = 2.0 * np.pi * self.plate_frequency * self.plate_amplitude * np.cos(2.0 * np.pi * self.plate_frequency * self.sim_time)
            else:
                target_velocity_z = 0.0  # Stop when hitting ground
            
            # Use kernel to update plate position and velocity
            wp.launch(
                self.update_plate_position_kernel,
                dim=1,
                inputs=[
                    self.state_1.body_q,
                    self.state_1.body_qd,
                    self.plate_body_id,
                    target_z,
                    target_velocity_z,
                ],
                device=self.model.device,
            )
        
        # Swap states
        self.state_0, self.state_1 = self.state_1, self.state_0
        
        self.sim_time += self.sim_dt
    
    def render(self):
        """Render the simulation."""
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            contacts = self.model.collide(self.state_0)
            if contacts:
                self.viewer.log_contacts(contacts, self.state_0)
            self.viewer.end_frame()
    
    def run(self, num_frames: int = 2000):
        """Run simulation loop."""
        print(f"\n--- Starting Simulation ---")
        print(f"  The soft sphere should settle on top of the rigid box\n")
        
        for frame in range(num_frames):
            # Step simulation (multiple substeps per frame)
            for _ in range(self.substeps):
                self.step()
            
            # Render
            self.render()
            
            if frame % 100 == 0:
                print(f"  Frame {frame}/{num_frames}, sim_time={self.sim_time:.2f}s")
        
        print(f"\nSimulation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Soft object on rigid box example")
    parser.add_argument(
        "--constraint-contacts",
        action="store_true",
        help="Use constraint-based contacts (prevents penetration)",
    )
    parser.add_argument(
        "--no-constraint-contacts",
        action="store_true",
        help="Use force-based contacts (reactive)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--num-frames", type=int, default=2000, help="Number of frames to simulate")
    parser.add_argument("--device", type=str, default=None, help="Compute device")
    args = parser.parse_args()
    
    use_constraint = args.constraint_contacts or not args.no_constraint_contacts
    
    # Initialize Warp
    wp.init()
    
    with wp.ScopedDevice(args.device):
        # Create viewer - use OpenGL viewer without explicit size (use defaults like other examples)
        if args.headless:
            viewer = None
            print("Running in headless mode (no GUI)")
        else:
            import os
            display = os.environ.get("DISPLAY")
            print(f"DISPLAY={display}")
            
            try:
                # Use ViewerGL with default size (matches other examples)
                viewer = newton.viewer.ViewerGL(
                    width=1920,
                    height=1080,
                )
                print("✓ ViewerGL created successfully")
                
                # Try to ensure window is visible and process initial events
                if hasattr(viewer, 'renderer') and hasattr(viewer.renderer, 'window'):
                    window = viewer.renderer.window
                    print(f"  Window created: {window.width}x{window.height}, visible={window.visible}")
                    
                    # Process events to make window appear
                    if hasattr(viewer.renderer, 'update'):
                        viewer.renderer.update()
                    
                    if hasattr(window, 'set_visible'):
                        window.set_visible(True)
                    if hasattr(window, 'activate'):
                        window.activate()
                    if hasattr(window, 'switch_to'):
                        window.switch_to()
                    
                    print(f"✓ Window should be visible now")
            except Exception as e:
                print(f"✗ Could not create OpenGL viewer: {e}")
                import traceback
                traceback.print_exc()
                try:
                    # Fall back to Rerun
                    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
                    print("✓ ViewerRerun created successfully (check browser at http://localhost:9090)")
                except Exception as e2:
                    print(f"✗ Could not create Rerun viewer: {e2}")
                    print("Running headless...")
                    viewer = None
        
        # Create example
        example = SoftOnBoxExample(
            viewer=viewer,
            use_constraint_contacts=use_constraint,
        )
        
        # Render initial frame to ensure window is visible
        if viewer:
            print("Rendering initial frame to show window...")
            example.render()
        
        # Run simulation
        example.run(num_frames=args.num_frames)
        
        # Keep viewer open if it exists (process events to keep window visible)
        if viewer:
            print("\nSimulation complete - viewer window will stay open")
            print("Press ESC or close the window to exit")
            while viewer.is_running():
                viewer.end_frame()  # Process events and render
            viewer.close()


if __name__ == "__main__":
    main()
