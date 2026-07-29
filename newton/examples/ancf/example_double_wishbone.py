# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ANCF FEDA – Full Double-Wishbone
#
# FED Alpha (FEDA) tactical vehicle with the full double-wishbone suspension
# mechanics from the Project Chrono FEDA_Full reference model.  All geometry
# is built from primitives (no OBJ/mesh files).
#
# Hardpoints ported from:
#   suspension/FEDA_DoubleWishboneFront/Rear.cpp
#   vehicle/FEDA_Vehicle.cpp
#   steering/FEDA_PitmanArm.cpp
#
# Suspension topology (per corner):
#   Chassis ──hinge──> UCA ──ball──> Upright ──hinge Y──> Spindle/Wheel
#   Chassis ──hinge──> LCA
#   equality connect: LCA_U ↔ Upright    (closes wishbone loop)
#   TSDA tendon: chassis site ↔ LCA site  (k=76 kN/m, c=38 kN·s/m)
#
# Pitman-arm steering:
#   Chassis ──motor hinge Z──> pitman_arm ──ball──> steering_link
#   Chassis ──hinge Z──> idler_arm
#   equality: idler_arm ↔ steering_link (closes idler loop)
#   equality: steering_link ↔ upright FL/FR (front tierods)
#   equality: chassis ↔ upright RL/RR (rear fixed tierods)
#
# GUI:  steering slider (±27.5°)  +  throttle slider (rad/s AWD)
# Solver: SolverMuJoCo with CUDA graph capture.
#
# Command: python -m newton.examples double_wishbone
###########################################################################

import os

import numpy as np
import warp as wp

import newton
import newton.examples

# ── Vehicle geometry constants ────────────────────────────────────────────
_R = 0.499  # outer tyre radius [m]
_WB_H = 1.651  # half-wheelbase [m]  (full WB 3.302 m)
_TR_H = 0.97663  # half-track [m]

# GUI limits
_MAX_STEER = 0.47947  # ±27.5° in radians (pitman arm travel)
_MAX_SPEED = 12.0  # wheel angular speed [rad/s]
_WHEEL_SPEED_RATE = 0.1  # [rad/s per frame] max speed change per step() call


# ── Drive kernel ──────────────────────────────────────────────────────────


@wp.kernel
def _drive_feda(
    steer_dof: int,
    cmd: wp.array[wp.float32],  # [0] steer_angle [rad], [1] wheel_speed [rad/s]
    throttle_dofs: wp.array[wp.int32],
    joint_target_pos: wp.array[wp.float32],
    joint_target_vel: wp.array[wp.float32],
):
    """Write steering position and wheel-speed targets into control arrays.

    One thread per throttle DOF; thread 0 also writes the steering target.
    Reads ``cmd`` from a device buffer so slider changes work inside the
    captured CUDA graph without breaking the capture.
    """
    tid = wp.tid()
    if tid == 0:
        joint_target_pos[steer_dof] = cmd[0]
    joint_target_vel[throttle_dofs[tid]] = cmd[1]


# ── Helpers ───────────────────────────────────────────────────────────────


def _find_dof(builder: newton.ModelBuilder, joint_name: str) -> int:
    """Return the DOF index of the single-DOF joint whose last path segment matches ``joint_name``."""
    for j, label in enumerate(builder.joint_label):
        if label.split("/")[-1] == joint_name:
            return int(builder.joint_qd_start[j])
    raise KeyError(f"joint '{joint_name}' not found in MJCF")


# ── Example class ─────────────────────────────────────────────────────────


class Example:
    """FEDA with full double-wishbone + Pitman-arm steering, rigid wheels."""

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.steer_angle = float(getattr(args, "steer_angle", 0.0))
        self.wheel_speed = float(getattr(args, "wheel_speed", 3.0))
        self._target_wheel_speed = self.wheel_speed

        # ── Load vehicle from MJCF ────────────────────────────────────────
        car = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car)
        car.default_shape_cfg.mu = 0.9

        asset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "double_wishbone_rigid.xml",
        )
        car.add_mjcf(asset_path, up_axis="Z")

        self.steer_dof = _find_dof(car, "steer_motor")
        throttle_names = ("axle_fl", "axle_fr", "axle_rl", "axle_rr")
        throttle_dofs = [_find_dof(car, n) for n in throttle_names]

        # ── Build world ───────────────────────────────────────────────────
        builder = newton.ModelBuilder()
        builder.add_world(car)
        builder.default_shape_cfg.mu = 0.9
        builder.add_ground_plane()

        self.model = builder.finalize()

        # ── Solver ────────────────────────────────────────────────────────
        use_mujoco_contacts = getattr(args, "use_mujoco_contacts", True)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            iterations=50,
            ls_iterations=10,
            njmax=500,
            nconmax=128,
            use_mujoco_contacts=use_mujoco_contacts,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.use_mujoco_contacts = use_mujoco_contacts
        if use_mujoco_contacts:
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            self.contacts = self.model.contacts()

        # Device buffers for the captured graph.
        self.throttle_dofs = wp.array(throttle_dofs, dtype=wp.int32, device=self.model.device)
        self._cmd_host = np.zeros(2, dtype=np.float32)
        self.cmd = wp.zeros(2, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(-12.0, -18.0, 9.0),
            pitch=-22.0,
            yaw=48.0,
        )

        self._capture_graph()

    def _capture_graph(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._simulate()
            self.graph = cap.graph

    def _simulate(self):
        wp.launch(
            _drive_feda,
            dim=len(self.throttle_dofs),
            inputs=[self.steer_dof, self.cmd, self.throttle_dofs],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
        )

        if not self.use_mujoco_contacts:
            self.model.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        d = self._target_wheel_speed - self.wheel_speed
        if abs(d) <= _WHEEL_SPEED_RATE:
            self.wheel_speed = self._target_wheel_speed
        else:
            self.wheel_speed += _WHEEL_SPEED_RATE if d > 0.0 else -_WHEEL_SPEED_RATE
        self._cmd_host[0] = self.steer_angle
        self._cmd_host[1] = self.wheel_speed
        self.cmd.assign(self._cmd_host)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.sim_time += self.frame_dt

    def gui(self, ui):
        """ImGui side panel: live steering + throttle sliders."""
        ui.text("FEDA Double-Wishbone Controls")
        ui.separator()

        changed, value = ui.slider_float("steering [rad]", self.steer_angle, -_MAX_STEER, _MAX_STEER)
        if changed:
            self.steer_angle = float(value)

        changed, value = ui.slider_float("throttle [rad/s]", self._target_wheel_speed, -_MAX_SPEED, _MAX_SPEED)
        if changed:
            self._target_wheel_speed = float(value)

        ui.separator()
        fwd_speed = self.wheel_speed * _R
        ui.text(f"forward speed ~ {fwd_speed:+.2f} m/s")

        if abs(self.steer_angle) > 1e-3:
            turn_radius = (2.0 * _WB_H) / np.tan(abs(self.steer_angle))
            ui.text(f"turn radius   ~ {turn_radius:.2f} m")
        else:
            ui.text("turn radius   ~ straight")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all FEDA bodies remain above the ground plane",
            lambda q, qd: q[2] > -0.05,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.add_argument(
            "--wheel-speed",
            type=float,
            default=3.0,
            help=f"Initial wheel angular speed [rad/s].  Forward ≈ speed × {_R:.3f} m.",
        )
        parser.add_argument(
            "--steer-angle",
            type=float,
            default=0.0,
            help=f"Initial steering angle [rad].  Range ±{_MAX_STEER:.5f} (±27.5°).  Positive = left turn.",
        )
        parser.set_defaults(use_mujoco_contacts=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
