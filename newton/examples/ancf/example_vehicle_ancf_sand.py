# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Vehicle on 4 ANCF FEM tires driving on implicit-MPM sand (vehicle from --vehicle-asset).

Extends vehicle_ancf_tires (3): a granular sand bed (SolverImplicitMPM) sits on
the floor (z in [0, h]) and the vehicle starts on top of it; the analytic ANCF ground
plane stays at the floor so the tires are always supported.

Coupling per frame (after the existing 10 ANCF/MuJoCo substeps)
  1. ANCF node positions/velocities -> sand-model tire particles (Y-up -> Z-up)
  2. refit the 4 deformable tire collider meshes (coupling_notify_input_state_update)
  3. MPM step at the frame rate over the whole bed
  4. collider impulses -> barycentric vertex forces -> ANCF node_f_ext_persistent,
     held constant over the next frame's substeps.  accumulate_wheel_wrenches sums
     global_f_ext, so the sand reaction also reaches the spindle.

Only the 4 tire meshes (and the bed floor) are registered as MPM colliders; the
chassis and suspension never touch the sand, so the sand lives in its own Model.

Command: python -m newton.examples vehicle_ancf_sand --vehicle-asset <usd>
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp
import warp.fem as fem

import newton
import newton.examples
from newton.examples.ancf import example_vehicle_ancf_tires as _dw
from newton.solvers import SolverImplicitMPM

# ── Sand bed defaults ─────────────────────────────────────────────────────────
_SAND_SIZE = (12.0, 6.0, 0.4)  # length x, width y, depth h [m]; top surface at z=0
_VOXEL = 0.05  # MPM grid voxel [m]
_PPC_AXIS = 2  # particles per voxel per axis (8 per cell)
_SAND_RHO = 1500.0  # [kg/m^3]
_SAND_FRIC = 0.68  # Drucker-Prager friction, ~34 deg angle of repose
_FLOOR_FRIC = 0.8
_MPM_ITERS = 25
_RENDER_STRIDE = 8  # draw one particle per cell (8 ppc)
# Whole-bed MPM steps at init so the sand is in static equilibrium before the vehicle lands on it.
_PRESETTLE_STEPS = 30


# ── Kernels ───────────────────────────────────────────────────────────────────


@wp.kernel
def _ancf_to_sand_particles(
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    offset: int,
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    """ANCF Y-up -> sand model Z-up: (x,y,z) -> (z,x,y)."""
    i = wp.tid()
    p = node_x[i]
    v = node_xd[i]
    q[offset + i] = wp.vec3(p[2], p[0], p[1])
    qd[offset + i] = wp.vec3(v[2], v[0], v[1])


@wp.kernel
def _sand_force_to_ancf(f_zu: wp.array[wp.vec3], offset: int, f_ext: wp.array[wp.vec3]):
    """Z-up -> ANCF Y-up: (x,y,z) -> (y,z,x).  Overwrites: f_ext is exactly last frame's sand reaction."""
    i = wp.tid()
    f = f_zu[offset + i]
    f_ext[i] = wp.vec3(f[1], f[2], f[0])


@wp.kernel
def _sum_tire_force(f_zu: wp.array[wp.vec3], offset: int, n_nodes: int, out: wp.array[wp.vec3]):
    i = wp.tid()
    wp.atomic_add(out, i // n_nodes, f_zu[offset + i])


@wp.kernel
def _gather_stride(src: wp.array[wp.vec3], stride: int, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i * stride]


# ── Example ───────────────────────────────────────────────────────────────────


class Example(_dw.Example):
    """Vehicle (--vehicle-asset) + 4 ANCF tires on implicit-MPM sand."""

    def __init__(self, viewer=None, args=None):
        if args is None:
            args = argparse.Namespace()
        # Floor at z=0 (ANCF analytic plane + MJCF plane), sand bed z in [0, h] on top of it, car
        # lifted by h so the treads start at the sand surface.  The floor stays a contact surface
        # so the tires are supported if they leave the bed or sink through it.
        args.ground_z = 0.0
        args.world_z_offset = float(args.sand_size[2])
        super().__init__(viewer, args)
        self._build_sand(args)

    # ── Sand model / MPM solver ───────────────────────────────────────────────

    def _build_sand(self, args) -> None:
        dev = "cuda:0"
        n_nodes = self._n_nodes
        n_tire_pts = _dw._N_TIRES * n_nodes

        size = tuple(float(v) for v in args.sand_size)
        voxel = float(args.sand_voxel)
        ppc_axis = int(args.sand_ppc)
        rho = float(args.sand_rho)
        fric = float(args.sand_friction)
        mpm_iters = int(args.mpm_iters)
        sand_h = size[2]
        self._sand_h = sand_h

        builder = newton.ModelBuilder()  # Z-up
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = _FLOOR_FRIC
        builder.add_ground_plane()

        # Bed on the ground plane: z in [0, sand_h]; the vehicle was lifted by sand_h.
        cell = voxel / ppc_axis
        lo = np.array([-0.5 * size[0], -0.5 * size[1], 0.0])
        res = np.ceil(np.array([size[0], size[1], sand_h]) / cell).astype(int)
        r_p = 0.5 * cell
        builder.add_particle_grid(
            pos=wp.vec3(*lo.tolist()),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]),
            dim_y=int(res[1]),
            dim_z=int(res[2]),
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            mass=cell**3 * rho,
            jitter=0.5 * r_p,  # breaks grid aliasing without a visibly rough surface
            radius_mean=r_p,
            custom_attributes={"mpm:friction": fric},
        )
        n_sand = builder.particle_count
        self._n_sand = n_sand

        # Tire vertices (Z-up, already mirrored into the car model for rendering).
        tire_zu = self.state_0.particle_q.numpy()[:n_tire_pts]
        builder.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in tire_zu],
            vel=[(0.0, 0.0, 0.0)] * n_tire_pts,
            mass=[0.0] * n_tire_pts,
            radius=[0.001] * n_tire_pts,
        )

        tris = self._closed_tire_triangles()
        meshes = []
        particle_ids = []
        for e in range(_dw._N_TIRES):
            pts = wp.array(tire_zu[e * n_nodes : (e + 1) * n_nodes], dtype=wp.vec3, device=dev)
            meshes.append(wp.Mesh(pts, wp.array(tris.flatten(), dtype=wp.int32, device=dev), wp.zeros_like(pts)))
            particle_ids.append(list(range(n_sand + e * n_nodes, n_sand + (e + 1) * n_nodes)))

        self.sand_model = builder.finalize(device=dev)
        self.sand_state = self.sand_model.state()

        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = voxel
        cfg.grid_type = "fixed"
        cfg.grid_padding = 8
        # Active-cell budget: warp.fem silently truncates the partition beyond it (and prints
        # "Number of elements exceeded ..."), so derive it from the whole bed.
        n_bed_cells = int(np.prod(np.ceil(np.array([size[0], size[1], sand_h]) / voxel)))
        cfg.max_active_cell_count = int(1.5 * n_bed_cells)
        cfg.strain_basis = "P0"
        # Collider basis == velocity basis skips the collider->velocity coupling matrix, which
        # with "S2" was a 22 M-entry BSR rebuilt (two 64-bit radix sorts) every step: 38 ms of
        # the 120 ms GPU frame at 1.84 M particles (nsys 2026-09-11).
        cfg.collider_basis = "Q1"
        cfg.max_iterations = mpm_iters
        cfg.critical_fraction = 0.0
        cfg.air_drag = 1.0
        cfg.transfer_scheme = "pic"
        cfg.collider_velocity_mode = "forward"
        self.mpm = SolverImplicitMPM(self.sand_model, cfg, temporary_store=fem.TemporaryStore())
        n_t = _dw._N_TIRES
        self.mpm.setup_collider(
            collider_meshes=[*meshes, None],
            collider_body_ids=[None] * n_t + [-1],
            collider_margins=[0.5 * voxel] * n_t + [None],
            collider_friction=[float(getattr(args, "mu", _dw._MU))] * n_t + [_FLOOR_FRIC],
            collider_particle_ids=[*particle_ids, None],
        )

        # Pre-settle the whole bed before the vehicle starts interacting with it.
        n_presettle = int(args.presettle_steps)
        for _ in range(n_presettle):
            self.mpm.step(self.sand_state, self.sand_state, None, None, _dw._FRAME_DT)

        # Render buffers, allocated once.  The GL viewer draws each point as an instanced
        # 72-triangle sphere and rebuilds a 4x4 transform per instance every frame, so only
        # every `render_stride`-th particle is drawn (8 = one per cell).  Colors are uploaded
        # through the host, so they are sent on the first frame only.
        self._render_stride = max(1, int(args.render_stride))
        n_draw = n_sand // self._render_stride
        self._sand_draw = wp.zeros(n_draw, dtype=wp.vec3, device=dev)
        self._sand_radii = wp.full(n_draw, 0.35 * voxel, dtype=wp.float32, device=dev)
        self._sand_colors = wp.full(n_draw, wp.vec3(0.72, 0.60, 0.42), dtype=wp.vec3, device=dev)
        self._sand_colors_sent = False

        self._tire_f = wp.zeros(self.sand_model.particle_count, dtype=wp.vec3, device=dev)
        self._tire_f_sum = wp.zeros(n_t, dtype=wp.vec3, device=dev)
        self._gui_sand_fz = [0.0] * n_t
        self._gui_sink = [0.0] * n_t

        print(
            f"[SAND] bed {size[0]:g}x{size[1]:g}x{sand_h:g} m  voxel={voxel:g}  cell={cell:.4f}  "
            f"particles={n_sand:,}  tire_pts={n_tire_pts}  "
            f"cell_budget={cfg.max_active_cell_count:,}  presettle={n_presettle} steps"
        )

        # Warm-up (JIT, grid allocation) then capture the whole sand step as one graph.
        self._sand_graph = None
        self._sand_step()
        wp.synchronize_device(dev)
        try:
            wp.capture_begin(device=dev)
            self._sand_step()
            self._sand_graph = wp.capture_end(device=dev)
            print("[SAND] sand-step graph captured (1 launch/frame)")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._sand_graph = None
            print(f"[SAND] sand-step graph capture failed ({e!r}) — eager mode")
        wp.synchronize_device(dev)

        if self.viewer is not None:
            # Closer/lower than the base vehicle camera (self.spec.camera_pos), to frame the
            # sand bed as well as the vehicle.
            cx, cy, cz = self.spec.camera_pos
            self.viewer.set_camera(pos=wp.vec3(0.7 * cx, 0.7 * cy, 0.7 * cz), pitch=-20.0, yaw=50.0)

    def _closed_tire_triangles(self) -> np.ndarray:
        """Shell triangles plus a rim cylinder between the two bead rings, all oriented outward.

        The ANCF shell is open at the beads; without a closed surface the SDF sign (average
        face normal) is undefined there and sand enters the cavity.  Rest frame is Y-up with
        the axle along X: the shell's outward normal is +radial, the rim's is -radial.
        """
        x0 = self.ancf_model.node_x0.numpy()
        en = self.ancf_model.elem_nodes.numpy()
        shell = np.empty((len(en) * 2, 3), dtype=np.int64)
        shell[0::2] = en[:, [0, 1, 2]]
        shell[1::2] = en[:, [0, 2, 3]]

        ax = x0[:, 0]
        left = np.where(np.isclose(ax, ax.min(), atol=1e-4))[0]
        right = np.where(np.isclose(ax, ax.max(), atol=1e-4))[0]
        assert len(left) == len(right), f"bead rings differ: {len(left)} vs {len(right)}"
        ang = np.arctan2(x0[:, 2], x0[:, 1])
        left = left[np.argsort(ang[left])]
        right = right[np.argsort(ang[right])]
        n = len(left)
        nxt = np.roll(np.arange(n), -1)
        rim = np.concatenate(
            [
                np.stack([left, left[nxt], right], axis=1),
                np.stack([left[nxt], right[nxt], right], axis=1),
            ]
        )

        def orient(tris: np.ndarray, sign: float) -> np.ndarray:
            p = x0[tris]
            normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            c = p.mean(axis=1)
            radial = np.stack([np.zeros(len(c)), c[:, 1], c[:, 2]], axis=1)
            flip = np.einsum("ij,ij->i", normal, radial) * sign < 0.0
            tris = tris.copy()
            tris[flip] = tris[flip][:, [0, 2, 1]]
            return tris

        return np.concatenate([orient(shell, +1.0), orient(rim, -1.0)]).astype(np.int32)

    # ── Per-frame sand step ───────────────────────────────────────────────────

    def _sand_step(self) -> None:
        dev = "cuda:0"
        ancf = self.ancf_solver
        n = _dw._N_TIRES * self._n_nodes
        n_sand = self._n_sand

        wp.launch(
            _ancf_to_sand_particles,
            dim=n,
            inputs=[ancf.node_x, ancf.node_xd, n_sand, self.sand_state.particle_q, self.sand_state.particle_qd],
            device=dev,
        )
        self.mpm.coupling_notify_input_state_update(
            self.sand_state, newton.StateFlags.PARTICLE_Q | newton.StateFlags.PARTICLE_QD
        )

        self.mpm.step(self.sand_state, self.sand_state, None, None, _dw._FRAME_DT)

        self._tire_f.zero_()
        self.mpm.collect_deformable_collider_particle_forces(self.sand_state, _dw._FRAME_DT, self._tire_f)
        wp.launch(
            _sand_force_to_ancf,
            dim=n,
            inputs=[self._tire_f, n_sand, ancf.node_f_ext_persistent],
            device=dev,
        )

    def simulate(self) -> None:
        super().simulate()
        if self._sand_graph is not None:
            wp.capture_launch(self._sand_graph)
        else:
            self._sand_step()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _print_diag(self, fps: float = 0.0) -> None:
        super()._print_diag(fps)
        self._tire_f_sum.zero_()
        wp.launch(
            _sum_tire_force,
            dim=_dw._N_TIRES * self._n_nodes,
            inputs=[self._tire_f, self._n_sand, self._n_nodes, self._tire_f_sum],
            device="cuda:0",
        )
        f_sum = self._tire_f_sum.numpy()
        xpos_all = self.solver.xpos.numpy()
        smj = self._spindle_mj_arr.numpy()
        for label, e in _dw._WHEEL_ORDER:
            sink = self.spec.tire_R_outer + self._sand_h - float(xpos_all[0, int(smj[e])][2])
            self._gui_sand_fz[e] = float(f_sum[e][2])
            self._gui_sink[e] = sink
            print(
                f"  {label}: sand F=({f_sum[e][0]:+8.0f},{f_sum[e][1]:+8.0f},{f_sum[e][2]:+8.0f}) N"
                f"  hub_drop={sink * 1e3:6.1f} mm"
            )

    def gui(self, ui) -> None:
        super().gui(ui)
        ui.separator()
        ui.text(f"Sand  {self._n_sand:,} particles   (tire loads updated every --diag-period frames)")
        for label, e in _dw._WHEEL_ORDER:
            ui.text(f"  {label}  Fz_sand={self._gui_sand_fz[e]:+.0f} N  hub_drop={self._gui_sink[e] * 1e3:.1f} mm")

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self._t)
        self.viewer.log_state(self.state_0)
        wp.launch(
            _gather_stride,
            dim=self._sand_draw.shape[0],
            inputs=[self.sand_state.particle_q, self._render_stride, self._sand_draw],
            device="cuda:0",
        )
        self.viewer.log_points(
            "sand",
            self._sand_draw,
            radii=self._sand_radii,
            colors=None if self._sand_colors_sent else self._sand_colors,
        )
        self._sand_colors_sent = True
        self.viewer.log_lines("bead_rings", self._ring_line_s, self._ring_line_e, colors=(1.0, 0.45, 0.0))
        self.viewer.log_lines("bead_spokes", self._spoke_start_zu, self._bead_pos_zu, colors=(1.0, 0.90, 0.1))
        wp.launch(
            _dw._gather_contact_spikes,
            dim=_dw._N_TIRES * self._n_nodes,
            inputs=[
                self.ancf_solver.node_x,
                self._n_nodes,
                self._sand_h,
                self._contact_vis_scale,
                self._contact_line_s,
                self._contact_line_e,
            ],
            device="cuda:0",
        )
        self.viewer.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, colors=(0.0, 1.0, 1.0))
        self.viewer.end_frame()

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_final(self) -> None:
        super().test_final()
        sand_q = self.sand_state.particle_q.numpy()[: self._n_sand]
        assert np.all(np.isfinite(sand_q)), "non-finite sand particle positions"
        xpos_all = self.solver.xpos.numpy()
        smj = self._spindle_mj_arr.numpy()
        for e in range(_dw._N_TIRES):
            drop = self.spec.tire_R_outer + self._sand_h - float(xpos_all[0, int(smj[e])][2])
            assert 0.005 < drop < 0.3, f"FAIL tire {e}: hub drop {drop * 1e3:.1f} mm not in (5, 300) mm"
        print(f"[PASS] sand: {self._n_sand:,} particles finite, all 4 tires supported by the bed")

    # ── Parser ────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = _dw.Example.create_parser()
        parser.add_argument(
            "--sand-size",
            type=float,
            nargs=3,
            default=list(_SAND_SIZE),
            metavar=("X", "Y", "H"),
            help="Sand bed length, width and depth [m]; the surface is at z=0.",
        )
        parser.add_argument("--sand-voxel", type=float, default=_VOXEL, help="MPM voxel size [m].")
        parser.add_argument(
            "--sand-ppc", type=int, default=_PPC_AXIS, help="Particles per voxel per axis (cubed per cell)."
        )
        parser.add_argument("--sand-rho", type=float, default=_SAND_RHO, help="Sand density [kg/m^3].")
        parser.add_argument("--sand-friction", type=float, default=_SAND_FRIC, help="Sand internal friction.")
        parser.add_argument("--mpm-iters", type=int, default=_MPM_ITERS, help="Implicit MPM solver iterations.")
        parser.add_argument(
            "--presettle-steps",
            type=int,
            default=_PRESETTLE_STEPS,
            help="Whole-bed MPM steps at init before the vehicle lands on the bed (0 = none).",
        )
        parser.add_argument(
            "--render-stride",
            type=int,
            default=_RENDER_STRIDE,
            help="Draw every N-th sand particle (instanced spheres; 8 = one per cell). 1 draws all.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
