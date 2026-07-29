# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Example ANCF Rim MPM
#
# FEDA ANCF3423 shell tire on a rigid rim, dropped onto implicit MPM
# granular sand.  Combines:
#   • ancf_rim_shell  — hub/bead/rim mechanics (bead prescription,
#                        wrench accumulation, hub vertical integration)
#   • ancf_mpm        — MPM sand terrain (tread sphere colliders,
#                        frame-rate MPM step, impulse → f_ext coupling)
#
# The rim kernels are imported directly from example_ancf_rim_shell so
# hub/bead physics stays identical.
#
# Simulate loop (per substep)
#   1. prescribe_bead_batched   — pin beads to hub (Dirichlet)
#   2. advance_phi_and_z        — advance kinematic spin + forward motion
#   3. solver.graph_step()      — ANCF FEM (pre-compiled inner graph)
#   4. prescribe_bead_batched   — corrector re-pin
#   5. accum_bead_wrench        — read bead reactions from global_f_int
#   6. integrate_hub_dof        — hub Y + spin integration (corner mass)
#
# Per frame (after all substeps)
#   7. write ANCF tread positions → MPM sphere body_q
#   8. MPM step  (60 Hz)
#   9. collect MPM impulses → f_ext_persistent for next frame
#
# Command: python -m newton.examples ancf_rim_mpm
###########################################################################
from __future__ import annotations

import json
import math
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.ancf_shell import (
    SolverANCFShell,
    build_ancf_tire_mesh,
    isotropic_ancf_material,
)

# Import rim/hub kernels and geometry constants from ancf_rim_shell
from newton.examples.ancf.example_ancf_rim_shell import (
    _BUILD_PRESSURE,
    _G,
    _GAUGE_RATE,
    _MAX_GAUGE,
    _NU_BALLOON,
    _R_INNER,
    _R_OUTER,
    _RHO_BALLOON,
    _RPM_RATE,
    _SEC_DIVS,
    _THICK,
    _WIDTH,
    _accum_bead_wrench_kinematic,
    _advance_phi_and_z_batched,
    _build_ring_lines,
    _gather_bead_positions,
    _gather_contact_spikes_yup,
    _integrate_hub_dof,
    _prescribe_bead_batched,
)
from newton.solvers import SolverImplicitMPM

# ── MPM / Sand defaults ─────────────────────────────────────────────────────────
_SAND_H = 0.15  # sand pile height above Y=0 [m]
_SAND_HX = 0.80  # half-width X [m]
_SAND_HZ = 0.80  # half-depth Z [m]
_VOXEL = 0.04  # MPM voxel size [m]
_SAND_RHO = 1500.0
_SAND_FRIC = 0.68
_DROP_CLR = 0.10  # clearance of tread above sand top [m]


# ── MPM coupling kernels ────────────────────────────────────────────────────────


@wp.kernel
def _copy_tread_q(
    body_q: wp.array[wp.transform],
    node_x: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
):
    i = wp.tid()
    body_q[i] = wp.transform(node_x[tread_idx[i]], wp.quat_identity())


@wp.kernel
def _copy_tread_qd(
    body_qd: wp.array[wp.spatial_vector],
    node_xd: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
):
    i = wp.tid()
    v = node_xd[tread_idx[i]]
    body_qd[i] = wp.spatial_vector(v, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def _scatter_mpm_impulses(
    ids: wp.array[int],
    impulses: wp.array[wp.vec3],
    n: int,
    out: wp.array[wp.vec3],
):
    k = wp.tid()
    cid = ids[k]
    if cid >= 0 and cid < n:
        wp.atomic_add(out, cid, impulses[k])


@wp.kernel
def _apply_mpm_f_ext(
    f_ext: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
    f_node: wp.array[wp.vec3],  # uniform per-node force (N)
):
    """Add uniformly distributed MPM reaction force to ANCF f_ext_persistent."""
    i = wp.tid()
    wp.atomic_add(f_ext, tread_idx[i], f_node[i])


@wp.kernel
def _build_mpm_spikes(
    node_x: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
    impulse: wp.array[wp.vec3],
    scale: float,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    i = wp.tid()
    p = node_x[tread_idx[i]]
    mag = wp.length(impulse[i])
    starts[i] = p
    ends[i] = wp.vec3(p[0], p[1] + mag * scale, p[2])


# ── Example ────────────────────────────────────────────────────────────────────


class Example:
    """FEDA ANCF rim-tire on MPM granular sand — driven or free spin."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(getattr(args, "substeps", 10))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        dev = "cuda:0"

        # ── Per-tire config ───────────────────────────────────────────────
        tires_raw = getattr(args, "shell_tires", None)
        if isinstance(tires_raw, str):
            tires_raw = json.loads(tires_raw)
        if tires_raw:
            tire_cfgs = [t for t in tires_raw if t.get("active", True)]
        else:
            tire_cfgs = [{"name": "feda", "position": [0.0, 0.0, 0.0], "rpm": 0.0}]
        n_envs = len(tire_cfgs)
        self._n_envs = n_envs

        rim_cfgs = [cfg.get("rim", {}) for cfg in tire_cfgs]

        # ── Sand config ───────────────────────────────────────────────────
        _sh = getattr(args, "mpm_sand_height", None)
        with_sand = _sh is not None
        sand_h = float(_sh) if with_sand else _SAND_H
        sand_hx = float(getattr(args, "mpm_sand_half_x", _SAND_HX)) if with_sand else _SAND_HX
        sand_hz = float(getattr(args, "mpm_sand_half_z", _SAND_HZ)) if with_sand else _SAND_HZ
        voxel = float(getattr(args, "mpm_sand_voxel_size", _VOXEL)) if with_sand else _VOXEL
        self._with_sand = with_sand

        # ── ANCF mesh ─────────────────────────────────────────────────────
        n_circ = int(tire_cfgs[0].get("n-circ", int(getattr(args, "n_circ", 16))))
        thick = float(tire_cfgs[0].get("thickness", _THICK))
        E0 = tire_cfgs[0].get("E", None)
        if E0 is not None:
            mat0 = isotropic_ancf_material(
                E=float(E0),
                nu=float(tire_cfgs[0].get("nu", _NU_BALLOON)),
                rho=float(tire_cfgs[0].get("rho", _RHO_BALLOON)),
                alpha_damp=float(tire_cfgs[0].get("alpha-damp", 0.15)),
            )
            sec_mats = (mat0, mat0, mat0)
            sec_h = (thick, thick, thick)
        else:
            sec_mats = None
            sec_h = None

        build_kw = {
            "R_outer": _R_OUTER,
            "R_inner": _R_INNER,
            "width": _WIDTH,
            "n_circ": n_circ,
            "section_divs": _SEC_DIVS,
            "pressure": float(tire_cfgs[0].get("pressure", _BUILD_PRESSURE)),
            "device": dev,
        }
        if sec_mats is not None:
            build_kw["section_mats"] = sec_mats
            build_kw["section_h"] = sec_h
        self.ancf_model = build_ancf_tire_mesh(**build_kw)

        n_nodes = self.ancf_model.n_nodes
        x0_np = self.ancf_model.node_x0.numpy()

        # ── Bead + tread node identification ─────────────────────────────
        r_local = np.sqrt(x0_np[:, 1] ** 2 + x0_np[:, 2] ** 2)
        bead_mask = r_local < _R_INNER * 1.15
        tread_mask = r_local > 0.95 * _R_OUTER
        bead_np = np.where(bead_mask)[0].astype(np.int32)
        tread_np = np.where(tread_mask)[0].astype(np.int32)
        n_bead = len(bead_np)
        n_tread = len(tread_np)
        self._n_bead = n_bead
        self._n_tread = n_tread * n_envs
        self._n_nodes = n_nodes
        print(f"[INIT] beads={n_bead}  tread={n_tread}  nodes={n_nodes}  envs={n_envs}")

        bead_idx_all = np.concatenate([bead_np + e * n_nodes for e in range(n_envs)])
        tread_idx_all = np.concatenate([tread_np + e * n_nodes for e in range(n_envs)])
        self._bead_idx = wp.array(bead_idx_all, dtype=wp.int32, device=dev)
        self._tread_idx = wp.array(tread_idx_all, dtype=wp.int32, device=dev)

        bead_rest_np = x0_np[bead_np].astype(np.float32)
        self._bead_rest = wp.array(np.tile(bead_rest_np, (n_envs, 1)), dtype=wp.vec3, device=dev)
        bead_D0_np = self.ancf_model.node_D0.numpy()[bead_np].astype(np.float32)
        self._bead_D0 = wp.array(np.tile(bead_D0_np, (n_envs, 1)), dtype=wp.vec3, device=dev)

        # ── Hub state ─────────────────────────────────────────────────────
        hub_y_init = _R_OUTER + sand_h + _DROP_CLR
        self._hub_y_init = hub_y_init
        hub_mass_def = float(rim_cfgs[0].get("hub-mass", float(getattr(args, "hub_mass", 74.2))))
        I_rim_def = float(rim_cfgs[0].get("inertia", 8.0 * _R_INNER**2))
        corner_mass = float(getattr(args, "corner_mass", 0.0))
        self._hub_mass = hub_mass_def + corner_mass
        self._I_rim = I_rim_def
        self._g_load = self._hub_mass * _G

        self._hub_x = [float(cfg.get("position", [float(e)])[0]) for e, cfg in enumerate(tire_cfgs)]
        self._hub_y = [hub_y_init] * n_envs
        self._hub_vy = [0.0] * n_envs
        self._hub_z = [0.0] * n_envs
        self._rim_phi = [0.0] * n_envs
        self._target_rpm = [float(cfg.get("rpm", float(getattr(args, "rpm", 0.0)))) for cfg in tire_cfgs]
        self._current_rpm = list(self._target_rpm)
        self._rim_omega = [r * (2 * math.pi / 60) for r in self._current_rpm]
        self._tire_names = [cfg.get("name", f"tire{e}") for e, cfg in enumerate(tire_cfgs)]
        self._mode = str(getattr(args, "mode", "driven"))

        # GPU hub arrays — written each frame, read inside kernels
        self._hub_x_wp = wp.array(self._hub_x, dtype=float, device=dev)
        self._hub_y_wp = wp.array(self._hub_y, dtype=float, device=dev)
        self._hub_vy_wp = wp.array(self._hub_vy, dtype=float, device=dev)
        self._hub_z_wp = wp.array(self._hub_z, dtype=float, device=dev)
        self._rim_phi_wp = wp.array(self._rim_phi, dtype=float, device=dev)
        self._rim_omega_wp = wp.array(self._rim_omega, dtype=float, device=dev)
        self._wrench_buf = wp.zeros(n_envs, dtype=wp.spatial_vector, device=dev)

        # Tire self-weight tare (subtracted from bead reactions so hub falls at g in free fall)
        m_shell = (
            self.ancf_model.elem_mat.numpy()[: self.ancf_model.n_elems, 9].mean()
            * thick
            * 2
            * math.pi
            * _R_OUTER
            * _WIDTH
        )
        self._tare_fy = m_shell * _G

        # ── World node positions ──────────────────────────────────────────
        world_x = np.empty((n_envs * n_nodes, 3), dtype=np.float32)
        for e in range(n_envs):
            off = np.array([self._hub_x[e], hub_y_init, 0.0], dtype=np.float32)
            world_x[e * n_nodes : (e + 1) * n_nodes] = x0_np.astype(np.float32) + off

        tread_world = np.empty((n_envs * n_tread, 3), dtype=np.float32)
        for e in range(n_envs):
            off = np.array([self._hub_x[e], hub_y_init, 0.0], dtype=np.float32)
            tread_world[e * n_tread : (e + 1) * n_tread] = x0_np[tread_np] + off

        # ── Newton model: ground + ANCF particles + sand + tread spheres ──
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        if with_sand:
            SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_ground_plane(color=(0.65, 0.65, 0.65))

        builder.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x],
            vel=[(0.0, 0.0, 0.0)] * (n_envs * n_nodes),
            mass=[1.0] * (n_envs * n_nodes),
            radius=[0.001] * (n_envs * n_nodes),
        )
        en_np = self.ancf_model.elem_nodes.numpy()
        tris_one = np.empty((len(en_np) * 2, 3), dtype=np.int32)
        tris_one[0::2] = en_np[:, [0, 1, 2]]
        tris_one[1::2] = en_np[:, [0, 2, 3]]
        tris_all = np.concatenate([tris_one + e * n_nodes for e in range(n_envs)], axis=0)
        builder.add_triangles(i=tris_all[:, 0].tolist(), j=tris_all[:, 1].tolist(), k=tris_all[:, 2].tolist())

        if with_sand:
            sand_lo = np.array([-sand_hx * n_envs, 0.0, -sand_hz])
            sand_hi = np.array([sand_hx * n_envs, sand_h, sand_hz])
            ppc = 1.5
            res = np.array(np.ceil(ppc * (sand_hi - sand_lo) / voxel), dtype=int).clip(1)
            cell = (sand_hi - sand_lo) / res
            r_p = float(np.max(cell) * 0.5)
            m_p = float(np.prod(cell) * _SAND_RHO)
            builder.add_particle_grid(
                pos=wp.vec3(*sand_lo.tolist()),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=int(res[0]) + 1,
                dim_y=int(res[1]) + 1,
                dim_z=int(res[2]) + 1,
                cell_x=float(cell[0]),
                cell_y=float(cell[1]),
                cell_z=float(cell[2]),
                mass=m_p,
                jitter=2.0 * r_p,
                radius_mean=r_p,
                custom_attributes={"mpm:friction": _SAND_FRIC},
            )
            r_sph = voxel
            for i in range(self._n_tread):
                b = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(tread_world[i, 0]), float(tread_world[i, 1]), float(tread_world[i, 2])),
                        q=wp.quat_identity(),
                    ),
                    mass=0.001,
                    label=f"tr{i}",
                )
                s = builder.add_shape_sphere(b, radius=r_sph)
                builder.shape_flags[s] = newton.ShapeFlags.COLLIDE_PARTICLES

        self.model = builder.finalize(device=dev)
        self.model.set_gravity(wp.vec3(0.0, -_G, 0.0))
        if with_sand:
            self.model.mpm.friction.fill_(_SAND_FRIC)
            self.model.mpm.yield_pressure.fill_(1.0e12)

        # ── ANCF solver (flat contact at sand surface level) ──────────────
        ancf_kn = float(getattr(args, "kn", 20000.0))
        ancf_kd = float(getattr(args, "kd", 20.0))
        self.solver = SolverANCFShell(
            model=self.model,
            ancf_model=self.ancf_model,
            ground_z=sand_h,  # flat contact at sand surface
            kn=ancf_kn,
            kd=ancf_kd,
            mu=0.9,
            nr_max_iter=int(getattr(args, "nr_iters", 2)),
            pcg_max_iter=int(getattr(args, "pcg_iters", 25)),
            n_envs=n_envs,
            rim_radius=_R_INNER,
            rim_kn=float(rim_cfgs[0].get("kn", 50000.0)),
            rim_kd=float(rim_cfgs[0].get("kd", 20.0)),
        )
        # Register bead Dirichlet so graph_step zeros NR updates at bead nodes
        self.solver._dirichlet_idx = wp.array(bead_idx_all, dtype=wp.int32, device=dev)
        self.solver.node_x.assign(world_x)

        # Cavity pressure
        pressures = [float(cfg.get("pressure", _BUILD_PRESSURE)) for cfg in tire_cfgs]
        builds = list(pressures)
        self.solver.set_cavity(pressures, builds)
        self._current_pressure = list(pressures)
        self._build_pressure = list(builds)
        self._target_pressure = list(pressures)
        max_gauges = [float(cfg.get("max-gauge-pressure", _MAX_GAUGE)) for cfg in tire_cfgs]
        gauge_rates = [float(cfg.get("gauge-rate", _GAUGE_RATE)) for cfg in tire_cfgs]
        self._p_min = [max(0.0, builds[i] - max_gauges[i]) for i in range(n_envs)]
        self._p_max = [builds[i] + max_gauges[i] for i in range(n_envs)]
        self._gauge_rate = gauge_rates

        self.solver.capture_graph(self.sim_dt)
        # Restore after warmup
        self.solver.node_x.assign(world_x)
        self.solver.node_xd.zero_()
        self.solver.node_xdd.zero_()
        self.solver.global_f_int.zero_()
        self.solver.global_f_int0.zero_()
        self.solver.set_cavity(pressures, builds)

        # ── MPM solver ────────────────────────────────────────────────────
        self.mpm_solver = None
        self._tread_imp = wp.zeros(self._n_tread, dtype=wp.vec3, device=dev)
        self._imp = wp.zeros(1, dtype=wp.vec3, device=dev)
        self._imp_ids = wp.full(1, -1, dtype=int, device=dev)
        self._f_node_buf = wp.zeros(self._n_tread, dtype=wp.vec3, device=dev)

        if with_sand:
            mpm_cfg = SolverImplicitMPM.Config()
            mpm_cfg.voxel_size = voxel
            mpm_cfg.grid_type = "fixed"
            mpm_cfg.grid_padding = 8
            mpm_cfg.max_active_cell_count = 1 << 16
            mpm_cfg.strain_basis = "P0"
            mpm_cfg.max_iterations = 25
            mpm_cfg.critical_fraction = 0.0
            mpm_cfg.air_drag = 1.0
            # "pic" is ~3× more dissipative than "apic" (same as mpm_anymal) →
            # reduces contact impulse so the tire doesn't rocket upward.
            mpm_cfg.transfer_scheme = "pic"
            # "forward" uses current sphere velocity — less aggressive than
            # "backward" (finite-difference from previous frame).
            mpm_cfg.collider_velocity_mode = "forward"
            self.mpm_solver = SolverImplicitMPM(self.model, mpm_cfg)
            self.state_0 = self.model.state()
            wp.copy(self.state_0.particle_q, self.solver.node_x)
            self.mpm_solver.setup_collider(
                body_mass=wp.zeros_like(self.model.body_mass),
                body_q=self.state_0.body_q,
            )
            _MAX = 1 << 20
            self._imp = wp.zeros(_MAX, dtype=wp.vec3, device=dev)
            self._imp_ids = wp.full(_MAX, -1, dtype=int, device=dev)
            self._collect_impulses()
        else:
            self.state_0 = self.model.state()
            wp.copy(self.state_0.particle_q, self.solver.node_x)

        # MPM stride: run MPM every N frames, reuse last impulse in between.
        self._mpm_stride = int(getattr(args, "mpm_stride", 2))
        self._mpm_frame = self._mpm_stride - 1  # trigger on first frame

        # ── Visualization buffers ─────────────────────────────────────────
        _N_r = n_circ
        _n_rings = 2 * _SEC_DIVS[0]
        seg_s = np.array([i + k * _N_r for k in range(_n_rings) for i in range(_N_r)], dtype=np.int32)
        seg_e = np.array([(i + 1) % _N_r + k * _N_r for k in range(_n_rings) for i in range(_N_r)], dtype=np.int32)
        seg_s_all = np.concatenate([seg_s + e * n_bead for e in range(n_envs)])
        seg_e_all = np.concatenate([seg_e + e * n_bead for e in range(n_envs)])
        self._ring_seg_s = wp.array(seg_s_all, dtype=wp.int32, device=dev)
        self._ring_seg_e = wp.array(seg_e_all, dtype=wp.int32, device=dev)
        self._n_ring_segs = len(seg_s_all)
        self._bead_pos_yup = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=dev)
        self._ring_line_s = wp.zeros(self._n_ring_segs, dtype=wp.vec3, device=dev)
        self._ring_line_e = wp.zeros(self._n_ring_segs, dtype=wp.vec3, device=dev)
        self._contact_line_s = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=dev)
        self._contact_line_e = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=dev)
        self._mpm_spike_s = wp.zeros(self._n_tread, dtype=wp.vec3, device=dev)
        self._mpm_spike_e = wp.zeros(self._n_tread, dtype=wp.vec3, device=dev)
        self._contact_vis_scale = 100.0

        if viewer is not None:
            viewer.set_model(self.model)
            viewer.show_particles = with_sand
            viewer.set_camera(pos=wp.vec3(0.0, 1.0, 2.5), pitch=-15.0, yaw=-90.0)

        print(
            f"[INIT] {n_envs} tire(s) | sand={'ON' if with_sand else 'OFF'} | "
            f"hub_mass={self._hub_mass:.0f} kg | g_load={self._g_load:.0f} N"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _collect_impulses(self):
        if self.mpm_solver is None:
            return
        imp, _pos, ids = self.mpm_solver.collect_collider_impulses(self.state_0)
        self._imp_ids.fill_(-1)
        n = min(imp.shape[0], self._imp.shape[0])
        if n > 0:
            self._imp[:n].assign(imp[:n])
            self._imp_ids[:n].assign(ids[:n])

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self):
        dev = "cuda:0"
        dt = self.sim_dt
        N = self._n_envs
        nb = self._n_bead
        nn = self._n_nodes

        prescribe_inputs = [
            self.solver.node_x,
            self.solver.node_xd,
            self.solver.node_xdd,
            self.solver.node_D,
            self.solver.node_Dd,
            self.solver.node_Ddd,
            self._bead_idx,
            self._bead_rest,
            self._bead_D0,
            self._hub_x_wp,
            self._hub_y_wp,
            self._hub_vy_wp,
            self._hub_z_wp,
            self._rim_phi_wp,
            self._rim_omega_wp,
            _R_OUTER,
            nb,
            nn,
        ]

        # MPM impulse → hub-level upward force (NOT applied to soft shell nodes).
        #
        # Applying to tread (E=1MPa shell) causes 12,700 m/s² per node → tears shell.
        # Instead: treat sand reaction as an upward force on the HUB.
        # The ANCF flat contact at ground_z=sand_h already stops tread penetration;
        # the MPM just provides the upward push to the hub that limits settling depth.
        #
        # Effective g_load per substep = hub_mass*g - J_mpm_y/frame_dt
        self.solver.node_f_ext_persistent.zero_()
        _mpm_hub_fy = 0.0  # extra upward force on hub from sand [N]
        if self._with_sand:
            self._tread_imp.zero_()
            wp.launch(
                _scatter_mpm_impulses,
                dim=self._imp_ids.shape[0],
                inputs=[self._imp_ids, self._imp, self._n_tread, self._tread_imp],
                device=dev,
            )
            imp_np = self._tread_imp.numpy()
            # Total upward Y-force from sand on hub (J_y / frame_dt)
            _mpm_hub_fy = float(imp_np[:, 1].sum()) / self.frame_dt  # N, upward

        for _ in range(self.sim_substeps):
            # 1. Pre-step Dirichlet prescription
            wp.launch(_prescribe_bead_batched, dim=N * nb, inputs=prescribe_inputs, device=dev)
            # 2. Advance kinematic spin + forward
            wp.launch(
                _advance_phi_and_z_batched,
                dim=N,
                inputs=[self._rim_phi_wp, self._rim_omega_wp, self._hub_z_wp, dt, _R_OUTER],
                device=dev,
            )
            # 3. ANCF FEM step
            self.solver.graph_step()
            # 4. Post-step corrector
            wp.launch(_prescribe_bead_batched, dim=N * nb, inputs=prescribe_inputs, device=dev)
            # 5. Bead wrench → hub
            wp.launch(
                _accum_bead_wrench_kinematic,
                dim=N * nb,
                inputs=[
                    self.solver.global_f_int,
                    self.solver.node_x,
                    self._bead_idx,
                    self._hub_x_wp,
                    self._hub_y_wp,
                    self._hub_z_wp,
                    self._wrench_buf,
                    nb,
                    nn,
                ],
                device=dev,
            )
            # 6. Integrate hub vertical + spin.
            # g_load_eff = hub weight - MPM upward force (sand supports hub).
            # _mpm_hub_fy is positive upward, spread over all substeps.
            _g_load_eff = self._g_load - _mpm_hub_fy
            _update_om = 0 if self._mode == "driven" else 1
            wp.launch(
                _integrate_hub_dof,
                dim=N,
                inputs=[
                    self._hub_y_wp,
                    self._hub_vy_wp,
                    self._rim_omega_wp,
                    self._wrench_buf,
                    self._hub_mass,
                    self._I_rim,
                    _g_load_eff,
                    self._tare_fy,
                    dt,
                    1,
                    _update_om,
                ],
                device=dev,
            )

        # ── MPM step (stride rate) ────────────────────────────────────────
        # Impulse from the last MPM step is reused for the skipped frames;
        # sand forces change slowly enough at 60 Hz that this is imperceptible.
        if self._with_sand:
            self._mpm_frame = (self._mpm_frame + 1) % self._mpm_stride
            if self._mpm_frame == 0:
                _t0_mpm = time.perf_counter()
                wp.launch(
                    _copy_tread_q,
                    dim=self._n_tread,
                    inputs=[self.state_0.body_q, self.solver.node_x, self._tread_idx],
                    device=dev,
                )
                wp.launch(
                    _copy_tread_qd,
                    dim=self._n_tread,
                    inputs=[self.state_0.body_qd, self.solver.node_xd, self._tread_idx],
                    device=dev,
                )
                self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)
                self._collect_impulses()
                wp.synchronize_device(dev)
                self._last_mpm_ms = 1000.0 * (time.perf_counter() - _t0_mpm)
                self._mpm_run_count += 1
            else:
                self._last_mpm_ms = 0.0

    _dbg_frame = 0
    _diag_period = 10  # print every N frames
    _prev_hub_y = None
    _fps_t0 = None  # wall time at frame _fps_n0 (set after JIT on frame 2)
    _fps_n0 = 0
    _last_mpm_ms = 0.0
    _mpm_run_count = 0
    _last_step_ms = 0.0

    def step(self):
        _t0_step = time.perf_counter()
        # Ramp pressure
        for i in range(self._n_envs):
            d = self._target_pressure[i] - self._current_pressure[i]
            r = self._gauge_rate[i]
            self._current_pressure[i] += r if abs(d) > r else d
        self.solver.set_cavity(self._current_pressure, self._build_pressure)

        # Ramp RPM
        if self._mode == "driven":
            for i in range(self._n_envs):
                d = self._target_rpm[i] - self._current_rpm[i]
                if abs(d) <= _RPM_RATE:
                    self._current_rpm[i] = self._target_rpm[i]
                else:
                    self._current_rpm[i] += _RPM_RATE if d > 0.0 else -_RPM_RATE
                self._rim_omega[i] = self._current_rpm[i] * (2.0 * math.pi / 60.0)
            self._rim_omega_wp.assign(wp.array(self._rim_omega, dtype=float))

        self.simulate()

        # Mirror hub state back to Python arrays
        self._hub_y = list(self._hub_y_wp.numpy())
        self._hub_vy = list(self._hub_vy_wp.numpy())

        # Mirror ANCF nodes → render particles
        wp.copy(self.state_0.particle_q, self.solver.node_x)
        self.sim_time += self.frame_dt
        self._dbg_frame += 1
        _now = time.perf_counter()
        self._last_step_ms = 1000.0 * (_now - _t0_step)
        if self._dbg_frame == 2:  # after JIT: start the clock
            self._fps_t0 = _now
            self._fps_n0 = 2

        if self._dbg_frame % self._diag_period == 0 and self._with_sand:
            imp_np = self._tread_imp.numpy()
            mag = np.linalg.norm(imp_np, axis=1)
            J_sum = float(mag.sum())
            float(mag.max())
            int((mag > 1e-6).sum())
            hub_y = self._hub_y[0]
            self._hub_vy[0]
            # Estimate required corner_mass for half-height bounce
            v_i = math.sqrt(2 * 9.81 * 0.10)  # impact velocity (10cm drop)
            v_r = math.sqrt(2 * 9.81 * 0.05)  # half-height rebound
            M_half = J_sum / (v_i + v_r) if (v_i + v_r) > 0 else 0
            max(0.0, M_half - self._hub_mass)
            dt_block = _now - self._fps_t0 if self._fps_t0 else 1.0
            fps_block = (self._dbg_frame - self._fps_n0) / dt_block if dt_block > 0 else 0.0
            self._fps_t0 = _now  # reset window each block
            self._fps_n0 = self._dbg_frame
            frame_ms = 1000.0 / fps_block if fps_block > 0 else 0.0
            mpm_tag = f"mpm={self._last_mpm_ms:.0f}ms" if self._last_mpm_ms > 0 else "mpm=skip"
            render_ms = frame_ms - self._last_step_ms
            print(
                f"[{self._dbg_frame:4d}]  {fps_block:.1f} fps  "
                f"frame={frame_ms:.0f}ms  step={self._last_step_ms:.0f}ms  "
                f"render≈{render_ms:.0f}ms  "
                f"{mpm_tag}  mpm_runs={self._mpm_run_count}/{self._dbg_frame}  "
                f"hub_y={hub_y:.3f}m  J_sum={J_sum:.2f}N·s"
            )

    def render(self):
        if self.viewer is None:
            return
        dev = "cuda:0"
        N, nb, nn = self._n_envs, self._n_bead, self._n_nodes

        # Bead ring (orange)
        wp.launch(
            _gather_bead_positions,
            dim=N * nb,
            inputs=[self.solver.node_x, self._bead_idx, self._bead_pos_yup, nb, nn],
            device=dev,
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_yup, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device=dev,
        )

        # Contact spikes (cyan — ground contact penetration)
        wp.launch(
            _gather_contact_spikes_yup,
            dim=N * nn,
            inputs=[self.solver.node_x, 0.0, self._contact_vis_scale, self._contact_line_s, self._contact_line_e],
            device=dev,
        )

        # MPM impulse spikes (yellow — sand reaction)
        wp.launch(
            _build_mpm_spikes,
            dim=self._n_tread,
            inputs=[self.solver.node_x, self._tread_idx, self._tread_imp, 2.0, self._mpm_spike_s, self._mpm_spike_e],
            device=dev,
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("bead_ring", self._ring_line_s, self._ring_line_e, (1.0, 0.45, 0.0))
        self.viewer.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, (0.0, 1.0, 1.0))
        self.viewer.log_lines("mpm_spikes", self._mpm_spike_s, self._mpm_spike_e, (1.0, 1.0, 0.0))
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Pressure [Pa]")
        for i, name in enumerate(self._tire_names):
            changed, v = ui.slider_float(f"{name}##{i}", self._target_pressure[i], self._p_min[i], self._p_max[i])
            if changed:
                self._target_pressure[i] = float(v)
            ui.text(f"   {self._current_pressure[i]:8.0f} Pa")

        ui.separator()
        ui.text("RPM (driven mode)")
        for i, name in enumerate(self._tire_names):
            changed, v = ui.slider_float(f"RPM {name}##{i}", self._target_rpm[i], -300.0, 300.0)
            if changed:
                self._target_rpm[i] = float(v)

    def test_final(self):
        xn = self.solver.node_x.numpy()
        assert not np.any(np.isnan(xn)), "NaN in node positions"
        assert xn[:, 1].min() > -0.2, "tire fell through terrain"

    @staticmethod
    def create_parser():
        p = newton.examples.create_parser()
        p.add_argument("--substeps", type=int, default=10)
        p.add_argument("--nr-iters", type=int, default=2, dest="nr_iters")
        p.add_argument("--pcg-iters", type=int, default=25, dest="pcg_iters")
        p.add_argument("--kn", type=float, default=20000.0)
        p.add_argument("--kd", type=float, default=20.0)
        p.add_argument("--n-circ", type=int, default=16, dest="n_circ")
        p.add_argument("--rpm", type=float, default=0.0)
        p.add_argument("--mode", type=str, default="driven", choices=["driven", "free"])
        p.add_argument(
            "--corner-mass",
            type=float,
            default=0.0,
            dest="corner_mass",
            help="Vehicle corner mass [kg] added to hub_mass for g_load.",
        )
        p.add_argument("--shell-tires", type=str, default=None, dest="shell_tires")
        p.add_argument("--mpm-sand-height", type=float, default=None, dest="mpm_sand_height")
        p.add_argument("--mpm-sand-half-x", type=float, default=None, dest="mpm_sand_half_x")
        p.add_argument("--mpm-sand-half-z", type=float, default=None, dest="mpm_sand_half_z")
        p.add_argument("--mpm-sand-voxel-size", type=float, default=None, dest="mpm_sand_voxel_size")
        p.add_argument(
            "--mpm-stride",
            type=int,
            default=2,
            dest="mpm_stride",
            help="Run MPM every N render frames (1=every frame, 2=every other, …).",
        )
        return p


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
