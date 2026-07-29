# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Example ANCF MPM
#
# ANCF3423 shell tire(s) dropped onto implicit MPM granular sand.
# Architecture mirrors example_mpm_anymal: MPM particles share the same
# Newton model as the ANCF render particles, and the ANCF tread nodes act
# as kinematic sphere colliders for the sand.
#
# Coupling (staggered, per frame):
#   1. ANCF substeps  — tire deforms under gravity + tread impulse kick
#   2. Tread sync     — write ANCF tread positions → MPM sphere body_q
#   3. MPM step       — sand deforms under tread load (frame rate)
#   4. Collect        — MPM grid impulses → applied to ANCF tread next frame
#
# Command: python -m newton.examples ancf_mpm
###########################################################################

from __future__ import annotations

import json

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.ancf_shell import (
    SolverANCFShell,
    build_ancf_tire_mesh,
    isotropic_ancf_material,
)
from newton.solvers import SolverImplicitMPM

# ── Defaults ───────────────────────────────────────────────────────────────────
_R_OUTER = 0.329
_R_INNER = 0.130
_WIDTH = 0.230
_N_CIRC = 16
_E = 10.0e6
_NU = 0.45
_RHO = 1100.0
_H_SHELL = 0.012
_PRESSURE = 110_000.0
_SUBSTEPS = 5
_KN = 0.0  # flat ground disabled; MPM provides contact
_VOXEL = 0.04  # MPM voxel size [m]
_SAND_H = 0.15  # sand pile height above Y=0 [m]
_SAND_HX = 0.60  # sand half-extent X [m]
_SAND_HZ = 0.60  # sand half-extent Z [m]
_SAND_RHO = 1500.0
_SAND_FRIC = 0.68
_DROP_CLR = 0.10  # clearance above sand top [m]


# ── ANCF ↔ MPM coupling kernels ────────────────────────────────────────────────


@wp.kernel
def _copy_tread_to_body_q(
    body_q: wp.array[wp.transform],
    node_x: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
):
    i = wp.tid()
    body_q[i] = wp.transform(node_x[tread_idx[i]], wp.quat_identity())


@wp.kernel
def _copy_tread_to_body_qd(
    body_qd: wp.array[wp.spatial_vector],
    node_xd: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
):
    i = wp.tid()
    v = node_xd[tread_idx[i]]
    body_qd[i] = wp.spatial_vector(v, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def _scatter_impulses(
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
def _build_contact_spikes(
    node_x: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
    tread_impulse: wp.array[wp.vec3],
    vis_scale: float,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    """Cyan spike per tread node: base at node pos, tip = base + impulse_mag * vis_scale in +Y."""
    i = wp.tid()
    p = node_x[tread_idx[i]]
    mag = wp.length(tread_impulse[i])
    starts[i] = p
    ends[i] = wp.vec3(p[0], p[1] + mag * vis_scale, p[2])


@wp.kernel
def _add_mpm_force_to_f_ext(
    f_ext: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
    impulse: wp.array[wp.vec3],
    inv_dt: float,
):
    """Convert per-node MPM impulse to force and add to ANCF f_ext_persistent.

    F = impulse / frame_dt.  Using f_ext (not a velocity kick) lets the
    implicit HHT solver balance the MPM force against the shell stiffness —
    critical for light tires where imp/m would give thousands of m/s.
    """
    i = wp.tid()
    wp.atomic_add(f_ext, tread_idx[i], impulse[i] * inv_dt)


@wp.kernel
def _apply_tread_velocity_kick(
    node_xd: wp.array[wp.vec3],
    tread_idx: wp.array[wp.int32],
    impulse: wp.array[wp.vec3],
    mass: wp.array[float],
    inv_sub: float,
):
    """Δv = J/(m × substeps). J is physically bounded because setup_collider
    received finite body_mass — MPM can't give more momentum than m×v_impact."""
    i = wp.tid()
    m = mass[i]
    if m > 1.0e-9:
        node_xd[tread_idx[i]] = node_xd[tread_idx[i]] + impulse[i] * (inv_sub / m)


# ── Example ────────────────────────────────────────────────────────────────────


class Example:
    """ANCF FEM tire(s) dropped onto MPM granular sand."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        substeps = int(getattr(args, "substeps", _SUBSTEPS))

        self.sim_substeps = substeps
        self.sim_dt = self.frame_dt / substeps

        dev = "cuda:0"

        # ── Per-tire config (same schema as ancf_shell_drop) ─────────────
        tires_raw = getattr(args, "shell_tires", None)
        if isinstance(tires_raw, str):
            tires_raw = json.loads(tires_raw)
        if tires_raw:
            tire_cfgs = [t for t in tires_raw if t.get("active", True)]
        else:
            tire_cfgs = [{"name": "polaris_ref", "position": [0.0, 0.0, 0.0]}]
        n_envs = len(tire_cfgs)

        # ── ANCF tire mesh (shared geometry, per-tire material) ───────────
        # Build with Polaris orthotropic defaults (no section_mats) — same as
        # ancf_shell_drop.  The Polaris reference geometry is calibrated for
        # these materials.  Per-tire material is overridden below via elem_mat.
        n_circ = int(tire_cfgs[0].get("n-circ", getattr(args, "n_circ", _N_CIRC)))
        self.ancf_model = build_ancf_tire_mesh(
            R_outer=_R_OUTER,
            R_inner=_R_INNER,
            width=_WIDTH,
            n_circ=n_circ,
            section_divs=(1, 2, 3),
            pressure=float(tire_cfgs[0].get("pressure", _PRESSURE)),
            device=dev,
        )
        n_nodes = self.ancf_model.n_nodes
        x0_np = self.ancf_model.node_x0.numpy()
        base_mat = self.ancf_model.elem_mat.numpy()

        # Per-tire material override — same logic as ancf_shell_drop.
        # E=null → keep Polaris orthotropic base.  E=float → uniform isotropic.
        per_env_mats = []
        ne = self.ancf_model.n_elems
        for cfg in tire_cfgs:
            E = cfg.get("E", None)
            if E is not None:
                mat = isotropic_ancf_material(
                    E=float(E),
                    nu=float(cfg.get("nu", 0.3)),
                    rho=float(cfg.get("rho", 1100.0)),
                    alpha_damp=float(cfg.get("alpha-damp", 0.0)),
                )
                row = np.array(
                    [
                        mat.C11,
                        mat.C22,
                        mat.C33,
                        mat.C12,
                        mat.C13,
                        mat.C23,
                        mat.G23,
                        mat.G13,
                        mat.G12,
                        mat.rho,
                        mat.alpha_damp,
                    ],
                    dtype=np.float32,
                )
                per_env_mats.append(np.tile(row, (ne, 1)))
            else:
                per_env_mats.append(base_mat.copy())
        # Always assign — the n_envs>1 guard was silently dropping all per-tire
        # material params (alpha-damp, E, rho) when running a single tire.
        self.ancf_model.elem_mat = wp.array(np.concatenate(per_env_mats, axis=0), dtype=float, device=dev)

        # ── Sand config (parsed early — needed for hub_y0 drop height) ─────
        # The docker CLI resolver expands "mpm-sand": {...} into individual
        # --mpm-sand-height / --mpm-sand-half-x / ... args.  Sand is enabled
        # when at least one of them is present (non-None default = null in JSON).
        _sh = getattr(args, "mpm_sand_height", None)
        with_sand = _sh is not None
        sand_h = float(_sh) if with_sand else _SAND_H
        sand_hx = float(getattr(args, "mpm_sand_half_x", _SAND_HX)) if with_sand else _SAND_HX
        sand_hz = float(getattr(args, "mpm_sand_half_z", _SAND_HZ)) if with_sand else _SAND_HZ
        voxel = float(getattr(args, "mpm_sand_voxel_size", _VOXEL)) if with_sand else _VOXEL

        def _sf(attr, default):
            v = getattr(args, attr, None)
            return float(v) if v is not None else float(default)

        sand_density = _sf("mpm_sand_density", _SAND_RHO) if with_sand else _SAND_RHO
        sand_friction = _sf("mpm_sand_friction", _SAND_FRIC) if with_sand else _SAND_FRIC
        sand_yield_p = _sf("mpm_sand_yield_pressure", 1.0e12) if with_sand else 1.0e12
        sand_viscosity = _sf("mpm_sand_viscosity", 0.0) if with_sand else 0.0
        sand_air_drag = _sf("mpm_sand_air_drag", 1.0) if with_sand else 1.0
        self._with_sand = with_sand

        # ── Initial positions: each tire at its config position + drop height
        hub_y0 = sand_h + _R_OUTER + _DROP_CLR
        self._drop_center_y = hub_y0  # initial tire center Y [m] — never bounce above this
        world_x = np.empty((n_envs * n_nodes, 3), dtype=np.float32)
        for e, cfg in enumerate(tire_cfgs):
            pos = cfg.get("position", [float(e), 0.0, 0.0])
            off = np.array([float(pos[0]), float(pos[1]) + hub_y0, float(pos[2])], dtype=np.float32)
            world_x[e * n_nodes : (e + 1) * n_nodes] = x0_np + off
        x_spacing = 1.0  # fallback spacing (only used for tread world init)

        # ── Tread node indices (crown: r > 0.95 × R_outer) ────────────────
        r_local = np.sqrt(x0_np[:, 1] ** 2 + x0_np[:, 2] ** 2)
        tread_np = np.where(r_local > 0.95 * _R_OUTER)[0].astype(np.int32)
        n_tread = len(tread_np)
        print(f"[MPM] {n_tread} tread nodes per tire → MPM sphere colliders")

        # Tread world positions at drop height (for MPM body initialisation)
        tread_world = np.empty((n_envs * n_tread, 3), dtype=np.float32)
        for e, cfg in enumerate(tire_cfgs):
            pos = cfg.get("position", [float(e), 0.0, 0.0])
            off = np.array([float(pos[0]), float(pos[1]) + hub_y0, float(pos[2])], dtype=np.float32)
            tread_world[e * n_tread : (e + 1) * n_tread] = x0_np[tread_np] + off
        tread_idx_all = np.concatenate([tread_np + e * n_nodes for e in range(n_envs)]).astype(np.int32)
        n_tread_all = len(tread_idx_all)

        self._tread_idx = wp.array(tread_idx_all, dtype=wp.int32, device=dev)
        self._n_tread = n_tread_all
        self._n_nodes = n_nodes
        self._n_envs = n_envs

        # ── Newton model: ANCF render particles + MPM sand (same builder) ─
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        if with_sand:
            SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_ground_plane(color=(0.65, 0.65, 0.65))

        # ANCF render particles (topology only — dynamics from ANCF solver)
        builder.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x],
            vel=[(0.0, 0.0, 0.0)] * (n_envs * n_nodes),
            mass=[1.0] * (n_envs * n_nodes),
            radius=[0.001] * (n_envs * n_nodes),
        )
        en = self.ancf_model.elem_nodes.numpy()
        tris_one = np.empty((len(en) * 2, 3), dtype=np.int32)
        tris_one[0::2] = en[:, [0, 1, 2]]
        tris_one[1::2] = en[:, [0, 2, 3]]
        tris_all = np.concatenate([tris_one + e * n_nodes for e in range(n_envs)], axis=0)
        builder.add_triangles(
            i=tris_all[:, 0].tolist(),
            j=tris_all[:, 1].tolist(),
            k=tris_all[:, 2].tolist(),
        )

        if with_sand:
            # Sand particles: Y = 0 → sand_h (pile ON the Y=0 ground plane)
            sand_lo = np.array([-sand_hx * n_envs, 0.0, -sand_hz])
            sand_hi = np.array([sand_hx * n_envs, sand_h, sand_hz])
            ppc = 2.5
            res = np.array(np.ceil(ppc * (sand_hi - sand_lo) / voxel), dtype=int).clip(1)
            cell = (sand_hi - sand_lo) / res
            r_p = float(np.max(cell) * 0.5)
            m_p = float(np.prod(cell) * sand_density)
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
                custom_attributes={"mpm:friction": sand_friction},
            )
            # Tread sphere bodies (kinematic MPM colliders)
            r_sph = voxel
            for i in range(n_tread_all):
                b = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(tread_world[i, 0]), float(tread_world[i, 1]), float(tread_world[i, 2])),
                        q=wp.quat_identity(),
                    ),
                    mass=0.001,
                    label=f"tread_{i}",
                )
                s = builder.add_shape_sphere(b, radius=r_sph)
                builder.shape_flags[s] = newton.ShapeFlags.COLLIDE_PARTICLES

        self.model = builder.finalize(device=dev)
        self.model.set_gravity(wp.vec3(0.0, -9.81, 0.0))
        if with_sand:
            self.model.mpm.friction.fill_(sand_friction)
            self.model.mpm.yield_pressure.fill_(sand_yield_p)
            self.model.mpm.viscosity.fill_(sand_viscosity)

        # ── ANCF solver ───────────────────────────────────────────────────
        # ground_z = sand_h: ANCF penalty contact at the sand *surface*.
        # This gives immediate same-frame reaction stopping the tire at the sand top,
        # exactly as in mpm_anymal where rigid bodies collide with the ground plane
        # while MPM sand handles granular deformation around the contact zone.
        # Without this, MPM impulses arrive one frame late → tire penetrates deeply
        # → huge impulse → explosion.
        ancf_kn = float(getattr(args, "kn", 2e3))
        ancf_kd = float(getattr(args, "kd", 2.0))
        self.ancf_solver = SolverANCFShell(
            model=self.model,
            ancf_model=self.ancf_model,
            ground_z=sand_h,  # sand surface level — stops tire immediately
            kn=ancf_kn,
            kd=ancf_kd,
            mu=0.9,
            nr_max_iter=int(getattr(args, "nr_iters", 3)),
            pcg_max_iter=int(getattr(args, "pcg_iters", 50)),
            n_envs=n_envs,
        )
        self.ancf_solver.node_x.assign(world_x)

        pressures = [float(cfg.get("pressure", _PRESSURE)) for cfg in tire_cfgs]
        self.ancf_solver.set_cavity(pressures, pressures)

        # Step 1: capture internal ANCF graph (graph_step() path).
        self.ancf_solver.capture_graph(self.sim_dt)

        def _restore():
            self.ancf_solver.node_x.assign(world_x)
            self.ancf_solver.node_xd.zero_()
            self.ancf_solver.node_xdd.zero_()
            self.ancf_solver.global_f_int.zero_()
            self.ancf_solver.global_f_int0.zero_()
            self.ancf_solver.node_f_ext_persistent.zero_()
            self.ancf_model.elem_eas_alpha.zero_()
            self.ancf_solver.set_cavity(pressures, pressures)

        _restore()

        # Step 2: pre-warm step() so kernels compile BEFORE capture_begin.
        # (same pattern as ancf_shell_drop — capture_graph() only warms _step_batched,
        #  not the unrolled step() path used by the frame graph)
        self.ancf_solver.step(None, None, None, None, self.sim_dt)
        wp.synchronize_device(dev)
        _restore()

        # Step 3: capture frame graph (all substeps in one launch).
        self._substep_graph = None
        try:
            wp.capture_begin(device=dev)
            for _ in range(substeps):
                self.ancf_solver.step(None, None, None, None, self.sim_dt)
            self._substep_graph = wp.capture_end(device=dev)
            print(f"[ANCF] Frame graph: {substeps} substeps captured ✓")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            print(f"[ANCF] Frame graph failed ({e!r}) — loop fallback")

        self.state_0 = self.model.state()
        wp.copy(self.state_0.particle_q, self.ancf_solver.node_x)

        # ── MPM solver (only when sand is enabled) ─────────────────────────
        self.mpm_solver = None
        self._tread_imp = wp.zeros(n_tread_all, dtype=wp.vec3, device=dev)
        # Per-tread-node lumped mass — used for velocity kick Δv = J/(m×sub)
        m_lump = self.ancf_solver.lumped_mass.numpy()
        tm_np = np.array([m_lump[int(idx) * 6] for idx in tread_idx_all], dtype=np.float32).clip(1e-9)
        self._tread_mass = wp.array(tm_np, dtype=float, device=dev)
        self._imp = wp.zeros(1, dtype=wp.vec3, device=dev)
        self._imp_ids = wp.full(1, -1, dtype=int, device=dev)
        self._imp_pos = wp.zeros(1, dtype=wp.vec3, device=dev)

        if with_sand:
            mpm_cfg = SolverImplicitMPM.Config()
            mpm_cfg.voxel_size = voxel
            mpm_cfg.grid_type = "fixed"
            mpm_cfg.grid_padding = 50
            mpm_cfg.max_active_cell_count = 1 << 16
            mpm_cfg.strain_basis = "P0"
            mpm_cfg.max_iterations = 50
            mpm_cfg.critical_fraction = 0.0
            mpm_cfg.air_drag = sand_air_drag
            # "pic" is ~3× more dissipative than default "apic" — same as mpm_anymal.
            # "apic" preserves angular momentum (less damping), generating ~3× larger
            # impulses that overwhelm light ANCF nodes and cause explosion.
            mpm_cfg.transfer_scheme = "pic"
            # "forward" uses current sphere velocity — smoother than "backward"
            # (which uses finite-difference from previous position and generates
            # larger corrective impulses on first contact).
            mpm_cfg.collider_velocity_mode = "forward"
            self.mpm_solver = SolverImplicitMPM(self.model, mpm_cfg)
            # Kinematic (body_mass=zeros): tread spheres are immovable boundaries.
            # The bounce height is controlled by kd (contact damping), not by MPM
            # coupling. True two-way MPM feedback requires a separate tread model
            # (see mpm_ancf_tire.py) — applying collected impulses to ANCF nodes
            # from a unified model double-counts the momentum and explodes.
            self.mpm_solver.setup_collider(
                body_mass=wp.zeros_like(self.model.body_mass),
                body_q=self.state_0.body_q,
            )
            _MAX = 1 << 20
            self._imp = wp.zeros(_MAX, dtype=wp.vec3, device=dev)
            self._imp_ids = wp.full(_MAX, -1, dtype=int, device=dev)
            self._imp_pos = wp.zeros(_MAX, dtype=wp.vec3, device=dev)
            self._collect_impulses()

        # Viewer
        if viewer is not None:
            viewer.set_model(self.model)
            viewer.show_particles = True
            viewer.set_camera(
                pos=wp.vec3(float(n_envs - 1) * x_spacing * 0.5, 0.8, 2.0),
                pitch=-15.0,
                yaw=-90.0,
            )

        # Contact spike visualization (cyan lines, same as ancf_rim_shell)
        self._spike_s = wp.zeros(n_tread_all, dtype=wp.vec3, device=dev)
        self._spike_e = wp.zeros(n_tread_all, dtype=wp.vec3, device=dev)
        self._spike_scale = 20.0  # impulse [N·s] × scale → visible spike height [m]
        self._frame = 0
        self._diag_period = 10  # print every N frames

        print(
            f"[INIT] {n_envs} tire(s) × {n_nodes} nodes | "
            f"{self.model.particle_count - n_envs * n_nodes} sand particles | "
            f"{n_tread_all} tread colliders"
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _collect_impulses(self):
        if self.mpm_solver is None:
            return
        imp, pos, ids = self.mpm_solver.collect_collider_impulses(self.state_0)
        self._imp_ids.fill_(-1)
        n = min(imp.shape[0], self._imp.shape[0])
        if n > 0:
            self._imp[:n].assign(imp[:n])
            self._imp_pos[:n].assign(pos[:n])
            self._imp_ids[:n].assign(ids[:n])

    def _update_contact_viz(self):
        """Build cyan contact spikes: base at tread node, tip height ∝ MPM impulse."""
        wp.launch(
            _build_contact_spikes,
            dim=self._n_tread,
            inputs=[
                self.ancf_solver.node_x,
                self._tread_idx,
                self._tread_imp,
                self._spike_scale,
                self._spike_s,
                self._spike_e,
            ],
            device="cuda:0",
        )

    # ── simulation ─────────────────────────────────────────────────────────────

    def step(self):
        dev = "cuda:0"

        # 1. Scatter previous frame's MPM impulses into per-tread buckets
        self._tread_imp.zero_()
        wp.launch(
            _scatter_impulses,
            dim=self._imp_ids.shape[0],
            inputs=[self._imp_ids, self._imp, self._n_tread, self._tread_imp],
            device=dev,
        )

        # 2. No MPM feedback to ANCF nodes — bounce controlled by kd (contact damping).
        # kd is set from the critical-damping formula: kd = 2√(kn×m) × ζ
        # where ζ = 0.11 gives half-height bounce (e = 0.707).
        self.ancf_solver.node_f_ext_persistent.zero_()

        # 3. ANCF substeps (flat ground at sand surface; MPM provides deformation)
        if self._substep_graph is not None:
            wp.capture_launch(self._substep_graph)
        else:
            for _ in range(self.sim_substeps):
                self.ancf_solver.graph_step()

        if self._with_sand:
            # 4. Sync tread sphere POSITIONS to current ANCF tread positions.
            # Velocities are zeroed — the MPM sees static geometric boundaries,
            # not spheres racing at 15 m/s which drags sand particles upward and
            # creates phantom contacts (J=60 N·s at 21m height).
            wp.launch(
                _copy_tread_to_body_q,
                dim=self._n_tread,
                inputs=[self.state_0.body_q, self.ancf_solver.node_x, self._tread_idx],
                device=dev,
            )
            self.state_0.body_qd.zero_()

            # 5. MPM step (frame rate: 60 Hz)
            self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

            # 6. Collect MPM impulses for next frame + build contact viz
            self._collect_impulses()

        self._update_contact_viz()

        # Mirror ANCF nodes into render particle state
        wp.copy(self.state_0.particle_q, self.ancf_solver.node_x)
        self.sim_time += self.frame_dt
        self._frame += 1

        # ── Debug printout every N frames ─────────────────────────────────
        if self._frame % self._diag_period == 0:
            x_np = self.ancf_solver.node_x.numpy()
            xd_np = self.ancf_solver.node_xd.numpy()
            imp_np = self._tread_imp.numpy()
            mag = np.linalg.norm(imp_np, axis=1)

            nan_x = np.any(np.isnan(x_np))
            nan_v = np.any(np.isnan(xd_np))
            y_ctr = float(x_np[:, 1].mean())  # tire center of mass Y
            y_min = float(x_np[:, 1].min())  # tread bottom
            y_max = float(x_np[:, 1].max())  # tread top
            vy_ctr = float(xd_np[:, 1].mean())  # center vertical velocity
            float(np.abs(xd_np).max())
            n_contact = int((mag > 1e-6).sum())
            imp_sum = float(mag.sum())

            # How far center is above/below drop height (should never go above +0)
            above_drop = y_ctr - self._drop_center_y
            flag = ""
            if nan_x or nan_v:
                flag = "  ← NaN!"
            elif above_drop > 0.001:
                flag = f"  ← BOUNCED {above_drop * 100:.1f}cm ABOVE DROP HEIGHT!"
            elif n_contact > 0:
                flag = "  [contact]"
            else:
                flag = "  [free-fall]"

            print(
                f"[{self._frame:4d}]"
                f"  center_y={y_ctr:.3f}m  (drop={self._drop_center_y:.3f}  Δ={above_drop:+.3f}m)"
                f"  vy={vy_ctr:+.2f}m/s"
                f"  tread=[{y_min:.3f},{y_max:.3f}]"
                f"  contacts={n_contact}/{self._n_tread}"
                f"  J={imp_sum:.1f}N·s" + flag
            )

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Cyan spikes at tread nodes — height ∝ MPM impulse magnitude.
        # No spike = no MPM contact at that node.
        self.viewer.log_lines("contact_spikes", self._spike_s, self._spike_e, (0.0, 1.0, 1.0))  # cyan
        self.viewer.end_frame()

    def test_final(self):
        x_np = self.ancf_solver.node_x.numpy()
        assert not np.any(np.isnan(x_np)), "NaN in node positions"
        assert x_np[:, 1].min() > -0.1, "Tire fell through sand"

    @staticmethod
    def create_parser():
        p = newton.examples.create_parser()
        p.add_argument("--substeps", type=int, default=_SUBSTEPS)
        p.add_argument("--nr-iters", type=int, default=3, dest="nr_iters")
        p.add_argument("--pcg-iters", type=int, default=50, dest="pcg_iters")
        p.add_argument("--kn", type=float, default=2e3, help="ANCF contact stiffness at sand surface [N/m].")
        p.add_argument("--kd", type=float, default=2.0, help="ANCF contact damping [N·s/m].")
        p.add_argument(
            "--n-circ",
            type=int,
            default=_N_CIRC,
            dest="n_circ",
            help="Circumferential elements (overridden per-tire by shell-tires).",
        )
        p.add_argument(
            "--shell-tires",
            type=str,
            default=None,
            dest="shell_tires",
            help="JSON array of per-tire configs (same schema as ancf_shell_drop).",
        )
        p.add_argument(
            "--corner-mass",
            type=float,
            default=0.0,
            dest="corner_mass",
            help="Extra hub mass [kg] for hub vertical integration. "
            "M_total = corner_mass + shell_mass. "
            "Tune for target bounce: M = J_sand/(v_i+v_r) - m_shell.",
        )
        p.add_argument("--mpm-sand-density", type=float, default=None, dest="mpm_sand_density")
        p.add_argument("--mpm-sand-friction", type=float, default=None, dest="mpm_sand_friction")
        p.add_argument("--mpm-sand-yield-pressure", type=float, default=None, dest="mpm_sand_yield_pressure")
        p.add_argument("--mpm-sand-viscosity", type=float, default=None, dest="mpm_sand_viscosity")
        p.add_argument("--mpm-sand-air-drag", type=float, default=None, dest="mpm_sand_air_drag")
        p.add_argument(
            "--mpm-sand-height",
            type=float,
            default=None,
            dest="mpm_sand_height",
            help="Sand pile height [m]. Presence enables MPM sand.",
        )
        p.add_argument(
            "--mpm-sand-half-x",
            type=float,
            default=None,
            dest="mpm_sand_half_x",
            help="Sand bed half-width X per tire [m].",
        )
        p.add_argument(
            "--mpm-sand-half-z", type=float, default=None, dest="mpm_sand_half_z", help="Sand bed half-depth Z [m]."
        )
        p.add_argument(
            "--mpm-sand-voxel-size", type=float, default=None, dest="mpm_sand_voxel_size", help="MPM voxel size [m]."
        )
        return p


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
