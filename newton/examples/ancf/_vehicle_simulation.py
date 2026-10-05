# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Common USD vehicle runtime for flat ground, terrain, sand, and telemetry.

This class owns asset construction, controls, tire state, and the captured
MuJoCo/ANCF stepping path. Every vehicle uses the same coupling selection and
CUDA graphs; vehicle-specific geometry and drive properties come from VehicleUSD.
Scene subclasses extend ``_on_car_builder`` before graph capture and the existing
control/render hooks. Defaults and CLI options live in ``_vehicle_config``; device
coordinate conversions live in ``_vehicle_kernels``.

Rigid state uses Z-up (forward X, left Y); ANCF state uses Y-up
(lateral X, forward Z). Reaction transfer includes tire momentum and torque.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf import _vehicle_config as vehicle_config
from newton.examples.ancf._ancf_viz import (
    _ancf_yup_to_zu,
    _build_ring_lines,
    _gather_zu,
    bead_row_indices,
    material_row,
    quad_triangles,
    ring_segments,
)
from newton.examples.ancf._capture_utils import restore_arrays, snapshot_arrays, try_capture
from newton.examples.ancf._vehicle_kernels import (
    _CONTACT_SPIKE_STRIDE,
    advance_substep_pair,
    fill_spindle_positions,
    gather_contact_spikes,
    prescribe_beads,
    stage_interface_kinematics,
)
from newton.examples.ancf._vehicle_usd import VehicleUSD, asset_path, find_body
from newton.solvers import (
    SolverANCFShellRigid,
    isotropic_ancf_material,
    load_ancf_tire_usd,
)


@dataclass(frozen=True)
class TireSetup:
    """Shell material, thickness, solver budget and pressure resolved by :func:`resolve_tire_setup`."""

    e_tire: float  # [Pa]
    nu_tire: float
    rho_tire: float  # [kg/m^3]
    alpha_d: float  # stiffness-proportional damping [s]
    thickness: float | None  # uniform thickness requested by the options [m]; None = the asset's
    # Thickness for the lumped-mass estimates [m]: thickness, the asset's shellThickness, or the mesh mean.
    h_shell: float
    elem_h: np.ndarray  # per-element thickness the solver runs with [m]
    pcg_iters: int
    kn: float  # [N/m]
    kd: float  # [N·s/m]
    requested_pressure: float  # nominal cavity pressure before the CTIS clamp [Pa]
    pressure: float  # nominal cavity pressure inside the CTIS envelope [Pa] (unchanged when <= 0)
    envelope: tuple  # (unit, Pa/unit, min, max, rate [unit/s], presets, title) from VehicleUSD.ctis_envelope


def resolve_tire_setup(options: Mapping, tire_meta, elem_h: np.ndarray, vehicle: VehicleUSD) -> TireSetup:
    """Apply the tire option precedence shared by :class:`VehicleSimulation` and the diffsim tire-lift setup.

    ``options`` holds parser-style names (``shell_tires``, ``e_tire``, ``nu_tire``, ``rho_tire``,
    ``h_shell``, ``pressure``, ``pcg_iters``, ``kn``, ``kd``); a missing key falls back to
    ``_vehicle_config``.  ``--shell-tires`` (JSON text or list, first entry for every tire) overrides
    the flat material args.  Geometry is fixed at bake time — see
    third_party/newton-tire-tool/scripts/bake_tire.py — so only material, thickness and pressure
    are applied to the loaded mesh (``elem_h`` is the host copy of its per-element thickness).
    """
    shell_tires = options.get("shell_tires")
    if shell_tires:
        if isinstance(shell_tires, str):
            shell_tires = json.loads(shell_tires)
        t0 = shell_tires[0]
        e_tire = float(t0.get("E", vehicle_config.E_TIRE))
        nu_tire = float(t0.get("nu", vehicle_config.NU_TIRE))
        rho_tire = float(t0.get("rho", vehicle_config.RHO_TIRE))
        h_cfg = t0.get("thickness", vehicle_config.H_SHELL)
        thickness = None if h_cfg is None else float(h_cfg)  # None: the asset's shellThickness, else its bands
        alpha_d = float(t0.get("alpha-damp", vehicle_config.ALPHA_D))
        pressure = float(t0.get("pressure", vehicle_config.PRESSURE))
    else:
        e_tire = float(options.get("e_tire", vehicle_config.E_TIRE))
        nu_tire = float(options.get("nu_tire", vehicle_config.NU_TIRE))
        rho_tire = float(options.get("rho_tire", vehicle_config.RHO_TIRE))
        thickness = float(options.get("h_shell", vehicle_config.H_SHELL))
        alpha_d = float(vehicle_config.ALPHA_D)
        pressure = float(options.get("pressure", vehicle_config.PRESSURE))

    h_shell = thickness
    if h_shell is None and tire_meta.shell_thickness is not None:
        h_shell = float(tire_meta.shell_thickness)  # the asset's validated uniform thickness
    if h_shell is None:
        h_shell = float(elem_h.mean())
    else:
        elem_h = np.full(len(elem_h), h_shell, dtype=np.float32)

    # kn and the PCG budget default to the tire asset's recommendation; the CLI / JSON override when given.
    pcg_arg = options.get("pcg_iters")
    kn_arg = options.get("kn")
    kd_arg = options.get("kd")
    pcg_iters = int(pcg_arg) if pcg_arg is not None else int(tire_meta.pcg_iters or vehicle_config.PCG_ITERS)
    kn = float(kn_arg) if kn_arg is not None else float(tire_meta.contact_kn or vehicle_config.KN)
    # kd: CLI > asset recommendation (sized to the vehicle's ride mode at bake time —
    # an undamped rigid-hull ride mode never settles and its bounce can even ratchet the
    # car forward on an asymmetric tread) > the baseline damping ratio.
    if kd_arg is not None:
        kd = float(kd_arg)
    elif tire_meta.contact_kd is not None:
        kd = float(tire_meta.contact_kd)
    else:
        kd = kn * (vehicle_config.KD / vehicle_config.KN)  # same damping ratio as the baseline

    # The nominal pressure is clamped into the vehicle's CTIS envelope: 30 kPa is inside the
    # jeep's 0.5-35 psi but 2x the Sherp's 2.1 psi ceiling, which its slider could not reach
    # for minutes at the Sherp's fill rate.
    envelope = vehicle.ctis_envelope(e_tire, elem_h, tire_meta)
    requested_pressure = pressure
    if pressure > 0.0:
        _unit, per_unit, p_min, p_max, _rate, _presets, _title = envelope
        pressure = min(max(pressure, p_min * per_unit), p_max * per_unit)
    return TireSetup(
        e_tire=e_tire,
        nu_tire=nu_tire,
        rho_tire=rho_tire,
        alpha_d=alpha_d,
        thickness=thickness,
        h_shell=h_shell,
        elem_h=elem_h,
        pcg_iters=pcg_iters,
        kn=kn,
        kd=kd,
        requested_pressure=requested_pressure,
        pressure=pressure,
        envelope=envelope,
    )


def _find_free_joint(model: newton.Model, body: int) -> int:
    """Index of the (single) free joint whose child is ``body``."""
    j_child = model.joint_child.numpy()
    j_type = model.joint_type.numpy()
    free = [j for j in range(len(j_child)) if j_child[j] == body and j_type[j] == int(newton.JointType.FREE)]
    if len(free) != 1:
        raise ValueError(f"expected one free joint on body {body}, found {len(free)}")
    return free[0]


class VehicleSimulation:
    """A vehicle USD asset (--vehicle-asset) on 4 ANCF FEM tires (--tire-asset)."""

    # MuJoCo contact budget (flat plane: rims + arms on one plane). Terrain subclasses raise it.
    _NCONMAX = 128
    _NJMAX = 500

    def __init__(self, viewer=None, args=None):
        device = "cuda:0"
        if getattr(args, "fast_math", False):
            wp.config.fast_math = True
        wp.init()

        # Vehicle and tire are USD assets (--vehicle-asset / --tire-asset, baked by newton-tire-tool;
        # see _vehicle_usd.py). A subclass may construct self.vehicle before calling this __init__.
        if getattr(self, "vehicle", None) is None:
            vehicle_asset = getattr(args, "vehicle_asset", None)
            if not vehicle_asset:
                raise ValueError("--vehicle-asset is required (a vehicle USD baked by newton-tire-tool)")
            self.vehicle = VehicleUSD(asset_path(vehicle_asset))
        # The tire defaults to the one the vehicle asset was built for.
        if not getattr(args, "tire_asset", None):
            if not self.vehicle.default_tire_asset:
                raise ValueError("--tire-asset is required (the vehicle asset names no default tire)")
            args.tire_asset = self.vehicle.default_tire_asset

        self._frame = 0
        self._t = 0.0
        self.viewer = viewer
        self._diag_period = int(getattr(args, "diag_period", 30))
        self._t_wall = time.perf_counter()
        self._gui_fps = 0.0
        self._gui_fz = [0.0] * vehicle_config.N_TIRES
        self._gui_drift = [0.0] * vehicle_config.N_TIRES

        # ── Solver params ─────────────────────────────────────────────────────
        substeps = int(getattr(args, "substeps", vehicle_config.SIM_SUBSTEPS))
        nr_iters = int(getattr(args, "nr_iters", vehicle_config.NR_ITERS))
        mu = float(getattr(args, "mu", vehicle_config.MU))
        sim_dt = vehicle_config.FRAME_DT / substeps
        self._substeps = substeps
        self._sim_dt = sim_dt

        thick_gp = int(getattr(args, "thickness_gp", 3))

        # ── ANCF tire mesh (Y-up: axle along X, tread at Y=0) ────────────────
        self.ancf_model, tire_meta = load_ancf_tire_usd(asset_path(args.tire_asset), device=device)
        # Dimensions / limits from the vehicle USD and the tire USD.
        self.spec = self.vehicle.spec(args.tire_asset, tire_meta)
        n_elems = self.ancf_model.n_elems
        # Material, thickness, contact and pressure (shell-tires JSON array or flat CLI args).
        setup = resolve_tire_setup(vars(args), tire_meta, self.ancf_model.elem_h.numpy(), self.vehicle)
        rho_tire, h_shell, pressure = setup.rho_tire, setup.h_shell, setup.pressure
        pcg_iters, kn, kd = setup.pcg_iters, setup.kn, setup.kd
        mat = isotropic_ancf_material(E=setup.e_tire, nu=setup.nu_tire, rho=rho_tire, alpha_damp=setup.alpha_d)
        self.ancf_model.elem_mat = wp.array(np.tile(material_row(mat), (n_elems, 1)), dtype=float, device=device)
        self.ancf_model.elem_h = wp.array(setup.elem_h, device=device)
        self._tire_meta = tire_meta
        self._e_tire = setup.e_tire

        n_circ = tire_meta.n_circ
        n_bead_rows = tire_meta.n_bead_rows
        n_ax_divs = n_elems // n_circ
        n_bead_per_ring = n_circ
        n_bead = 2 * n_bead_rows * n_bead_per_ring
        n_nodes = self.ancf_model.n_nodes
        x0_np = self.ancf_model.node_x0.numpy()
        d0_np = self.ancf_model.node_D0.numpy()
        d0_np_tiled = np.tile(d0_np, (vehicle_config.N_TIRES, 1)).astype(np.float32)

        r_outer, r_inner, width = self.spec.tire_R_outer, self.spec.tire_R_inner, self.spec.tire_width
        m_tire = rho_tire * h_shell * (2.0 * math.pi * r_outer * width + 2.0 * math.pi * (r_outer**2 - r_inner**2))

        self._n_bead = n_bead
        self._n_nodes = n_nodes

        # ── Bead ring node indices: n_bead_rows rows pinned per side ─────────
        bead_np = bead_row_indices(n_bead_per_ring, n_bead_rows, n_ax_divs)
        if len(bead_np) != n_bead:
            raise ValueError(f"expected {n_bead} bead nodes, got {len(bead_np)}")

        self._bead_idx = wp.array(bead_np, dtype=wp.int32, device=device)
        bead_idx_all_np = np.concatenate([bead_np + e * n_nodes for e in range(vehicle_config.N_TIRES)])
        self._bead_idx_all = wp.array(bead_idx_all_np.astype(np.int32), device=device)
        self._bead_rest = wp.array(x0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)
        self._bead_D0 = wp.array(d0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)
        self._bead_idx_np = bead_np
        self._bead_rest_np = x0_np[bead_np]
        # Free crown node (max radial distance from the axle) for the diagnostics.
        self._crown_idx = int(np.argmax(x0_np[:, 1] ** 2 + x0_np[:, 2] ** 2))
        self._crown_rest_np = x0_np[self._crown_idx]

        # ── ANCF solver (Y-up, gravity along -Y) ─────────────────────────────
        ancf_builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        ancf_newton_model = ancf_builder.finalize(device=device)
        solver_kwargs = {
            "model": ancf_newton_model,
            "ancf_model": self.ancf_model,
            "n_tires": vehicle_config.N_TIRES,
            "torque_alpha": float(getattr(args, "torque_alpha", vehicle_config.TORQUE_ALPHA)),
            "ground_z": float(getattr(args, "ground_z", 0.0)),
            "kn": kn,
            "kd": kd,
            "mu": mu,
            "thickness_gp": thick_gp,
        }
        self.ancf_solver = SolverANCFShellRigid(nr_max_iter=nr_iters, pcg_max_iter=pcg_iters, **solver_kwargs)

        # Place each ANCF tire at its spindle's position, the whole vehicle (built by
        # self.vehicle.build below) lifted by world_z_offset and put at spawn_pose = (x, y, yaw)
        # for the terrain examples (the start pose of the terrain's reference track).
        # Spindle (x_fwd, y_lat, z_up) in Z-up -> ANCF (y_lat, z_up, x_fwd); the yaw turns the
        # tire nodes and their directors about z_up = ANCF component 1.
        z_off = float(getattr(args, "world_z_offset", 0.0))
        spawn = tuple(float(v) for v in getattr(args, "spawn_pose", (0.0, 0.0, 0.0)))
        self._spawn_pose = spawn
        c_yaw, s_yaw = math.cos(spawn[2]), math.sin(spawn[2])

        def yaw_ancf(p: np.ndarray, translate: bool) -> np.ndarray:
            """Rotate ANCF-frame vectors (lat, up, fwd) by the spawn yaw about up, then translate."""
            fwd, lat = p[:, 2], p[:, 0]
            out = np.empty_like(p)
            out[:, 2] = c_yaw * fwd - s_yaw * lat + (spawn[0] if translate else 0.0)
            out[:, 0] = s_yaw * fwd + c_yaw * lat + (spawn[1] if translate else 0.0)
            out[:, 1] = p[:, 1]
            return out

        spindle_zu = self.vehicle.spindle_positions_zu()
        world_x = np.concatenate(
            [
                yaw_ancf(
                    x0_np + np.array([spindle_zu[e, 1], spindle_zu[e, 2] + z_off, spindle_zu[e, 0]], dtype=np.float32),
                    translate=True,
                )
                for e in range(vehicle_config.N_TIRES)
            ],
            axis=0,
        ).astype(np.float32)
        d0_np_tiled = yaw_ancf(d0_np_tiled, translate=False).astype(np.float32)
        self.ancf_solver.node_x.assign(world_x)

        # Dirichlet bead nodes (global indices across all envs).
        bead_global_np = np.concatenate([bead_np + e * n_nodes for e in range(vehicle_config.N_TIRES)])
        self.ancf_solver.set_dirichlet_nodes(bead_global_np)
        self.ancf_solver.debug_residuals = bool(getattr(args, "debug_residuals", False))

        # CTIS (see _ctis.py; per-tire valves or one air line, from the vehicle asset).
        # Start already inflated — the ramp is a realistic 2 psi/s, so filling from the
        # build pressure would leave the jeep on flat tires for seconds and would not
        # reproduce the settled baseline (h=0.42 m, 14-17 kN/corner). resolve_tire_setup
        # clamped the scenario's pressure into the vehicle's CTIS envelope.
        self._build_pressure = float(getattr(args, "build_pressure", vehicle_config.BUILD_PRESSURE))
        self.ctis = None
        if pressure > 0.0:
            self.ctis = self.vehicle.make_ctis(
                self.ancf_solver, vehicle_config.N_TIRES, pressure, self._build_pressure, setup.envelope
            )
        else:
            self.ancf_solver.set_cavity([0.0] * vehicle_config.N_TIRES, [self._build_pressure] * vehicle_config.N_TIRES)

        # ── MuJoCo car builder (Z-up, 1 world) ───────────────────────────────
        car = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car)
        car.default_shape_cfg.mu = mu
        # The rigid vehicle (spindle bodies named by the asset, chassis free joint first).
        half_bead = float(np.abs(self._bead_rest_np[:, 0]).max())  # bead ring axial half-width
        self.vehicle.build(car, z_off, self._tire_meta.spindle, self.spec.tire_R_inner, half_bead, pose=spawn)
        self._on_car_builder(car)

        # Resolve the DOF indices _launch_drive writes to; the wheel motors become newton.actuators.
        self.vehicle.setup_drive(car, device, yaw=spawn[2], wheel_actuators=True)
        # Steer/throttle commands ride on a 2-element device buffer so GUI changes
        # do not invalidate a captured graph.
        self._cmd_host = np.zeros(2, dtype=np.float32)
        self.cmd = wp.zeros(2, dtype=wp.float32, device=device)

        # ANCF visualization particles (massless, Z-up) + triangles.
        world_x_zu = np.stack([world_x[:, 2], world_x[:, 0], world_x[:, 1]], axis=1).astype(np.float32)
        car.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x_zu],
            vel=[(0.0, 0.0, 0.0)] * (vehicle_config.N_TIRES * n_nodes),
            mass=[0.0] * (vehicle_config.N_TIRES * n_nodes),
            radius=[0.001] * (vehicle_config.N_TIRES * n_nodes),
        )
        all_tris = quad_triangles(self.ancf_model.elem_nodes.numpy(), n_nodes, vehicle_config.N_TIRES)
        car.add_triangles(i=all_tris[:, 0].tolist(), j=all_tris[:, 1].tolist(), k=all_tris[:, 2].tolist())

        self.model = car.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()
        # joint_q start of the chassis free joint: [0:3] is the chassis position (z at +2).
        self.chassis_q0 = int(
            self.model.joint_q_start.numpy()[_find_free_joint(self.model, find_body(self.model, "chassis"))]
        )

        # Pre-compress the tires so they can carry the TSDA preload on frame 0. Only a rough
        # estimate (docker/config/README.md) — the settle phase after graph capture below finds
        # the actual static sag, so this just gives it a shorter distance to travel.
        ride_drop = float(getattr(args, "ride_drop", vehicle_config.RIDE_DROP))
        if ride_drop != 0.0:
            jq = self.model.joint_q.numpy()
            jq[self.chassis_q0 + 2] -= ride_drop
            self.model.joint_q.assign(jq)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ── MuJoCo solver ─────────────────────────────────────────────────────
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            iterations=50,
            ls_iterations=10,
            njmax=self._NJMAX,
            nconmax=self._NCONMAX,
            # Must be 1: InterfaceCouplerGS rewinds state_0 to t_n between GS
            # iterations and step_kinematics only pushes state_0 into mjData
            # when update_data_interval > 0.
            update_data_interval=1,
        )
        # graph_conditional stays at its default (True): mujoco_warp then wraps the
        # solver loop in a capture_while node and exits once the world converges.
        # With it disabled the frame graph ran all 50 iterations every substep
        # (500 per frame, ~13 launch-bound kernels each; nsys 2026-09-09).

        # ── Spindle names -> MuJoCo body indices; register per-wheel coupling ──
        btow = self.solver.mjc_body_to_newton.numpy()[0]
        spindle_mj_list = []
        self._spindle_body_indices_np = []
        for name in self.vehicle.spindle_bodies:
            nidx = find_body(self.model, name)
            self._spindle_body_indices_np.append(nidx)
            m = np.where(btow == nidx)[0]
            if not len(m):
                raise ValueError(f"spindle '{name}' (newton={nidx}) not in MuJoCo body map")
            spindle_mj_list.append(int(m[0]))
        self._spindle_mj_arr = wp.array(spindle_mj_list, dtype=wp.int32, device=device)
        self._spindle_newton_arr = wp.array(self._spindle_body_indices_np, dtype=int, device=device)
        for w in range(vehicle_config.N_TIRES):
            self.ancf_solver.setup_wheel(
                tire_idx=w,
                spindle_mj=spindle_mj_list[w],
                bead_idx_np=(bead_np + w * n_nodes).astype(np.int32),
                tare_fz=0.0,
                world_idx=0,
                lateral_offset=0.0,
                device=device,
            )
        print(
            f"[DW] vehicle={self.vehicle.name} ({self.vehicle.kind}, {self.vehicle.steering} steer)"
            f"  tire={os.path.basename(self.spec.tire_asset)}  spindle_mj={spindle_mj_list}  "
            f"m_tire={m_tire:.3f} kg  torque_alpha={self.ancf_solver.torque_alpha:.3g}  "
            f"substeps={substeps} nr={nr_iters} pcg={pcg_iters}"
        )

        # ── GUI state ─────────────────────────────────────────────────────────
        self.steer_angle = float(getattr(args, "steer_angle", 0.0))
        self.wheel_speed = float(getattr(args, "wheel_speed", 0.0))
        self._target_wheel_speed = self.wheel_speed

        # ── Visualization buffers (bead rings, spokes, contact spikes) ───────
        seg_s_all, seg_e_all = ring_segments(n_bead_per_ring, n_bead_rows, n_bead, vehicle_config.N_TIRES)
        n_segs = len(seg_s_all)
        self._bead_pos_zu = wp.zeros(vehicle_config.N_TIRES * n_bead, dtype=wp.vec3, device=device)
        self._spoke_start_zu = wp.zeros(vehicle_config.N_TIRES * n_bead, dtype=wp.vec3, device=device)
        self._ring_seg_s = wp.array(seg_s_all, dtype=wp.int32, device=device)
        self._ring_seg_e = wp.array(seg_e_all, dtype=wp.int32, device=device)
        self._ring_line_s = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._ring_line_e = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._n_ring_segs = n_segs
        self._contact_line_s = wp.zeros(vehicle_config.N_TIRES * n_nodes, dtype=wp.vec3, device=device)
        self._contact_line_e = wp.zeros(vehicle_config.N_TIRES * n_nodes, dtype=wp.vec3, device=device)
        self._contact_vis_scale = 100.0

        # ── ANCF graph capture, then restore the state its warm-up consumed ───
        self.ancf_solver.capture_graph(sim_dt)
        self.reset_state(world_x, d0_np_tiled)

        # Warm up kinematics + bead prescription so MuJoCo data is initialised.
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, sim_dt)
        self._prescribe_beads()
        self._update_viz_buffers()

        # ── Interface Gauss-Seidel coupling ───────────────────────────────────
        self._gs_iters = max(1, int(getattr(args, "gs_iters", vehicle_config.GS_ITERS)))
        self._gs_coupler = None
        self._gs_prescribe_toggle = 0
        coupling_method = getattr(args, "coupling_method", "auto")
        if coupling_method == "coupled-newton" and self._substeps % 2:
            raise ValueError("Coupled Newton coupling requires an even number of substeps.")
        if coupling_method == "auto":
            # Response reuse depends on the rigid interface size and graph support,
            # independently of the vehicle asset or tire mesh resolution.
            can_reuse = self.state_0.joint_qd.shape[0] <= 64 and self._substeps % 2 == 0
            terrain = self.ancf_solver.terrain
            can_condense = can_reuse and self.model.articulation_count == 1 and (terrain is None or terrain.rigid)
            # Models coupled Newton rejects (singular rigid mass, > 32 dofs) fall
            # back to adaptive Aitken inside the coupler.
            coupling_method = "coupled-newton" if can_condense and self._gs_iters >= 3 else "adaptive"
        if self._gs_iters > 1:
            self._gs_coupler = newton.solvers.InterfaceCouplerGS(
                self.ancf_solver,
                n_iters=self._gs_iters,
                acceleration=True,
                adaptive=True,
                coupled_newton=coupling_method == "coupled-newton",
            )
            # elem_eas_alpha is mutated by the ANCF step and is not part of the
            # coupler's built-in pack, so it must be registered to be rewound.
            extra = [self.ancf_model.elem_eas_alpha]
            self._gs_coupler.allocate(
                self.state_0,
                extra_arrays=extra,
                model=self.model,
                interface_body_indices=self._spindle_body_indices_np if coupling_method == "coupled-newton" else None,
                kinematic_state_arrays=[
                    getattr(self.solver.mjw_data, name)
                    for name in ("qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com")
                ],
            )

        # MuJoCo kinematics/dynamics as standalone graphs (usable inside the GS
        # loop); with gs_iters == 1 the whole substep loop is one frame graph.
        self._kin_graph = None
        self._dyn_graph = None
        self._substep_graph = None
        self._try_capture_mujoco_graphs()
        if self._gs_coupler is None:
            self._try_capture_substep_graph()
        if self._kin_graph is not None:
            self._gs_kinematics_fn = lambda *a: wp.capture_launch(self._kin_graph)
        else:
            self._gs_kinematics_fn = self.solver.step_kinematics
        if self._dyn_graph is not None:
            self._gs_dynamics_fn = lambda *a: wp.capture_launch(self._dyn_graph)
        else:
            self._gs_dynamics_fn = self.solver.step_dynamics

        if self._gs_coupler is not None:
            self._try_capture_coupled_graph()

        if viewer is not None:
            viewer.set_model(self.model)
            viewer.set_camera(
                pos=wp.vec3(*self.spec.camera_pos), pitch=self.spec.camera_pitch, yaw=self.spec.camera_yaw
            )

    def _on_car_builder(self, car: newton.ModelBuilder) -> None:
        """Subclass hook: the vehicle is built, ``self.ancf_solver`` exists and nothing is captured yet.

        Add world shapes (terrain) here and attach anything the ANCF solver must bake into its graph.
        """

    def _launch_drive(self) -> None:
        """Write ``self.cmd`` ([0] steer, [1] wheel speed) into the joint targets (graph-safe)."""
        self.vehicle.launch_drive(self.cmd, self.control.joint_target_q, self.control.joint_target_qd)

    def _step_wheel_actuators(self) -> None:
        """This substep's wheel motor effort (newton.actuators) into ``control.joint_f``, read by step_kinematics."""
        self.control.joint_f.zero_()
        for act in self.model.actuators:
            act.step(self.state_0, self.control, dt=self._sim_dt)

    def reset_state(
        self,
        node_x: np.ndarray,
        node_D: np.ndarray,
        body_q: np.ndarray | None = None,
        joint_q: np.ndarray | None = None,
    ) -> None:
        """Place the tires at ``node_x`` / ``node_D`` (ANCF Y-up, all tires flat) with every velocity,
        acceleration, force history and EAS parameter zeroed.

        With ``body_q`` and ``joint_q`` both rigid states take that pose at rest and the applied
        wrenches and MuJoCo's warm start, which belong to the previous state, are dropped as well.
        Callers re-run ``step_kinematics`` / ``_prescribe_beads`` afterwards.
        """
        a = self.ancf_solver
        if body_q is not None:
            for st in (self.state_0, self.state_rigid):
                st.body_q.assign(body_q)
                st.body_qd.zero_()
                st.joint_q.assign(joint_q)
                st.joint_qd.zero_()
        a.node_x.assign(node_x)
        a.node_xd.zero_()
        a.node_xdd.zero_()
        a.node_D.assign(node_D)
        a.node_Dd.zero_()
        a.node_Ddd.zero_()
        a.global_f_int.zero_()
        a.global_f_int0.zero_()
        a.global_f_ext.zero_()
        if hasattr(a, "global_f_ext0"):
            a.global_f_ext0.zero_()
        a.node_f_ext_persistent.zero_()
        self.ancf_model.elem_eas_alpha.zero_()
        if body_q is not None:
            self.solver.xfrc_applied.zero_()
            ws = getattr(self.solver.mjw_data, "qacc_warmstart", None)
            if ws is not None:
                ws.zero_()

    # ── Graph capture ─────────────────────────────────────────────────────────

    def _try_capture_coupled_graph(self) -> None:
        """Capture an even number of substeps so snapshot buffer indices repeat."""
        if self._substeps % 2:
            return
        coupler = self._gs_coupler
        initial_indices = coupler._ancf_cur, coupler._rigid_cur
        self._substep_pairs = wp.zeros(1, dtype=int, device="cuda:0")

        def advance():
            self._step_wheel_actuators()
            coupler.substep(
                self.state_0,
                self.state_rigid,
                self.control,
                self._sim_dt,
                self.solver.step_kinematics,
                self._prescribe_beads_gs,
                self._accumulate_wrenches,
                self.solver.step_dynamics,
                trial_kinematics_fn=self._trial_kinematics if coupler.acceleration else None,
                ancf_step_fn=lambda: self.ancf_solver.step(None, None, None, None, self._sim_dt),
            )

        def pair():
            # Two substeps restore the snapshot-buffer ordering for the next loop.
            advance()
            advance()
            wp.launch(advance_substep_pair, dim=1, inputs=[self._substep_pairs], device="cuda:0")

        def frame():
            if coupler._coupled_solver is not None:
                self._substep_pairs.fill_(self._substeps // 2)
                wp.capture_while(self._substep_pairs, pair)
            else:
                for _ in range(self._substeps):
                    advance()

        self._substep_graph = try_capture(frame, "[DW] Coupled graph capture unavailable: {error!r}")
        if self._substep_graph is not None:
            method = (
                "coupled shell/interface Newton"
                if coupler._coupled_solver is not None
                else "adaptive Aitken"
                if coupler.adaptive
                else "Aitken"
            )
            maximum = coupler._coupled_solver.maximum + 1 if coupler._coupled_solver is not None else self._gs_iters
            print(f"[DW] Captured {self._substeps} substeps with up to {maximum} {method} interface iterations")
        coupler._ancf_cur, coupler._rigid_cur = initial_indices

    def _try_capture_mujoco_graphs(self) -> None:
        """Capture step_kinematics and step_dynamics as standalone CUDA graphs."""
        dev = "cuda:0"
        dt = self._sim_dt
        wp.synchronize_device(dev)
        self._kin_graph = try_capture(
            lambda: self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt),
            "[DW] kinematics graph capture failed: {error!r}",
            dev,
        )
        wp.synchronize_device(dev)
        self._dyn_graph = try_capture(
            lambda: self.solver.step_dynamics(self.state_rigid), "[DW] dynamics graph capture failed: {error!r}", dev
        )

        # Instantiate the graph execs outside any capture context.
        wp.synchronize_device(dev)
        if self._kin_graph is not None:
            wp.capture_launch(self._kin_graph)
        if self._dyn_graph is not None:
            wp.capture_launch(self._dyn_graph)
        wp.synchronize_device(dev)

    def _try_capture_substep_graph(self) -> None:
        """Capture one full frame (all substeps, explicit coupling) as a single CUDA graph.

        Uses ancf.step() (unrolled NR+PCG) since a graph launch is not capturable.
        Saves and restores all simulation state around the capture warm-up.
        """
        dev = "cuda:0"
        ancf = self.ancf_solver
        dt = self._sim_dt

        def live_state():
            return {
                "node_x": ancf.node_x,
                "node_xd": ancf.node_xd,
                "node_xdd": ancf.node_xdd,
                "node_D": ancf.node_D,
                "node_Dd": ancf.node_Dd,
                "node_Ddd": ancf.node_Ddd,
                "f_int": ancf.global_f_int,
                "f_int0": ancf.global_f_int0,
                "eas": self.ancf_model.elem_eas_alpha,
                "s0_bq": self.state_0.body_q,
                "s0_bqd": self.state_0.body_qd,
                "s0_jq": self.state_0.joint_q,
                "s0_jqd": self.state_0.joint_qd,
                "sr_bq": self.state_rigid.body_q,
                "sr_bqd": self.state_rigid.body_qd,
                "sr_jq": self.state_rigid.joint_q,
                "sr_jqd": self.state_rigid.joint_qd,
            }

        wp.synchronize_device(dev)
        saved = snapshot_arrays(live_state())

        force_history = {
            name: wp.clone(getattr(ancf, name)) for name in ("global_f_ext", "global_f_ext0") if hasattr(ancf, name)
        }

        def _restore():
            for name, value in force_history.items():
                wp.copy(getattr(ancf, name), value)
            restore_arrays(saved, live_state())

        # Force ANCF graph exec instantiation outside any capture context.
        ancf.graph_step()
        wp.synchronize_device(dev)
        _restore()

        def _ancf_step():
            ancf.step(None, None, None, None, dt)

        def frame():
            for _ in range(self._substeps):
                self._one_substep(self.solver.step_kinematics, _ancf_step, self.solver.step_dynamics)

        self._substep_graph = try_capture(
            frame, "[DW] frame graph capture failed ({error!r}) — falling back to loop mode", dev
        )
        if self._substep_graph is not None:
            print(f"[DW] frame graph captured ({self._substeps} substeps, 1 launch/frame)")

        wp.synchronize_device(dev)
        _restore()

    # ── Coupling helpers ───────────────────────────────────────────────────────

    def _trial_kinematics(self, *_args):
        """Stage the already-predicted Newton wheel poses without MuJoCo dynamics setup."""
        wp.launch(
            stage_interface_kinematics,
            dim=vehicle_config.N_TIRES,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.model.body_com,
                self._spindle_newton_arr,
                self._spindle_mj_arr,
                self.solver.xpos,
                self.solver.xquat,
                self.solver.cvel,
                self.solver.mjw_data.subtree_com,
                self.solver.mjw_model.body_rootid,
            ],
            device="cuda:0",
        )

    def _prescribe_beads(self, inv_dt: float = 0.0, vel_predict_dt: float = 0.0) -> None:
        wp.launch(
            prescribe_beads,
            dim=vehicle_config.N_TIRES * self._n_bead,
            inputs=[
                self.solver.xpos,
                self.solver.xquat,
                self.solver.cvel,
                self.solver.mjw_data.subtree_com,
                self.solver.mjw_model.body_rootid,
                self._spindle_mj_arr,
                self._bead_idx,
                self._bead_rest,
                self._bead_D0,
                self.ancf_solver.node_x,
                self.ancf_solver.node_xd,
                self.ancf_solver.node_xdd,
                self.ancf_solver.node_D,
                self.ancf_solver.node_Dd,
                self.ancf_solver.node_Ddd,
                self._n_bead,
                self._n_nodes,
                inv_dt,
                vel_predict_dt,
            ],
            device="cuda:0",
        )

    def _prescribe_beads_gs(self) -> None:
        """Zero-arg ``prescribe_fn`` for ``InterfaceCouplerGS.substep()``.

        Called twice per GS iteration: before the ANCF step (predict the hub
        pose forward by v*dt at k=0 and write a consistent bead acceleration)
        and after it (snap to the solved pose).  At GS iteration k>0
        ``solver.xpos`` already holds the k-th estimate of the t_{n+1} pose, so
        no extrapolation is applied.
        """
        predict = self._sim_dt if self._gs_coupler.gs_iter == 0 and not self._gs_coupler.acceleration else 0.0
        if self._gs_prescribe_toggle == 0:
            self._prescribe_beads(inv_dt=1.0 / self._sim_dt, vel_predict_dt=predict)
        else:
            # Post-step: same pose as pre-step (k = 0: extrapolated t_n; k > 0: the t_{n+1} estimate).
            self._prescribe_beads(vel_predict_dt=predict)
        self._gs_prescribe_toggle = 1 - self._gs_prescribe_toggle

    def _accumulate_wrenches(self) -> None:
        self.solver.xfrc_applied.zero_()
        self.ancf_solver.accumulate_wheel_wrenches(
            xfrc_applied=self.solver.xfrc_applied, xpos=self.solver.xpos, device="cuda:0"
        )

    def _copy_rigid_to_state0(self) -> None:
        wp.copy(self.state_0.body_q, self.state_rigid.body_q)
        wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
        wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
        wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)

    def _one_substep(self, kin_fn, ancf_step_fn, dyn_fn) -> None:
        """One explicit coupled substep (wheel motors -> kinematics -> beads -> ANCF -> beads -> wrench -> dynamics).

        ``kin_fn`` / ``dyn_fn`` take the ``step_kinematics`` / ``step_dynamics`` arguments; under
        capture they are the direct solver calls, in the fallback loop the graph launches.
        """
        self._step_wheel_actuators()
        kin_fn(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
        self.ancf_solver.begin_coupling_step(self._sim_dt)
        self._prescribe_beads(inv_dt=1.0 / self._sim_dt, vel_predict_dt=self._sim_dt)
        ancf_step_fn()
        # Beads stay at the end-of-step (extrapolated) pose the solve used; snapping back to the
        # start pose offset the ring from carcass and rim by v_hub·dt (see example 02).
        self._prescribe_beads(vel_predict_dt=self._sim_dt)
        self._accumulate_wrenches()
        dyn_fn(self.state_rigid)
        self._copy_rigid_to_state0()

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self) -> None:
        """Run one frame (substeps steps)."""
        if self._substep_graph is not None:
            wp.capture_launch(self._substep_graph)
            return
        if self._gs_coupler is not None:
            for _ in range(self._substeps):
                self._step_wheel_actuators()
                self._gs_coupler.substep(
                    self.state_0,
                    self.state_rigid,
                    self.control,
                    self._sim_dt,
                    self._gs_kinematics_fn,
                    self._prescribe_beads_gs,
                    self._accumulate_wrenches,
                    self._gs_dynamics_fn,
                    trial_kinematics_fn=self._trial_kinematics if self._gs_coupler.acceleration else None,
                )
            return

        for _ in range(self._substeps):
            self._one_substep(self._gs_kinematics_fn, self.ancf_solver.graph_step, self._gs_dynamics_fn)

    def _update_controls(self) -> None:
        """Upload steer/throttle commands for this frame (2 floats over the bus)."""
        self._cmd_host[0] = self.steer_angle
        self._cmd_host[1] = self.wheel_speed
        self.cmd.assign(self._cmd_host)
        self._launch_drive()

    def _update_viz_buffers(self) -> None:
        dev = "cuda:0"
        wp.launch(
            _ancf_yup_to_zu,
            dim=vehicle_config.N_TIRES * self._n_nodes,
            inputs=[self.ancf_solver.node_x, self.state_0.particle_q],
            device=dev,
        )
        wp.launch(
            _gather_zu,
            dim=vehicle_config.N_TIRES * self._n_bead,
            inputs=[self.state_0.particle_q, self._bead_idx_all, self._bead_pos_zu],
            device=dev,
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_zu, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device=dev,
        )
        wp.launch(
            fill_spindle_positions,
            dim=vehicle_config.N_TIRES * self._n_bead,
            inputs=[self.solver.xpos, self._spindle_mj_arr, self._spoke_start_zu, self._n_bead],
            device=dev,
        )

    def step(self) -> None:
        # CTIS air amounts ramp toward the GUI setpoints; build stays fixed.
        if self.ctis is not None:
            self.ctis.step(vehicle_config.FRAME_DT)

        # Ramp wheel speed toward the target.
        d = self._target_wheel_speed - self.wheel_speed
        if abs(d) <= vehicle_config.WHEEL_SPEED_RATE:
            self.wheel_speed = self._target_wheel_speed
        else:
            self.wheel_speed += vehicle_config.WHEEL_SPEED_RATE if d > 0.0 else -vehicle_config.WHEEL_SPEED_RATE

        self._update_controls()
        self.simulate()
        self._frame += 1
        self._t += vehicle_config.FRAME_DT

        self._update_viz_buffers()

        if self._frame % self._diag_period == 0:
            now = time.perf_counter()
            fps = self._diag_period / max(now - self._t_wall, 1e-9)
            self._gui_fps = fps
            self._t_wall = now
            self._print_diag(fps)

    # ── Diagnostics (host readback only every --diag-period frames) ───────────

    def _print_diag(self, fps: float = 0.0) -> None:
        x_all = self.ancf_solver.node_x.numpy()
        stg_per_wheel = [self.ancf_solver._xfrc_stg_per_tire[w].numpy()[0] for w in range(vehicle_config.N_TIRES)]
        # MuJoCo's cached xpos/xquat precede integration; body_q is at t_{n+1}.
        spindle_poses = self.state_0.body_q.numpy()[self._spindle_body_indices_np]
        nn = self._n_nodes
        rest_np = self._bead_rest_np
        if self.ctis is not None:
            self.ctis.read_live()

        print(f"\n[{self._frame:4d}] t={self._t:.2f}s  fps={fps:.1f}")
        if self.ancf_solver.debug_residuals:
            for e in range(vehicle_config.N_TIRES):
                print(f"  {self.ancf_solver.residual_report(e)}")
        for label, e in vehicle_config.WHEEL_ORDER:
            x_e = x_all[e * nn : (e + 1) * nn]
            any_nan = bool(np.any(np.isnan(x_e)))
            fz = float(stg_per_wheel[e][4])
            pos_zu = spindle_poses[e, :3]
            hub = np.array([float(pos_zu[1]), float(pos_zu[2]), float(pos_zu[0])])
            # Compare bead/crown positions to the integrated spindle at the same time.
            qx, qy, qz, qw = (float(value) for value in spindle_poses[e, 3:])
            u = np.array([qx, qy, qz])

            def rot(r_mj, qw=qw, u=u):
                return (
                    r_mj * (qw * qw - u @ u)
                    + 2.0 * (r_mj @ u)[..., None] * u
                    + 2.0 * qw * np.cross(np.broadcast_to(u, r_mj.shape), r_mj)
                )

            r_mj = np.stack([rest_np[:, 2], rest_np[:, 0], rest_np[:, 1]], axis=1)
            p_mj = rot(r_mj) + pos_zu
            expected = np.stack([p_mj[:, 1], p_mj[:, 2], p_mj[:, 0]], axis=1)
            drift_mm = float(np.max(np.linalg.norm(x_e[self._bead_idx_np] - expected, axis=1))) * 1e3
            crown_mj = np.array([[self._crown_rest_np[2], self._crown_rest_np[0], self._crown_rest_np[1]]])
            cp = rot(crown_mj)[0] + pos_zu
            crown_drift_mm = float(np.linalg.norm(x_e[self._crown_idx] - np.array([cp[1], cp[2], cp[0]]))) * 1e3
            self._gui_fz[e] = fz
            self._gui_drift[e] = drift_mm
            p_txt = ""
            if self.ctis is not None:
                p_txt = f"  p={self.ctis.live_p[e]:.0f}Pa V/V0={self.ctis.live_v_ratio[e]:.3f}"
            print(
                f"  {label}: {'NaN!' if any_nan else 'ok  '}"
                f"  hub=({hub[0]:.3f},{hub[1]:.3f},{hub[2]:.3f})"
                f"  drift={drift_mm:.2f}mm  crown_drift={crown_drift_mm:.1f}mm  Fz={fz:+.0f}N{p_txt}"
            )

    # ── GUI ────────────────────────────────────────────────────────────────────

    def gui(self, ui) -> None:
        self.vehicle.gui_drive(ui, self)  # title + steer/throttle sliders + derived speed readouts

        if self.ctis is not None:
            self.ctis.gui(ui, vehicle_config.WHEEL_ORDER)

        ui.separator()
        ui.text("Live")
        ui.text(f"  fps   {self._gui_fps:6.1f}")
        for label, e in vehicle_config.WHEEL_ORDER:
            ui.text(f"  {label}  Fz={self._gui_fz[e]:+.0f} N  drift={self._gui_drift[e]:.2f} mm")

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self._t)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("bead_rings", self._ring_line_s, self._ring_line_e, colors=(1.0, 0.45, 0.0))
        self.viewer.log_lines("bead_spokes", self._spoke_start_zu, self._bead_pos_zu, colors=(1.0, 0.90, 0.1))
        wp.launch(
            gather_contact_spikes,
            dim=vehicle_config.N_TIRES * self._n_nodes,
            inputs=[
                self.ancf_solver.node_x,
                self._n_nodes,
                0.0,
                self._contact_vis_scale,
                _CONTACT_SPIKE_STRIDE,
                2,
                self._contact_line_s,
                self._contact_line_e,
            ],
            device="cuda:0",
        )
        self.viewer.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, colors=(0.0, 1.0, 1.0))
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        return vehicle_config.create_parser()
