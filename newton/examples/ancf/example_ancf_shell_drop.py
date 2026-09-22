# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Example ANCF Shell Drop
#
# ANCF3423 shell tire(s) loaded from a baked USD asset (--tire-asset; e.g. the
# Polaris 3-section orthotropic profile or the lugged Sherp carcass) dropped
# from 1 m onto a rigid ground plane. Sealed-gas cavity per tire with a CTIS
# slider; --n-envs batches independent tires in one solver instance.
#
# Solver: SolverANCFShell — implicit HHT-alpha (α=-0.2, β=0.36, γ=0.7)
#   - 5-point through-thickness Gauss quadrature
#   - ANS transverse-shear + ANS ε_zz correction
#   - EAS (5 modes per element) locking remedy
#   - Penalty ground contact: defaults kn=2e3 N/m, kd=2.0 N·s/m, μ=0.9
#     (--kn / --kd; kn scales as kn_chrono x (dt_chrono/dt)^2)
#   - Node-block CSR K_eff updated in place (graph-capture safe)
#   - Diagonal-preconditioned PCG (custom SpMV, graph-capture safe)
#
# Materials: the asset's baked per-section material ("E": null) or an isotropic
#   override per tire. The reference (uninflated) geometry IS the inflated
#   shape — elastic forces restore the tire to its baked profile.
#
# Command: python -m newton.examples ancf_shell_drop
###########################################################################

import json
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf._ancf_viz import material_row, quad_triangles
from newton.solvers import SolverANCFShell, isotropic_ancf_material, load_ancf_tire_usd

# Baked by third_party/newton-tire-tool/scripts/bake_tire.py — the tire mesh
# (node grid, quad connectivity, section thickness/material layout) is
# authored offline; this example only loads it.


@wp.kernel
def _gather_contact_spikes(
    node_x: wp.array[wp.vec3],  # ANCF Y-up, all envs flat
    ground_y: float,
    vis_scale: float,
    line_starts: wp.array[wp.vec3],
    line_ends: wp.array[wp.vec3],
):
    """GPU-only contact visualization: a spike of height pen*vis_scale per penetrating node.

    The model here is built Y-up (``up_axis=newton.Axis.Y``), so unlike the
    Z-up viewers used by ``example_ancf_rigid_mujoco_tires`` /
    ``example_vehicle_ancf_tires``, no axis conversion is needed.
    """
    i = wp.tid()
    p = node_x[i]
    pen = ground_y - p[1]
    base = wp.vec3(p[0], ground_y, p[2])
    if pen > 0.0:
        line_starts[i] = base
        line_ends[i] = wp.vec3(base[0], base[1] + pen * vis_scale, base[2])
    else:
        line_starts[i] = base
        line_ends[i] = base


class Example:
    """ANCF shell tire(s) dropped onto rigid terrain.

    Pass ``--n-envs N`` to batch N independent tires in a single solver instance,
    spaced 1 m apart along X.  Performance scales ~linearly up to ~20–30 envs.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._diag_period = int(args.diag_period)

        device = "cuda:0"

        # ----------------------------------------------------------------
        # Parse per-tire config from the --shell-tires JSON array (if provided).
        # Falls back to a single tire with the asset's material when absent.
        # ----------------------------------------------------------------
        tires_raw = args.shell_tires
        if isinstance(tires_raw, str):
            tires_raw = json.loads(tires_raw)
        if tires_raw:
            tire_cfgs = [t for t in tires_raw if t.get("active", True)]
        else:
            # Default: one tire at the origin with the asset's baked material
            tire_cfgs = [{"name": "reference", "position": [0.0, 0.0, 0.0]}]

        # --n-envs above the number of configured tires replicates them (cycling the
        # materials) 1 m apart along X — the batched-solver workload for profiling.
        n_envs_arg = int(args.n_envs or 1)
        if n_envs_arg > len(tire_cfgs):
            base_cfgs = tire_cfgs
            tire_cfgs = []
            for i in range(n_envs_arg):
                c = dict(base_cfgs[i % len(base_cfgs)])
                c["position"] = [float(i), 0.0, 0.0]
                c["name"] = f"{c.get('name', 'tire')}_{i}"
                tire_cfgs.append(c)

        n_envs = len(tire_cfgs)

        # ----------------------------------------------------------------
        # Load shared tire geometry (topology + reference positions) from the
        # baked USD asset. All tires in a run share this same mesh; materials
        # differ (below). The mesh is fixed at bake time — see
        # third_party/newton-tire-tool/scripts/bake_tire.py (Polaris/parabolic
        # profiles) or bake_sherp_ancf_tire.py (profile extracted from a real
        # tire STL). --tire-asset selects a filename under assets/, or an
        # absolute path.
        # ----------------------------------------------------------------
        tire_asset_arg = args.tire_asset
        if not tire_asset_arg:
            raise ValueError("--tire-asset is required (an ANCF tire USD baked by newton-tire-tool)")
        tire_asset_path = (
            tire_asset_arg
            if os.path.isabs(tire_asset_arg)
            else os.path.join(os.path.dirname(__file__), "assets", tire_asset_arg)
        )
        self.ancf_model, tire_meta = load_ancf_tire_usd(tire_asset_path, device=device)
        R_outer = tire_meta.R_outer

        ne = self.ancf_model.n_elems
        n_nodes = self.ancf_model.n_nodes

        # ----------------------------------------------------------------
        # Build per-tire elem_mat arrays, concatenate → [N*n_elems, 11].
        # "E": null  →  the asset's baked per-section material.
        # "E": float →  use isotropic_ancf_material(E, nu, rho).
        # ----------------------------------------------------------------
        base_mat_np = self.ancf_model.elem_mat.numpy()  # (n_elems, 11) the asset's material

        per_env_mats = []
        for cfg in tire_cfgs:
            E = cfg.get("E", None)
            nu = cfg.get("nu", None)
            rho = cfg.get("rho", None)
            if E is not None:
                # Isotropic material — build a uniform mat for all elements
                mat = isotropic_ancf_material(
                    E=float(E),
                    nu=float(nu) if nu is not None else 0.3,
                    rho=float(rho) if rho is not None else 1100.0,
                )
                per_env_mats.append(np.tile(material_row(mat), (ne, 1)))
            else:
                # Baked material (copy base, optionally override rho)
                env_mat = base_mat_np.copy()
                if rho is not None:
                    env_mat[:, 9] = float(rho)
                per_env_mats.append(env_mat)

        if n_envs > 1:
            self.ancf_model.elem_mat = wp.array(np.concatenate(per_env_mats, axis=0), dtype=float, device=device)

        # ----------------------------------------------------------------
        # Per-tire world node positions = reference shape + offset + drop.
        # JSON "position" = [x, y, z] world offset of the tire centroid.
        # ----------------------------------------------------------------
        x0_np = self.ancf_model.node_x0.numpy()  # (n_nodes, 3) reference
        drop_height = R_outer + 1.0
        world_x = np.empty((n_envs * n_nodes, 3), dtype=np.float32)
        for i, cfg in enumerate(tire_cfgs):
            pos = cfg.get("position", [float(i), 0.0, 0.0])
            off = np.array([float(pos[0]), float(pos[1]) + drop_height, float(pos[2])], dtype=np.float32)
            world_x[i * n_nodes : (i + 1) * n_nodes] = x0_np + off

        # ----------------------------------------------------------------
        # Newton model: ground plane + ALL tire nodes as particles, with the
        # quad mesh registered as triangle elements so the viewer draws each
        # tire as a surface via the standard log_state() pipeline.
        # The ANCF solver owns the dynamics; these tris are render topology.
        # ----------------------------------------------------------------
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane(color=(0.65, 0.65, 0.65))
        builder.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x],
            vel=[(0.0, 0.0, 0.0)] * (n_envs * n_nodes),
            mass=[1.0] * (n_envs * n_nodes),
            radius=[0.001] * (n_envs * n_nodes),
        )

        # Two triangles per quad element; tile node indices per tire.
        tris_all = quad_triangles(self.ancf_model.elem_nodes.numpy(), n_nodes, n_envs)
        builder.add_triangles(
            i=tris_all[:, 0].tolist(),
            j=tris_all[:, 1].tolist(),
            k=tris_all[:, 2].tolist(),
        )
        self.model = builder.finalize(device=device)

        # ----------------------------------------------------------------
        # Solver
        # ----------------------------------------------------------------
        self.solver = SolverANCFShell(
            model=self.model,
            ancf_model=self.ancf_model,
            ground_z=0.0,
            kn=float(args.kn),
            kd=float(args.kd),
            mu=0.9,
            v_reg=1.0e-3,
            nr_max_iter=int(args.nr_iters),
            pcg_max_iter=int(args.pcg_iters or tire_meta.pcg_iters or 50),
            n_envs=n_envs,
        )
        # Seed the solver node positions with the dropped/offset world layout.
        self.solver.node_x.assign(world_x)

        for i, cfg in enumerate(tire_cfgs):
            if n_envs > 8 and 4 <= i < n_envs - 2:
                if i == 4:
                    print(f"[ANCF] ... {n_envs - 6} more tires ...")
                continue
            e = world_x[i * n_nodes : (i + 1) * n_nodes]
            print(
                f"[ANCF] env={i} ({cfg.get('name', '')})  "
                f"centroid=({e[:, 0].mean():.3f},{e[:, 1].mean():.3f},{e[:, 2].mean():.3f})  "
                f"y_min={e[:, 1].min():.3f}"
            )

        # Capture ANCF inner graph (1 substep NR+PCG).
        self.solver.capture_graph(self.sim_dt)

        self._n_nodes = n_nodes
        self._n_envs = n_envs

        # ── Contact visualization (spike per penetrating node) ─────────────────
        self._contact_line_s = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=device)
        self._contact_line_e = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=device)
        self._contact_vis_scale = 100.0  # tune per kn

        # ── Combined substep frame graph ──────────────────────────────────────
        # Captures all sim_substeps iterations into ONE CUDA graph launch per frame,
        # eliminating Python loop overhead (same pattern as the vehicle examples).
        # Uses solver.step() (unrolled NR+PCG) — graph_step() cannot be nested.
        self._substep_graph = None
        _dev = "cuda:0"
        _dt = self.sim_dt

        # Pre-warm solver.step() so kernels compile before capture_begin.
        self.solver.step(None, None, None, None, _dt)
        wp.synchronize_device(_dev)
        # Restore state after pre-warm
        self.solver.node_x.assign(world_x)
        self.solver.node_xd.zero_()
        self.solver.node_xdd.zero_()
        self.solver.global_f_int.zero_()
        self.solver.global_f_int0.zero_()
        self.solver.node_f_ext_persistent.zero_()
        self.ancf_model.elem_eas_alpha.zero_()

        try:
            wp.capture_begin(device=_dev)
            for _ in range(self.sim_substeps):
                self.solver.step(None, None, None, None, _dt)
            self._substep_graph = wp.capture_end(device=_dev)
            print(f"[ANCF] Frame graph captured {self.sim_substeps} substeps — 1 launch/frame ✓")
        except Exception as e:
            try:
                wp.capture_end(device=_dev)
            except Exception:
                pass
            self._substep_graph = None
            print(f"[ANCF] Frame graph capture failed ({e!r}) — using graph_step loop")

        # Particle state holds ALL tire nodes; the viewer renders the surface.
        self.state_0 = self.model.state()
        wp.copy(self.state_0.particle_q, self.solver.node_x)

        # ----------------------------------------------------------------
        # CTIS — sealed-gas cavity per tire.  The slider sets the NOMINAL
        # pressure (the pressure the cavity holds at its reference volume), which
        # maps to the air amount K_gas = p_nominal · V_ref.  The live pressure
        # self-regulates as p = K_gas / V(x): inflate → volume grows → pressure
        # eases off, so the sweep is stable.  The shell sees the gauge load
        # (p − build_pressure); at the build pressure the load is zero and the
        # tire holds its built shape.  The slider target ramps gradually.
        # ----------------------------------------------------------------
        self._tire_names = [cfg.get("name", f"tire{i}") for i, cfg in enumerate(tire_cfgs)]
        self._build_pressure = [float(cfg.get("pressure", 110000.0)) for cfg in tire_cfgs]
        self._max_gauge = [float(cfg.get("max-gauge-pressure", 20000.0)) for cfg in tire_cfgs]
        self._gauge_rate = [float(cfg.get("gauge-rate", 2000.0)) for cfg in tire_cfgs]
        # Nominal-pressure slider bounds (clamped at 0 = fully deflated).
        self._p_min = [max(0.0, self._build_pressure[i] - self._max_gauge[i]) for i in range(n_envs)]
        self._p_max = [self._build_pressure[i] + self._max_gauge[i] for i in range(n_envs)]
        self._target_pressure = list(self._build_pressure)
        self._current_pressure = list(self._build_pressure)
        # Seed the cavity so the first step replays with valid gas state.
        self.solver.set_cavity(self._current_pressure, self._build_pressure)

        if viewer is not None:
            viewer.set_model(self.model)
            # Camera centers on the midpoint of all tires.
            xs = [cfg.get("position", [float(i)])[0] for i, cfg in enumerate(tire_cfgs)]
            viewer.set_camera(
                pos=wp.vec3(float(np.mean(xs)), 1.5, 3.0),
                pitch=-15.0,
                yaw=-90.0,
            )

        self._frame = 0

    def step(self):
        self._frame += 1

        # Ramp the nominal cavity pressure toward the slider target (gradual),
        # then update the sealed-gas cavity (air amount + build pressure).
        for i in range(self._n_envs):
            d = self._target_pressure[i] - self._current_pressure[i]
            r = self._gauge_rate[i]
            if abs(d) <= r:
                self._current_pressure[i] = self._target_pressure[i]
            else:
                self._current_pressure[i] += r if d > 0.0 else -r
        self.solver.set_cavity(self._current_pressure, self._build_pressure)

        if self._substep_graph is not None:
            wp.capture_launch(self._substep_graph)
        else:
            for _ in range(self.sim_substeps):
                self.solver.graph_step()

        # Mirror all tire nodes into the Newton particle state for log_state.
        wp.copy(self.state_0.particle_q, self.solver.node_x)
        self.sim_time += self.frame_dt

        # Host readback of env-0 only every --diag-period frames.
        if self._frame % self._diag_period == 0:
            x_np = self.solver.node_x.numpy()
            xd_np = self.solver.node_xd.numpy()
            # Track env-0 (first tire) only
            n = self._n_nodes
            x0 = x_np[:n]
            xd0 = xd_np[:n]
            y_min = float(x0[:, 1].min())
            y_max = float(x0[:, 1].max())
            y_ctr = float(x0[:, 1].mean())
            vy_min = float(xd0[:, 1].min())
            vy_max = float(xd0[:, 1].max())
            in_contact = y_min < 0.01
            print(
                f"[{self._frame:4d}]  y_ctr={y_ctr:.4f}  y_min={y_min:.4f}"
                f"  y_max={y_max:.4f}  vy=[{vy_min:.3f},{vy_max:.3f}]"
                f"  p={self._current_pressure[0]:.0f} Pa"
                f"{'  CONTACT' if in_contact else ''}"
            )

    def gui(self, ui):
        """One nominal-pressure (CTIS) slider + live readout per tire [Pa].

        Slider starts at the build pressure; drag down to deflate, up to inflate.
        The sealed gas self-regulates the actual pressure as the tire deforms.
        """
        ui.text("CTIS pressure  (down = deflate, up = inflate)")
        ui.separator()
        for i, name in enumerate(self._tire_names):
            changed, value = ui.slider_float(
                f"{name} [Pa]##{i}", self._target_pressure[i], self._p_min[i], self._p_max[i]
            )
            if changed:
                self._target_pressure[i] = float(value)
            # Commanded nominal pressure in Pa + psi (1 psi = 6894.76 Pa).
            ui.text(
                f"   {self._current_pressure[i]:8.0f} Pa"
                f"   ({self._current_pressure[i] / 6894.76:.1f} psi)"
                f"   build {self._build_pressure[i]:.0f}"
            )

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        # All tires render through the standard model pipeline (particles + tris).
        self.viewer.log_state(self.state_0)
        wp.launch(
            _gather_contact_spikes,
            dim=self._n_envs * self._n_nodes,
            inputs=[self.solver.node_x, 0.0, self._contact_vis_scale, self._contact_line_s, self._contact_line_e],
            device="cuda:0",
        )
        self.viewer.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, colors=(0.0, 1.0, 1.0))
        self.viewer.end_frame()

    def test_final(self):
        x_np = self.solver.node_x.numpy()
        assert not np.any(np.isnan(x_np)), "NaN in node positions"
        assert not np.any(np.isinf(x_np)), "Inf in node positions"
        min_y = float(x_np[:, 1].min())
        assert min_y > -0.05, f"Excessive ground penetration: y_min={min_y:.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--tire-asset",
        type=str,
        default=None,
        help=(
            "Baked ANCF tire USD to load — filename under examples/ancf/assets/ or an absolute "
            "path. Required; run-examples.sh prompts for it. See "
            "third_party/newton-tire-tool/scripts/bake_tire.py / bake_sherp_ancf_tire.py."
        ),
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel tire environments (batched solver). Spaced 1 m apart along X.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=5,
        help="Substeps per frame. dt = 1/60/substeps. Stability: kn·β·dt²/M_node ≈ 4.",
    )
    parser.add_argument(
        "--kn", type=float, default=2e3, help="Normal contact stiffness [N/m]. Scale as kn_chrono×(dt_chrono/dt)²."
    )
    parser.add_argument(
        "--kd", type=float, default=2.0, help="Normal contact damping [N·s/m]. Critical damping: 2√(kn·M_node)≈3.8."
    )
    parser.add_argument("--nr-iters", type=int, default=3, help="Newton-Raphson iterations per substep.")
    parser.add_argument(
        "--diag-period",
        type=int,
        default=60,
        help="Print env-0 height / velocity diagnostics every N frames (host readback).",
    )
    parser.add_argument(
        "--pcg-iters",
        type=int,
        default=None,
        help="PCG iterations per NR step (default: the tire asset's recommendation).",
    )
    parser.add_argument(
        "--shell-tires",
        type=str,
        default=None,
        help="JSON array of per-tire configs (position, material, pressure, etc.).",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
