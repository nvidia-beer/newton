# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Example ANCF Shell Drop
#
# A Polaris ANCF3423 shell tire (40×12 elements, orthotropic 3-section
# material, 71-point Polaris cross-section profile) is dropped from 1 m
# onto a rigid ground plane and bounces elastically.
#
# Solver: SolverANCFShell — implicit HHT-alpha (α=-0.2, β=0.36, γ=0.7)
#   - 5-point through-thickness Gauss quadrature
#   - ANS transverse-shear + ANS ε_zz correction
#   - EAS (5 modes per element) locking remedy
#   - Penalty ground contact: kn=2e6 N/m, kd=13 N·s/m, μ=0.9
#   - Full BSR sparse K_eff with in-place scatter_map update
#   - Diagonal-preconditioned PCG (custom SpMV, graph-capture safe)
#
# Materials: Polaris 3-section orthotropic (bead / sidewall / tread)
#   The reference (uninflated) geometry IS the inflated shape — elastic
#   forces restore the tire to its Polaris profile.
#
# Command: uv run -m newton.examples ancf.example_ancf_shell_drop
###########################################################################

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
        self.sim_substeps = int(getattr(args, "substeps", 5))
        self.sim_dt = self.frame_dt / self.sim_substeps

        device = "cuda:0"

        # ----------------------------------------------------------------
        # Parse per-tire config from "tires" JSON array (if provided).
        # Falls back to a single default Polaris tire when absent.
        # ----------------------------------------------------------------
        tires_raw = getattr(args, "shell_tires", None)
        if isinstance(tires_raw, str):
            tires_raw = json.loads(tires_raw)
        if tires_raw:
            tire_cfgs = [t for t in tires_raw if t.get("active", True)]
        else:
            # Default: one Polaris tire at the origin
            tire_cfgs = [{"name": "polaris_ref", "position": [0.0, 0.0, 0.0]}]

        n_envs = len(tire_cfgs)

        # ----------------------------------------------------------------
        # Build shared tire geometry (topology + reference positions).
        # All tires share the same Polaris profile; materials differ.
        # ----------------------------------------------------------------
        R_outer = 0.329
        n_circ = int(tire_cfgs[0].get("n-circ", getattr(args, "n_circ", 20)))

        self.ancf_model = build_ancf_tire_mesh(
            R_outer=R_outer,
            R_inner=0.13,
            width=0.23,
            n_circ=n_circ,
            section_divs=(1, 2, 3),
            pressure=float(tire_cfgs[0].get("pressure", 110e3)),
            device=device,
        )

        ne = self.ancf_model.n_elems
        n_nodes = self.ancf_model.n_nodes

        # ----------------------------------------------------------------
        # Build per-tire elem_mat arrays, concatenate → [N*n_elems, 11].
        # "E": null  →  use Polaris 3-section orthotropic preset.
        # "E": float →  use isotropic_ancf_material(E, nu, rho).
        # ----------------------------------------------------------------
        base_mat_np = self.ancf_model.elem_mat.numpy()  # (n_elems, 11) Polaris default

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
                # Polaris orthotropic preset (copy base, optionally scale rho)
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
        en = self.ancf_model.elem_nodes.numpy()  # (n_elems, 4)
        tris_one = np.empty((len(en) * 2, 3), dtype=np.int32)
        tris_one[0::2] = en[:, [0, 1, 2]]
        tris_one[1::2] = en[:, [0, 2, 3]]
        tris_all = np.concatenate([tris_one + i * n_nodes for i in range(n_envs)], axis=0)
        builder.add_triangles(
            i=tris_all[:, 0].tolist(),
            j=tris_all[:, 1].tolist(),
            k=tris_all[:, 2].tolist(),
        )
        self.model = builder.finalize(device=device)

        # ----------------------------------------------------------------
        # Solver
        # ----------------------------------------------------------------
        kn = float(getattr(args, "kn", 2e3))
        kd = float(getattr(args, "kd", 2.0))
        self.solver = SolverANCFShell(
            model=self.model,
            ancf_model=self.ancf_model,
            ground_z=0.0,
            kn=kn,
            kd=kd,
            mu=0.9,
            v_reg=1.0e-3,
            nr_max_iter=int(getattr(args, "nr_iters", 3)),
            pcg_max_iter=int(getattr(args, "pcg_iters", 50)),
            n_envs=n_envs,
        )
        # Seed the solver node positions with the dropped/offset world layout.
        self.solver.node_x.assign(world_x)

        for i, cfg in enumerate(tire_cfgs):
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

        # ── Combined substep frame graph ──────────────────────────────────────
        # Captures all sim_substeps iterations into ONE CUDA graph launch per frame,
        # eliminating Python loop overhead (same pattern as ancf_rigid_mujoco_tires).
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

        if self._frame % 5 == 0:
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
        "--n-circ",
        type=int,
        default=20,
        help="Elements around circumference (20=fast/stable, 40=full Polaris reference).",
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
    parser.add_argument("--pcg-iters", type=int, default=50, help="PCG iterations per NR step.")
    parser.add_argument(
        "--shell-tires",
        type=str,
        default=None,
        help="JSON array of per-tire configs (position, material, pressure, etc.).",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
