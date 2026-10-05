# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learn ground friction through a rolling ANCF tire's dynamic trajectory.

A gray reference tire runs a prescribed spin-and-brake command with an unknown
friction coefficient. The white tire receives exactly the same command. Its
spindle is free to translate; traction determines its travel and speed. Learn
one friction coefficient from both histories, then test a different command.
Pressure, stiffness, mass and motor commands are fixed during identification.

Uses the Warthog ANCF3423 shell asset with ANS/EAS, cavity pressure, and tanh
Coulomb friction. A dedicated dense float64 backward-Euler solver integrates
shell and translating spindle together, propagating the analytic friction
sensitivity through every converged step. This is an isolated dynamics and
gradient experiment, not MuJoCo/HHT autodiff or measured terrain calibration.
Normal contact damping and structural Rayleigh damping are zero; backward
Euler supplies numerical damping. An ideal velocity motor prescribes rim spin.

The CPU solve deliberately prioritizes converged derivatives over real-time
speed. The viewer remains responsive between steps. OPENBLAS_NUM_THREADS=1 is
recommended. No finite differences or reference friction values enter updates.

Run: python -m newton.examples diffsim_ancf_tire_traction
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf._ancf_viz import bead_row_indices, material_row, quad_triangles
from newton.examples.ancf._vehicle_usd import asset_path
from newton.examples.ancf.diffsim._ancf_common import COLORS, PA_PER_PSI
from newton.solvers import (
    ANCFTireEquilibrium,
    ANCFTireTraction,
    SolverANCFShell,
    isotropic_ancf_material,
    load_ancf_tire_usd,
)

_PRESSURE = 2.0 * PA_PER_PSI
_MASS = 50.0
_DT = 0.01
_MU_BOUNDS = (0.15, 1.0)


def make_experiment(dt=_DT):
    """Build one fixed tire/load experiment on CPU; return dynamics, reset state, mesh and hub."""
    mesh, meta = load_ancf_tire_usd(asset_path("warthog_ancf_tire_simple.usda"), device="cpu")
    material = material_row(isotropic_ancf_material(50e6, 0.45, 700.0, alpha_damp=0.0))
    mesh.elem_mat.assign(np.tile(material, (mesh.n_elems, 1)))
    if meta.shell_thickness is not None:
        mesh.elem_h.fill_(meta.shell_thickness)
    model = newton.ModelBuilder(up_axis=newton.Axis.Y).finalize(device="cpu")
    source = SolverANCFShell(model, mesh, kn=meta.contact_kn, kd=0.0, v_reg=0.01)
    beads = bead_row_indices(meta.n_circ, meta.n_bead_rows, mesh.n_nodes // meta.n_circ - 1)
    equilibrium = ANCFTireEquilibrium(source, beads, _MASS, build_pressure=0.0)
    coordinates = equilibrium.coordinates(mesh.node_x0.numpy() + np.array([0, 0.27, 0]), mesh.node_D0.numpy(), 0.27)
    static = equilibrium.solve(_PRESSURE, coordinates)
    dynamics = ANCFTireTraction(equilibrium, _PRESSURE, _MASS, dt=dt, contact_damping=0.0)
    return dynamics, dynamics.initial_state(static), mesh, meta.spindle


def motor_commands(validation=False, dt=_DT):
    """Fixed ideal-motor speed [rad/s]; validation changes amplitude and braking time."""
    t = (np.arange(round(0.8 / dt)) + 1) * dt
    if validation:
        return np.where(t <= 0.30 + 1e-9, 7.0 * np.minimum(t / 0.12, 1.0), 0.0)
    return np.where(t <= 0.40 + 1e-9, 10.0 * np.minimum(t / 0.10, 1.0), 0.0)


class Example:
    """Observe, fit one friction coefficient, and validate a new motion sequence."""

    def __init__(self, viewer, args):
        if not math.isfinite(args.reference_mu) or not _MU_BOUNDS[0] <= args.reference_mu <= _MU_BOUNDS[1]:
            raise ValueError("Reference friction must be in [0.15, 1.0]")
        if not math.isfinite(args.initial_mu) or not _MU_BOUNDS[0] <= args.initial_mu <= _MU_BOUNDS[1]:
            raise ValueError("Initial friction must be in [0.15, 1.0]")
        if args.report and args.report.suffix.lower() != ".json":
            raise ValueError("Use a .json report path")
        self.viewer, self.args = viewer, args
        self.dynamics, self.initial, mesh, spindle = make_experiment()
        self.reference_mu = args.reference_mu
        self.initial_mu = args.initial_mu
        self._commands = {False: motor_commands(), True: motor_commands(True)}
        self.references = {}
        self.history = []
        self.results = {}
        self._paths = {}
        self._frame = 0
        self._pending = None
        self._report_written = False
        self._build_view(mesh, spindle)
        self._restart(references=True)

    def _build_view(self, mesh, spindle):
        points = self.initial.q[:, 0, [2, 0, 1]]
        triangles = quad_triangles(mesh.elem_nodes.numpy(), mesh.n_nodes, 1)
        scene = newton.ModelBuilder()
        scene.add_particles([wp.vec3(*p) for p in points], [wp.vec3(0.0)] * len(points), [0.0] * len(points))
        for a, b, c in triangles:
            scene.add_triangle(int(a), int(b), int(c))
        scene.add_ground_plane()
        self.model = scene.finalize()
        self.state = self.model.state()
        device = self.model.device
        self._triangles = wp.array(triangles.ravel(), dtype=int, device=device)
        self._ghost_points = wp.array(points, dtype=wp.vec3, device=device)
        if spindle is None:
            raise ValueError("The learning tire requires its baked spindle asset")
        self._spindle = spindle.points
        self._hub_indices = wp.array(spindle.triangle_indices.ravel(), dtype=int, device=device)
        self._hub_points = wp.empty(len(spindle.points), dtype=wp.vec3, device=device)
        self._ghost_hub = wp.empty_like(self._hub_points)
        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = False
            self.viewer.set_camera(pos=(1.7, -2.5, 1.3), pitch=-23.0, yaw=125.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.draw_edges = False

    def _restart(self, *, references=False):
        if references:
            self.references.clear()
        self.mu = self.initial_mu
        self.history.clear()
        self.results.clear()
        self._paths.clear()
        self._accepted_loss = math.inf
        self._accepted_mu = self.mu
        self._next_log_step = 0.0
        self._backtracks = 0
        self._report_written = False
        self._begin("Reference" if False not in self.references else "Before")

    def _begin(self, phase):
        self.phase = phase
        self.current = self.initial
        self.ghost = self.initial
        self._trace = []
        self._index = 0
        self._validation = phase.startswith("Validation")
        if phase in ("Validation reference", "Validation before"):
            self._paths.clear()
        self.status = phase
        print(f"[traction] {phase}; candidate mu={self.mu:.5f}", flush=True)

    def step(self):
        self._frame += 1
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if pending == "validation":
                self._begin("Validation before")
            else:
                self._restart(references=pending == "ground")
        if self.phase in ("Ready", "Stopped"):
            if (
                self.phase == "Ready"
                and self.args.report
                and not self._report_written
                and "validation_after" in self.results
            ):
                self.export_report(self.args.report)
                self._report_written = True
            return
        reference = self.phase in ("Reference", "Validation reference")
        mu = self.reference_mu if reference else self.initial_mu if self.phase == "Validation before" else self.mu
        command = self._commands[self._validation][self._index]
        self.current = self.dynamics.step(self.current, float(command), mu, sensitivity=not reference)
        self._trace.append(self.current)
        self.ghost = self.current if reference else self.references[self._validation][self._index]
        self._index += 1
        self.status = f"{self.phase}: {self._index}/{len(self._commands[self._validation])} steps"
        if self._index == len(self._commands[self._validation]):
            self._finish()

    def _finish(self):
        if self.phase in ("Reference", "Validation reference"):
            self.references[self._validation] = self._trace
            self._begin("Validation before" if self._validation else "Before")
            return
        loss, gradient, curvature = self.dynamics.loss_gradient(self._trace, self.references[self._validation])
        position_rmse = float(
            np.sqrt(
                np.mean(
                    [
                        (s.z[-1] - r.z[-1]) ** 2
                        for s, r in zip(self._trace, self.references[self._validation], strict=True)
                    ]
                )
            )
        )
        result = {"mu": self.mu, "loss": loss, "position_rmse_m": position_rmse}
        if self._validation:
            key = "validation_before" if self.phase == "Validation before" else "validation_after"
            result["mu"] = self.initial_mu if key == "validation_before" else self.mu
            self.results[key] = result
            name = "Before" if key == "validation_before" else "Learned"
            self._paths[name] = [[float(s.z[-1]), 0.0, 0.008] for s in self._trace]
            self._paths["Reference"] = [[float(s.z[-1]), 0.0, 0.012] for s in self.references[True]]
            if key == "validation_before":
                self._begin("Validation learned")
            else:
                self.phase, self.status = "Ready", "Validated on a different motor command"
            print(f"[traction] {key}: {result}", flush=True)
            return
        if loss > self._accepted_loss * (1 + 1e-8):
            self._backtracks += 1
            if self._backtracks > 8:
                self.phase, self.status = "Stopped", "No improving friction step"
                return
            self._next_log_step *= 0.5
            self.mu = float(np.clip(self._accepted_mu * math.exp(self._next_log_step), *_MU_BOUNDS))
            self._begin("Learning")
            return
        self._accepted_mu, self._accepted_loss = self.mu, loss
        self.history.append({**result, "gradient": gradient})
        self._backtracks = 0
        path = [[float(s.z[-1]), 0.0, 0.008] for s in self._trace]
        self._paths["Before" if self.phase == "Before" else "Learned"] = path
        self._paths["Reference"] = [[float(s.z[-1]), 0.0, 0.012] for s in self.references[False]]
        print(f"[traction] update {len(self.history) - 1}: {self.history[-1]}", flush=True)
        if self.phase == "Before":
            self.results["before"] = result
            if not self.args.train:
                self.phase, self.status = "Ready", "Before learning — press Learn friction"
                return
        if loss < 1e-7:
            self.results["after"] = result
            self._begin("Validation reference" if True not in self.references else "Validation before")
            return
        if curvature < 1e-10 or len(self.history) >= 10:
            self.phase, self.status = "Stopped", "Friction is unobservable or the fit did not converge"
            return
        self._next_log_step = float(np.clip(-gradient / (curvature * self.mu), -0.4, 0.4))
        candidate = float(np.clip(self.mu * math.exp(self._next_log_step), *_MU_BOUNDS))
        if abs(candidate - self.mu) < 1e-8:
            self.phase, self.status = "Stopped", "Friction fit reached its bounds"
            return
        self.mu = candidate
        self._begin("Learning")

    def export_report(self, path):
        """Save synthetic experiment provenance, learned friction and independent validation."""
        if "validation_after" not in self.results:
            raise RuntimeError("Finish learning and validation before exporting")
        report = {
            "format": "newton.ancf_traction/1",
            "data_source": "synthetic single-wheel trajectories",
            "reference_mu": self.reference_mu,
            "initial_mu": self.initial_mu,
            "learned_mu": self.mu,
            "nominal_pressure_pa": _PRESSURE,
            "rim_and_carried_mass_kg": _MASS,
            "dt_s": _DT,
            "integrator": "backward Euler, free translating rim, prescribed spin, zero structural/normal damping",
            "gradient": "analytic implicit forward sensitivity through all shell/rim/contact steps",
            "training_motor_rad_s": self._commands[False].tolist(),
            "validation_motor_rad_s": self._commands[True].tolist(),
            "history": self.history,
            "results": self.results,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def gui(self, ui):
        if getattr(self, "_ui_frames", 0) < 2:
            for label in ("Model Information", "Visualization", "Rendering Options", "Controls"):
                ui.get_state_storage().set_int(ui.get_id(label), 0)
            self._ui_frames = getattr(self, "_ui_frames", 0) + 1
        ui.text("Learn tire-ground friction")
        ui.text(f"Estimated friction: {self.mu:.3f}")
        if self.phase == "Validation before":
            ui.text(f"This test uses initial friction: {self.initial_mu:.3f}")
        changed, ground = ui.combo("Ground", int(self.reference_mu >= 0.5), ["Slippery", "Grippy"])
        if changed:
            self.reference_mu = (0.35, 0.65)[ground]
            self.initial_mu = 0.9 if ground == 0 else 0.2
            self.args.train = True
            self._pending = "ground"
        if ui.button("Learn friction"):
            self.args.train = True
            self._pending = "learn"
        ui.same_line()
        ui.begin_disabled("validation_after" not in self.results)
        if ui.button("Replay test"):
            self._pending = "validation"
        ui.end_disabled()
        ui.text_wrapped(self.status)
        ui.text(f"Travel: {self.current.z[-1]:.3f} m")
        ui.text(f"Speed: {self.current.zd[-1]:.2f} m/s")
        if self.history:
            ui.text(f"Trajectory loss: {self.history[-1]['loss']:.6f}")
        for label, key in (("Before", "before"), ("Learned", "after")):
            result_key = "validation_" + key if "validation_after" in self.results else key
            if result_key in self.results:
                ui.text(f"{label} error: {100 * self.results[result_key]['position_rmse_m']:.2f} cm RMS")
        ui.text_wrapped("Gray = reference. White = simulated tire. Both receive the same wheel-speed command.")
        ui.text_wrapped("Only friction changes. Pressure and load stay fixed.")
        if ui.collapsing_header("Advanced"):
            ui.text(f"Synthetic reference friction: {self.reference_mu:.3f}")
            ui.text("2 PSI; 50 kg rim + carried load")
            ui.text_wrapped(
                "Analytic trajectory derivative, CPU implicit ANCF solve. Independent spin-and-brake validation follows learning."
            )

    def _hub(self, state):
        c, s = math.cos(state.angle), math.sin(state.angle)
        rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        points = self._spindle @ rotation.T + [0, state.z[-2], state.z[-1]]
        return points[:, [2, 0, 1]]

    def render(self):
        if self.viewer is None:
            return
        self.state.particle_q.assign(self.current.q[:, 0, [2, 0, 1]])
        self._ghost_points.assign(self.ghost.q[:, 0, [2, 0, 1]])
        self._hub_points.assign(self._hub(self.current))
        self._ghost_hub.assign(self._hub(self.ghost))
        self.viewer.begin_frame(self._frame * _DT)
        self.viewer.log_state(self.state)
        reference_phase = self.phase in ("Reference", "Validation reference")
        tire_key = self.viewer._qualify("/model/triangles") if hasattr(self.viewer, "_qualify") else "/model/triangles"
        tire = getattr(self.viewer, "objects", {}).get(tire_key)
        if tire is not None:
            tire.hidden = reference_phase
        self.viewer.log_mesh(
            "/traction/hub", self._hub_points, self._hub_indices, color=(0.35, 0.35, 0.75), hidden=reference_phase
        )
        coincident = np.linalg.norm(self.current.z[-2:] - self.ghost.z[-2:]) < 1e-4
        for name, points, indices in (
            ("/traction/reference", self._ghost_points, self._triangles),
            ("/traction/reference_hub", self._ghost_hub, self._hub_indices),
        ):
            self.viewer.log_mesh(
                name,
                points,
                indices,
                color=COLORS["Reference"],
                hidden=coincident and not reference_phase,
                backface_culling=False,
            )
            key = self.viewer._qualify(name) if hasattr(self.viewer, "_qualify") else name
            obj = getattr(self.viewer, "objects", {}).get(key)
            if obj is not None:
                obj.alpha, obj.cast_shadow = 0.35, False
        for name, color in COLORS.items():
            points = np.asarray(self._paths.get(name, []), dtype=np.float32).reshape(-1, 3)
            self.viewer.log_lines(
                f"/traction/{name}",
                wp.array(points[:-1], dtype=wp.vec3, device=self.model.device),
                wp.array(points[1:], dtype=wp.vec3, device=self.model.device),
                color,
            )
        self.viewer.end_frame()

    def test_final(self):
        from newton.tests.ancf_diffsim_checks import check_tire_traction_result  # noqa: PLC0415

        check_tire_traction_result(self)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1600)
        parser.add_argument(
            "--reference-mu",
            type=float,
            default=0.65,
            help="Synthetic ground friction used only to generate observations, 0.15–1.0.",
        )
        parser.add_argument(
            "--initial-mu",
            type=float,
            default=0.2,
            help="Initial friction estimate, 0.15–1.0; pressure, stiffness and carried mass remain fixed.",
        )
        parser.add_argument(
            "--train",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Fit friction with implicit trajectory derivatives, then validate a different spin/brake sequence; "
            "--no-train shows the initial mismatch until Learn friction is pressed.",
        )
        parser.add_argument(
            "--report",
            type=Path,
            help="Save a .json report with synthetic provenance, commands, accepted losses and independent validation "
            "errors; this is not a calibrated telemetry preset.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
