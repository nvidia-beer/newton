# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""See tire deformation, then learn pressure for a loaded height on flat ground.

Use Low PSI / High PSI or drag the pressure slider to inspect the white tire.
A purple cylinder matches the physical spindle. A translucent gray cylinder
marks the requested target height. The tire uses the same smooth shading as
the other ANCF examples, without mesh edges.
The target moves immediately with the height slider, even while paused.
All 3D geometry is true scale; markers are visual only. Choose a target ride
height and Learn pressure to repeat the optimization. The main panel's tread
detail compares the low/high-pressure shapes with the live tire at their axles.

The default demonstration uses an experimental 0.25–8 nominal PSI range, checked
with static and coupled inflation/deflation and learning runs on both supports
at the default Warthog load. The height slider is computed from converged
equilibria after checking intermediate pressures for monotonic height response and
ground penetration below 25 mm. Use ``--pressure-range telemetry`` to retain
the source vehicle's limits (2–4 PSI for Warthog). CSV fitting and replay export
select the telemetry range automatically; the wider range is a simulation
demonstration, not a validated vehicle operating envelope.

Ground switches between the telemetry preset's contact settings and a firmer flat support
(twice the contact stiffness and damping). It rebuilds the forward solver
and recomputes the pressure bounds' shapes, keeping the selected height goal.
These are penalty-contact compliance experiments, not calibrated terrain
materials, uneven terrain or deformable soil. Learning minimizes static height
error; it does not optimize traction or train a reusable terrain controller.

Run with ``python -m newton.examples diffsim_ancf_tire_lift`` (requires CUDA).
Reuses example 02's single-wheel rig: a simplified Warthog ANCF tire, a
MuJoCo spindle free to move vertically, and an additional downward load.
Spin and the horizontal velocity target are zero. Assets, material, contact,
build pressure, and telemetry limits are resolved from ``--telemetry-config``.
The default additional load is the asset's estimated chassis share per wheel,
not a measured corner load. Override it with ``--load`` for a known experiment.
The preview uses 10 substeps / 2 Newton iterations, with 20 PCG iterations
for the softer demo range and the original 10 for telemetry calibration.

The displayed live loss is 0.5 * (height - target_height)**2 [m^2]. Learning
uses the converged static counterpart of this objective, with analytic implicit
pressure sensitivities, including rim balance, EAS, follower pressure, and contact.
MuJoCo continues advancing the physical rim; its transient trajectory is not
differentiated. Use --no-train for manual pressure control.

Without measurements, the default target is near the opposite end of the
reachable height range, making the pressure change easy to see. It is a
synthetic demonstration that checks the optimization path only.
``--calibration-csv`` instead fits one shared elastic-modulus multiplier at
fixed nominal pressures and loads. Required CSV columns are
``nominal_pressure_pa,additional_load_n,axle_height_m,height_std_m,split``;
split is ``train`` or ``validation``. At least two distinct training cases are
required. Additional load excludes spindle weight; height is the axle centre
above a flat test surface, not the telemetry ghost's base_link height.
Pressures must already be expressed in the model's nominal convention.

``--export-config PATH`` writes a complete vehicle_telemetry replay config and
a sibling ``PATH.stem.report.json`` after convergence. These are preparation
artifacts: neither a synthetic target nor a static fit validates vehicle motion.

Pressure uses the existing cavity convention: nominal pressure sets gas
amount at reference volume, while build pressure remains fixed. PSI is only
a unit conversion here, not a calibrated gauge-pressure interpretation.
The differentiation mathematics lives in newton.solvers.ANCFTireEquilibrium;
production stepping kernels are unchanged.
"""

import argparse
import copy
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton.examples import _positive_float
from newton.examples.ancf import _vehicle_config as vehicle_config
from newton.examples.ancf._vehicle_simulation import resolve_tire_setup
from newton.examples.ancf._vehicle_usd import VehicleUSD, asset_path
from newton.examples.ancf.diffsim._ancf_common import PA_PER_PSI
from newton.examples.ancf.diffsim._tire_calibration import fit_stiffness, read_measurements
from newton.examples.ancf.example_ancf_rigid_mujoco_tires import Example as TireRig
from newton.solvers import ANCFTireEquilibrium, load_ancf_tire_usd

_GROUND_NAMES = ("Telemetry flat ground", "Firmer flat ground")
_GROUND_SCALES = (1.0, 2.0)
_DEMO_PRESSURE_PSI = (0.25, 8.0)


def _telemetry_setup(path: Path):
    """Resolve physical inputs with the same precedence as VehicleSimulation."""
    config = vehicle_config.load_telemetry_preset(path)
    defaults = vars(vehicle_config.create_parser().parse_args([]))
    options = {**defaults, **{k.replace("-", "_"): v for k, v in config["args"].items() if v is not None}}
    vehicle = VehicleUSD(asset_path(options["vehicle_asset"]))
    if vehicle.kind != "rigid_hull" or vehicle.steering != "skid":
        raise ValueError("Tire preparation currently supports rigid-hull skid-steered vehicles")
    options["tire_asset"] = options["tire_asset"] or vehicle.default_tire_asset
    mesh, meta = load_ancf_tire_usd(asset_path(options["tire_asset"]), device="cpu")
    setup = resolve_tire_setup(options, meta, mesh.elem_h.numpy(), vehicle)
    shells = options["shell_tires"]
    if isinstance(shells, str):
        shells = json.loads(shells)
    # VehicleSimulation uses shared material for all four tires; the first entry's other
    # keys (rig-only, e.g. "position") pass through to the rig.
    material = {
        **(shells[0] if shells else {}),
        "E": setup.e_tire,
        "nu": setup.nu_tire,
        "rho": setup.rho_tire,
        "alpha-damp": setup.alpha_d,
        "thickness": setup.thickness,
        "pressure": setup.pressure,
    }
    # The optimizer needs a strictly positive lower pressure bound.
    bounds = (max(setup.envelope[2] * setup.envelope[1], 1.0), setup.envelope[3] * setup.envelope[1])
    material["pressure"] = min(max(material["pressure"], bounds[0]), bounds[1])
    if material["pressure"] != setup.requested_pressure:
        print(
            f"[telemetry preset] Nominal pressure {setup.requested_pressure:g} Pa -> {material['pressure']:g} Pa (vehicle limits)."
        )
    options["kn"], options["kd"] = setup.kn, setup.kd
    radius = float((mesh.node_x0.numpy()[:, 1:] ** 2).sum(axis=1).max() ** 0.5)
    # The rig performs the same load screen and accounts for shell weight too.
    options["reference_patch_support"] = options["kn"] * float(
        (0.025 - (mesh.node_x0.numpy()[:, 1] + radius)).clip(min=0.0).sum()
    )
    return config, options, material, vehicle, bounds


@wp.kernel
def _position_height_markers(
    local_points: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    spindle_index: int,
    target_height: float,
    actual_points: wp.array[wp.vec3],
    target_points: wp.array[wp.vec3],
):
    i = wp.tid()
    pose = body_q[spindle_index]
    height = wp.transform_get_translation(pose)[2]
    actual_points[i] = wp.transform_point(pose, local_points[i])
    # Separate the ghost surface to avoid depth flicker when heights coincide.
    target_points[i] = wp.transform_point(pose, 1.04 * local_points[i]) + wp.vec3(0.0, 0.0, target_height - height)


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


class Example:
    """Train pressure with an equilibrium gradient while the coupled rig advances."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self._config_path = args.telemetry_config.resolve()
        self._config, options, self._material, vehicle, self._pressure_bounds = _telemetry_setup(self._config_path)
        self._measurements = read_measurements(args.calibration_csv) if args.calibration_csv else []
        self._telemetry_pressure_bounds = self._pressure_bounds
        self._pressure_range = args.pressure_range or (
            "telemetry" if self._measurements or args.export_config else "demo"
        )
        if self._pressure_range == "demo":
            if self._measurements or args.export_config:
                raise ValueError("CSV fitting and replay export require --pressure-range telemetry")
            self._pressure_bounds = tuple(p * PA_PER_PSI for p in _DEMO_PRESSURE_PSI)
            self._material["pressure"] = self._pressure_bounds[0]
        self._ground_index = ("telemetry", "firm").index(args.ground)
        self._pending_ground = None
        self._contact_defaults = (options["kn"], options["kd"])
        self._reference_profiles = None
        self._range_samples = None
        if self._measurements and self._ground_index != 0:
            raise ValueError("CSV fitting uses the source telemetry contact settings; omit --ground")
        if self._measurements and not args.train:
            raise ValueError("--calibration-csv requires training")
        if self._measurements and (args.load is not None or args.target_height is not None):
            raise ValueError("CSV cases supply their own loads and heights; omit --load and --target-height")
        self._measurement_path = args.calibration_csv
        self._data_source = args.data_source if self._measurements else "demonstration"
        self._fit_report = None
        self._fit_done = False
        self._export_path = args.export_config
        self._exported = False
        if self._export_path and self._export_path.resolve() == self._config_path:
            raise ValueError("Export to a new config; keep the source telemetry preset for comparison")
        self._automatic_target = args.target_height is None and not self._measurements
        self.target_height = args.target_height
        self.learning = args.train
        self._training_requested = args.train
        self.training_status = "Settling" if self.learning else "Manual"
        self.training_iterations = 0
        self._next_update_frame = 120
        self._equilibrium = None
        self._equilibrium_result = None
        self._equilibrium_initial = None
        self._height_bounds = None
        self._target_outside_range = False
        self._height_tolerance = 2.5e-4
        self.max_penetration = 0.0
        self._initial_height_error = None
        self._error_history = deque(maxlen=300)
        self._run_start_pressure_pa = None
        self._pressure_history = deque(maxlen=300)

        rig_args = TireRig.create_parser().parse_args([])
        rig_args.vehicle_asset = options["vehicle_asset"]
        rig_args.tire_asset = options["tire_asset"]
        rig_args.n_envs = 1
        rig_args.substeps = 10
        rig_args.nr_iters = 2
        # The softer tire below 1 PSI needs a more accurate linear solve.
        rig_args.pcg_iters = 20 if self._pressure_range == "demo" else 10
        rig_args.rpm = 0.0
        rig_args.drop_clearance = 0.0
        # The full vehicle has no additional rig-only mass-proportional damping.
        rig_args.alpha_m_damp = 0.0
        self._load_source = (
            "explicit additional spindle load" if args.load is not None else "asset estimate; equal chassis share"
        )
        rig_args.f_load = (
            args.load
            if args.load is not None
            else (vehicle.rigid_corner_mass - vehicle.wheel_mass) * vehicle_config.GRAVITY
        )
        rig_args.kn, rig_args.kd, rig_args.mu = options["kn"], options["kd"], options["mu"]
        rig_args.kn *= _GROUND_SCALES[self._ground_index]
        rig_args.kd *= _GROUND_SCALES[self._ground_index]
        rig_args.thickness_gp = options["thickness_gp"]
        rig_args.build_pressure = (
            options["build_pressure"] if args.build_pressure_psi is None else args.build_pressure_psi * PA_PER_PSI
        )
        rig_args.diag_period = 60
        if args.pressure_psi is not None:
            self._material["pressure"] = args.pressure_psi * PA_PER_PSI
        self._telemetry_pressure = self._material["pressure"]
        rig_args.shell_tires = [dict(self._material)]
        if self._measurements:
            for row in self._measurements:
                if not self._pressure_bounds[0] <= row.nominal_pressure_pa <= self._pressure_bounds[1]:
                    raise ValueError("Measurement pressure is outside this vehicle's inflation limits")
                if (
                    row.additional_load_n + vehicle.wheel_mass * vehicle_config.GRAVITY
                    > options["reference_patch_support"]
                ):
                    raise ValueError("Measurement load exceeds this rig's supported contact-load limit")
            first = next(row for row in self._measurements if row.split == "train")
            rig_args.f_load = first.additional_load_n
            rig_args.shell_tires[0]["pressure"] = first.nominal_pressure_pa
            self.target_height = first.axle_height_m
            self._load_source = "calibration CSV; additional load excludes spindle weight"
        if not self._pressure_bounds[0] <= self._telemetry_pressure <= self._pressure_bounds[1]:
            raise ValueError("--pressure-psi is outside the selected pressure limits")
        if not math.isfinite(rig_args.build_pressure) or rig_args.build_pressure < 0.0:
            raise ValueError("Build pressure must be finite and nonnegative")
        self._rig_args = rig_args
        self.rig = TireRig(viewer, rig_args)
        # The existing rig clamps excessive load; never silently fit a different experiment.
        if self.rig._f_load != rig_args.f_load:
            raise ValueError(f"--load exceeds this rig's supported limit of {self.rig._f_load_safe_max:.1f} N")
        if any(row.additional_load_n > self.rig._f_load_safe_max for row in self._measurements):
            raise ValueError("A CSV load exceeds the reference contact-patch load screen")
        self.model = self.rig.model
        self._initial_pressure_pa = self.pressure_pa
        self._update_observation()
        if self.target_height is None:
            self.target_height = self.height

        self._configure_tire_view()
        print(
            f"[telemetry preset] {self._data_source}; additional load={rig_args.f_load:.2f} N ({self._load_source}); "
            f"build pressure={rig_args.build_pressure:g} Pa; {self._pressure_range} limits={self._pressure_bounds} Pa."
        )

    @property
    def pressure_pa(self) -> float:
        """Applied nominal cavity pressure [Pa], ramped by the existing rig."""
        return self.rig._pressure_currents[0]

    @property
    def height_error(self) -> float:
        """Current spindle height minus requested height [m]."""
        return self.height - self.target_height

    @property
    def loss(self) -> float:
        """Instantaneous height loss [m^2], including edits made while paused."""
        return 0.5 * self.height_error**2

    def _configure_tire_view(self) -> None:
        rest = self.rig.ancf_model.node_x0.numpy()
        n_circ = self.rig._n_bead_per_ring
        radii = np.linalg.norm(rest[:, 1:], axis=1).reshape(-1, n_circ).mean(axis=1)
        row = int(np.argmax(radii))
        self._profile_indices = np.arange(row * n_circ, (row + 1) * n_circ)
        marker = newton.Mesh.create_cylinder(0.014, 0.38, up_axis=newton.Axis.Y, compute_inertia=False)
        self._marker_local_points = wp.array(marker.vertices, dtype=wp.vec3, device=self.model.device)
        self._marker_indices = wp.array(marker.indices, dtype=wp.int32, device=self.model.device)
        self._actual_marker_points = wp.empty_like(self._marker_local_points)
        self._target_marker_points = wp.empty_like(self._marker_local_points)
        if self.viewer is not None:
            self.viewer.set_camera(pos=(0.20, -1.10, 0.40), pitch=-7.0, yaw=108.0)
            renderer = getattr(self.viewer, "renderer", None)
            if renderer is not None:
                renderer.draw_edges = False

    def _tread_profile(self, nodes: np.ndarray, height: float) -> np.ndarray:
        # Align at the axle to remove rigid lift: this exposes tire deformation.
        profile = nodes[self._profile_indices][:, [2, 1]].copy()
        profile[:, 1] += self.rig._r_outer - height
        return profile

    def _select_ground(self, index: int) -> None:
        if index == self._ground_index:
            return
        self._rig_args.kn, self._rig_args.kd = (v * _GROUND_SCALES[index] for v in self._contact_defaults)
        self._rig_args.shell_tires[0]["pressure"] = self.pressure_pa
        # Contact constants are captured in CUDA graphs: rebuild both solvers.
        self.rig = TireRig(self.viewer, self._rig_args)
        self.model = self.rig.model
        self._ground_index = index
        self._equilibrium = self._equilibrium_result = self._equilibrium_initial = None
        self._height_bounds = self._reference_profiles = None
        self._range_samples = None
        automatic_target = self._automatic_target
        self.set_target_height(self.target_height)
        self._automatic_target = automatic_target
        self._next_update_frame = 120
        self.training_status = "Settling"
        self._update_observation()
        self._configure_tire_view()
        # set_model clears the viewer's example callbacks during the rebuild.
        if self.viewer is not None:
            self.viewer.register_ui_callback(self.gui, position="side")

    def _manual_pressure(self, pressure_pa: float) -> None:
        self.set_pressure(pressure_pa)
        self._equilibrium_initial = None
        self.learning = False
        self.training_status = "Manual"
        self._exported = False
        self._initial_height_error = None
        self._run_start_pressure_pa = None
        self._error_history.clear()
        self._pressure_history.clear()

    def set_target_height(self, height: float) -> None:
        """Set target axle height above ground [m] and start pressure learning."""
        if not math.isfinite(height) or height <= 0.0:
            raise ValueError("Target height must be finite and positive")
        self.target_height = height
        self._automatic_target = False
        self._exported = False
        self.training_iterations = 0
        self.learning = True
        self._initial_height_error = None
        self._error_history = deque(maxlen=300)
        self._run_start_pressure_pa = None
        self._pressure_history = deque(maxlen=300)
        self._target_outside_range = not self._target_is_reachable()
        self.training_status = "Target changed"
        if self._target_outside_range:
            self.training_status = "Target outside reachable height range"
        self._next_update_frame = self.rig._frame + 1

    def _restart_learning(self) -> None:
        automatic_target = self._automatic_target
        self.set_target_height(self.target_height)
        self._automatic_target = automatic_target
        start = self._initial_pressure_pa
        if self._height_bounds is not None:
            # Repeated runs begin at the opposite end, even for a low target.
            start = self._pressure_bounds[int(self.target_height < sum(self._height_bounds) / 2)]
        self.set_pressure(start)
        self._equilibrium_result = None
        self._equilibrium_initial = None
        # Let the physical tire return to its starting pressure before measuring
        # the initial error and optimizing again. Keep the chosen target.
        self._next_update_frame = self.rig._frame + 120
        if not self._target_outside_range:
            self.training_status = "Settling"

    def _target_is_reachable(self) -> bool:
        if self._height_bounds is None:
            return True
        lower, upper = self._height_bounds
        # ImGui stores slider values as float32, including the endpoints.
        return lower - 1e-7 <= self.target_height <= upper + 1e-7

    def set_pressure(self, pressure_pa: float) -> None:
        """Request nominal cavity pressure [Pa] without changing build pressure."""
        if not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
            raise ValueError("pressure_pa must be finite and positive")
        if (
            hasattr(self, "_pressure_bounds")
            and not self._pressure_bounds[0] <= pressure_pa <= self._pressure_bounds[1]
        ):
            raise ValueError("Nominal pressure must be within the selected pressure limits")
        self.rig._pressure_targets[0] = pressure_pa

    def _update_observation(self) -> None:
        # Newton rendering state is Z-up, unlike the ANCF solver's Y-up state.
        spindle = self.rig.state_0.body_q.numpy()[self.rig._spindle_newton_idx]
        self.height = float(spindle[2])
        nodes = self.rig.ancf_solver.node_x.numpy()
        self._observed_nodes = nodes
        if not math.isfinite(self.height) or not np.isfinite(nodes).all():
            raise RuntimeError("Non-finite tire preparation state")
        self.penetration = max(0.0, float(self.rig.ancf_solver.ground_z - nodes[:, 1].min()))
        self.max_penetration = max(self.max_penetration, self.penetration)
        if self.penetration > 0.025:
            raise RuntimeError("Tire penetration exceeded 25 mm; the preparation experiment is unsupported")

    def step(self) -> None:
        if self._pending_ground is not None:
            self._select_ground(self._pending_ground)
            self._pending_ground = None
        was_converged = self.training_status == "Converged"
        self.rig.step()
        self._update_observation()
        if (
            not self.learning
            and not self._measurements
            and self._height_bounds is None
            and self.rig._frame >= self._next_update_frame
            and abs(self.pressure_pa - self.rig._pressure_targets[0]) < 1.0
        ):
            # Manual exploration needs the same reference shapes and target range.
            try:
                self._prepare_equilibrium()
            except (RuntimeError, ValueError) as error:
                self.training_status = f"Stopped: {error}"
                self._next_update_frame = self.rig._frame + 60
        if (
            self.learning
            and not self._target_outside_range
            and self.training_status != "Converged"
            and self.rig._frame >= self._next_update_frame
        ):
            self._learn_pressure()
        if self._initial_height_error is not None and not was_converged and self.rig._frame % 6 == 0:
            self._error_history.append(abs(self.height_error) * 1000.0)
            self._pressure_history.append(self.pressure_pa / PA_PER_PSI)
        if self._export_path and not self._exported and (self.training_status == "Converged" or self._fit_done):
            self.export_config(self._export_path)
            self._exported = True
        if self.rig._frame % 60 == 0:
            print(
                f"[tire lift] t={self.rig._t:.2f}s  p={self.pressure_pa:.0f} Pa  "
                f"height={self.height:.5f} m  target={self.target_height:.5f} m  "
                f"loss={self.loss:.6g} m^2  training={self.training_status}  updates={self.training_iterations}"
            )

    def pressure_gradient(self) -> float:
        """Return d(equilibrium height loss)/d(nominal pressure) [m^2/Pa]."""
        if self._equilibrium_result is None:
            raise RuntimeError("No converged equilibrium gradient is available yet")
        return self._equilibrium_result.loss_gradient(self.target_height)

    def _prepare_equilibrium(self):
        if self._equilibrium is None:
            self._equilibrium = ANCFTireEquilibrium(
                self.rig.ancf_solver,
                self.rig._bead_idx_np,
                self.rig._m_rigid,
                self.rig._f_load,
                self.rig._build_pressures[0],
            )
        if self._equilibrium_initial is None:
            self._equilibrium_initial = self._equilibrium.coordinates(
                self.rig.ancf_solver.node_x.numpy(), self.rig.ancf_solver.node_D.numpy(), self.height
            )
        if self._measurements:
            return None
        result = self._equilibrium.solve(self.pressure_pa, self._equilibrium_initial)
        self._equilibrium_initial = result.coordinates
        if self._height_bounds is None:
            checked = []
            # Half-octave increments also check the more compliant low-PSI region.
            lower_pressure, upper_pressure = self._pressure_bounds
            count = max(5, 1 + math.ceil(2 * math.log2(upper_pressure / lower_pressure)))
            pressures = np.geomspace(lower_pressure, upper_pressure, count)
            # Continue outwards from the current solution; an 8 -> 0.25 PSI
            # jump can fail at contact transitions despite valid equilibria.
            for sweep in (pressures[pressures < result.pressure][::-1], pressures[pressures >= result.pressure]):
                coordinates = result.coordinates
                for pressure in sweep:
                    sample = self._equilibrium.solve(float(pressure), coordinates)
                    nodes = self._equilibrium.positions(sample.coordinates)
                    penetration = max(0.0, float(self._equilibrium.ground - nodes[:, 1].min()))
                    if not np.isfinite(nodes).all() or not math.isfinite(sample.height) or penetration >= 0.025:
                        raise RuntimeError("Pressure range exceeds the supported deformation/contact limits")
                    if not math.isfinite(sample.dh_dp) or sample.dh_dp <= 0.0:
                        raise RuntimeError("Pressure range has no positive height sensitivity")
                    checked.append((sample, penetration))
                    coordinates = sample.coordinates
            checked.sort(key=lambda item: item[0].pressure)
            if np.any(np.diff([sample.height for sample, _ in checked]) <= 0.0):
                raise RuntimeError("Pressure range does not have a monotonic height response")
            # Publish limits only after every sample has passed.
            lower, upper = checked[0][0], checked[-1][0]
            self._height_bounds = (lower.height, upper.height)
            self._reference_profiles = tuple(
                self._tread_profile(self._equilibrium.positions(r.coordinates), r.height) for r in (lower, upper)
            )
            self._range_samples = [
                {"pressure_pa": r.pressure, "height_m": r.height, "penetration_m": penetration, "dh_dp": r.dh_dp}
                for r, penetration in checked
            ]
            print(f"[tire lift] Reachable axle height: {self._height_bounds} m on {_GROUND_NAMES[self._ground_index]}.")
        if self._automatic_target:
            lower, upper = self._height_bounds
            fraction = 0.9 if result.height < (lower + upper) / 2 else 0.1
            self.target_height = lower + fraction * (upper - lower)
            self._automatic_target = False
        self._target_outside_range = not self._target_is_reachable()
        return result

    def _learn_pressure(self) -> None:
        self._next_update_frame = self.rig._frame + 60
        if abs(self.pressure_pa - self.rig._pressure_targets[0]) > 1.0:
            self.training_status = "Ramping pressure"
            return
        try:
            result = self._prepare_equilibrium()
            if self._measurements:
                self._learn_stiffness()
                return
            self._equilibrium_result = result
            self._equilibrium_initial = result.coordinates
            self._target_outside_range = not self._target_is_reachable()
            if self._target_outside_range:
                self.training_status = "Target outside reachable height range"
                print(f"[learn] {self.training_status}: choose {self._height_bounds} m")
                return
            if self._initial_height_error is None:
                self._initial_height_error = result.height - self.target_height
                self._run_start_pressure_pa = result.pressure
                self._pressure_history.append(result.pressure / PA_PER_PSI)
            print(
                f"[gradient] p={result.pressure:.1f} Pa  h_eq={result.height:.7f} m  "
                f"dh/dp={result.dh_dp:.7g} m/Pa  dL/dp={self.pressure_gradient():.7g} m^2/Pa  "
                f"residual={result.residual_norm:.3g}"
            )
            if abs(result.height - self.target_height) < 5e-5:
                self.training_status = (
                    "Converged" if abs(self.height_error) < self._height_tolerance else "Settling rim"
                )
                return
            if self.training_iterations >= 20:
                raise RuntimeError("Pressure learning reached its 20-update limit")
            # Keep each genuine gradient update visible: at most 10% of the
            # allowed range per second (0.775 PSI for the wider demonstration).
            lower, upper = self._pressure_bounds
            max_change = 0.1 * (upper - lower)
            step_bounds = (max(lower, result.pressure - max_change), min(upper, result.pressure + max_change))
            candidate = self._equilibrium.pressure_step(result, self.target_height, step_bounds)
            self.set_pressure(candidate.pressure)
            self._equilibrium_initial = candidate.coordinates
            self.training_iterations += 1
            self.training_status = "Ramping pressure"
            print(f"[learn] update={self.training_iterations}  requested={candidate.pressure / PA_PER_PSI:.5f} psi")
        except (RuntimeError, ValueError) as error:
            self.learning = False
            self.training_status = f"Stopped: {error}"
            print(f"[learn] {self.training_status}")

    def _learn_stiffness(self) -> None:
        self._fit_report = fit_stiffness(self._equilibrium, self._equilibrium_initial, self._measurements)
        self._material["E"] *= self._fit_report["stiffness_scale"]
        self._rig_args.shell_tires[0]["E"] = self._material["E"]
        # Rebuild all material-dependent solver caches for the fitted preview.
        self.rig = TireRig(self.viewer, self._rig_args)
        self.model = self.rig.model
        self._equilibrium = self._equilibrium_result = self._equilibrium_initial = None
        self._update_observation()
        self._configure_tire_view()
        if self.viewer is not None:
            self.viewer.register_ui_callback(self.gui, position="side")
        self.training_iterations = self._fit_report["updates"]
        self.learning = False
        self._fit_done = True
        self.training_status = "Static stiffness fitted"
        print(f"[telemetry preset] E={self._material['E']:g} Pa; height RMS [m]: {self._fit_report['height_rms_m']}")

    def export_config(self, path: Path) -> None:
        """Write a telemetry replay config and provenance report after a successful static fit."""
        if not (self.training_status == "Converged" or self._fit_done):
            raise RuntimeError("Export requires a converged pressure demonstration or stiffness fit")
        if self._pressure_range != "telemetry":
            raise RuntimeError(
                "Replay export requires --pressure-range telemetry; demo pressures exceed the telemetry preset limits"
            )
        if path.resolve() == self._config_path:
            raise ValueError("Export to a new config; keep the source telemetry preset for comparison")
        config = copy.deepcopy(self._config)
        material = dict(self._material)
        material["pressure"] = self._telemetry_pressure if self._measurements else self.pressure_pa
        config["args"].update(
            {
                "vehicle-asset": self._rig_args.vehicle_asset,
                "tire-asset": self._rig_args.tire_asset,
                "shell-tires": [material],
                "build-pressure": self._rig_args.build_pressure,
                "kn": self._rig_args.kn,
                "kd": self._rig_args.kd,
                "mu": self._rig_args.mu,
                "thickness-gp": self._rig_args.thickness_gp,
                "replay": True,
            }
        )
        config["description"] = (
            "Static tire preparation for recorded-command replay; vehicle calibration remains unvalidated."
        )
        inputs = {"telemetry_config": self._config_path}
        inputs.update({key: Path(asset_path(config["args"][key])) for key in ("vehicle-asset", "tire-asset")})
        if self._measurement_path:
            inputs["measurements"] = self._measurement_path
        report = {
            "format": "newton.tire_preparation/1",
            "data_source": self._data_source,
            "vehicle_calibration_validated": False,
            "load_source": self._load_source,
            "ground": _GROUND_NAMES[self._ground_index],
            "contact_stiffness_n_per_m": self._rig_args.kn,
            "additional_load_n": self.rig._f_load,
            "spindle_mass_kg": self.rig._m_rigid,
            "build_pressure_pa": self._rig_args.build_pressure,
            "nominal_pressure_pa": material["pressure"],
            "pressure_bounds_pa": self._pressure_bounds,
            "telemetry_pressure_bounds_pa": self._telemetry_pressure_bounds,
            "pressure_range": self._pressure_range,
            "checked_pressure_samples": self._range_samples,
            "young_modulus_pa": material["E"],
            "target_height_m": self.target_height,
            "preview_height_m": self.height,
            "preview_max_penetration_m": self.max_penetration,
            "preview_substeps": self.rig._substeps,
            "preview_newton_iterations": self._rig_args.nr_iters,
            "preview_pcg_iterations": self._rig_args.pcg_iters,
            "static_fit": self._fit_report,
            "inputs": {
                key: {"path": str(value), "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}
                for key, value in inputs.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        report_path = path.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        config["preparation_report"] = report_path.name
        path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        print(f"[telemetry preset] Exported {path} and {report_path}; validate with recorded-command replay.")

    def gui(self, ui) -> None:
        if getattr(self, "_ui_frames", 0) < 2:
            # Keep the standard viewer tools available, initially folded away.
            for label in ("Model Information", "Visualization", "Rendering Options", "Controls"):
                ui.get_state_storage().set_int(ui.get_id(label), 0)
            self._ui_frames = getattr(self, "_ui_frames", 0) + 1
        if self._measurements:
            ui.text("Fit tire stiffness from measurements")
            ui.text(f"Status: {self.training_status}")
            ui.text(f"Measured axle height: {self.target_height * 1000:.2f} mm")
            ui.text(f"Simulated axle height: {self.height * 1000:.2f} mm")
            if self._fit_report:
                ui.text(f"Fitted stiffness (E): {self._material['E'] / 1e6:.3f} MPa")
                for split, rms in self._fit_report["height_rms_m"].items():
                    ui.text(f"{split} height error (RMS): {rms * 1000:.3f} mm")
            ui.text_wrapped("Calibration cases and replay export are configured on the command line.")
        else:
            self._gui_pressure_demo(ui)
        if self.viewer is not None and self.viewer.is_paused():
            ui.text_wrapped("Simulation paused. Uncheck Pause above to continue.")
        if self._export_path:
            ui.text_wrapped(f"Replay config: {self._export_path} ({'saved' if self._exported else 'pending'})")
        if ui.collapsing_header("Advanced"):
            self._gui_advanced(ui)

    def _gui_pressure_demo(self, ui) -> None:
        changed, ground = ui.combo("Ground", self._ground_index, list(_GROUND_NAMES))
        if changed:
            self._pending_ground = ground
        ui.text("Experimental pressure range" if self._pressure_range == "demo" else "Telemetry pressure range")
        ui.push_font(None, 26.0)
        ui.text(f"{self.pressure_pa / PA_PER_PSI:.2f} PSI")
        ui.pop_font()
        lower, upper = (p / PA_PER_PSI for p in self._pressure_bounds)
        ui.set_next_item_width(ui.get_content_region_avail().x)
        changed, pressure = ui.slider_float(
            "##pressure", self.rig._pressure_targets[0] / PA_PER_PSI, lower, upper, "%.2f PSI"
        )
        if changed:
            self._manual_pressure(float(np.clip(pressure * PA_PER_PSI, *self._pressure_bounds)))
        if ui.button(f"Low: {lower:.2f} PSI"):
            self._manual_pressure(self._pressure_bounds[0])
        ui.same_line()
        if ui.button(f"High: {upper:.2f} PSI"):
            self._manual_pressure(self._pressure_bounds[1])
        self._gui_tread_detail(ui)
        ui.separator()
        ui.text("Learn pressure for a ride height")
        if self._height_bounds is not None:
            ui.set_next_item_width(ui.get_content_region_avail().x)
            changed, height_mm = ui.slider_float(
                "##target_height",
                self.target_height * 1000,
                self._height_bounds[0] * 1000,
                self._height_bounds[1] * 1000,
                "Target: %.2f mm",
            )
            if changed:
                self.set_target_height(height_mm / 1000)
        else:
            ui.text("Preparing height range...")
        if ui.button("Learn pressure"):
            self._restart_learning()
        ui.same_line()
        status = "Height reached" if self.training_status == "Converged" else self.training_status
        ui.text_wrapped(status)
        ui.text(f"Actual axle: {self.height * 1000:.2f} mm")
        ui.text_colored(ui.ImVec4(0.75, 0.75, 0.75, 1.0), f"Gray target: {self.target_height * 1000:.2f} mm")
        if self._target_outside_range:
            ui.text_wrapped("This ground cannot reach that height within the PSI limits. Move the target slider.")

    def _gui_tread_detail(self, ui) -> None:
        ui.text("Tread detail (mm; vertical zoom)")
        ui.text_colored(ui.ImVec4(0.2, 0.65, 1.0, 1.0), "Low PSI")
        ui.same_line()
        ui.text_colored(ui.ImVec4(1.0, 0.6, 0.15, 1.0), "High PSI")
        ui.same_line()
        ui.text("Live")
        origin = ui.get_cursor_screen_pos()
        width = max(120.0, ui.get_content_region_avail().x)
        height = 155.0
        left, right = origin.x + 30, origin.x + width - 4
        top, bottom = origin.y + 5, origin.y + height - 15
        draw = ui.get_window_draw_list()
        ink = ui.get_color_u32(ui.ImVec4(0.65, 0.68, 0.72, 1.0))

        # The same axes are used for every pressure and ground setting.
        # Y is radial deformation measured from the reference tire's bottom.
        def point(x, y):
            return ui.ImVec2(left + (x + 0.1) / 0.2 * (right - left), bottom - y / 0.04 * (bottom - top))

        for mm in (0, 10, 20, 30, 40):
            y = point(0, mm / 1000).y
            draw.add_text(ui.ImVec2(origin.x, y - 7), ink, str(mm))
            draw.add_line(ui.ImVec2(left, y), ui.ImVec2(right, y), ink, 0.5)
        profiles = []
        if self._reference_profiles is not None:
            profiles.extend(zip(self._reference_profiles, ((0.2, 0.65, 1.0, 1.0), (1.0, 0.6, 0.15, 1.0)), strict=True))
        profiles.append((self._tread_profile(self._observed_nodes, self.height), (1.0, 1.0, 1.0, 1.0)))
        draw.push_clip_rect(ui.ImVec2(left, top), ui.ImVec2(right, bottom), True)
        for profile, rgba in profiles:
            color = ui.get_color_u32(ui.ImVec4(*rgba))
            for a, b in zip(profile, np.roll(profile, -1, axis=0), strict=True):
                draw.add_line(point(*a), point(*b), color, 2.0)
        draw.pop_clip_rect()
        ui.dummy(ui.ImVec2(width, height))
        squash = self.rig._r_outer - self.height + float(self._observed_nodes[:, 1].min())
        ui.text(f"Tire flattening: {squash * 1000:.1f} mm")

    def _gui_advanced(self, ui) -> None:
        ui.text(f"Live height loss: {self.loss:.6g} m^2")
        ui.text(f"Learning updates: {self.training_iterations}")
        if self._equilibrium_result is not None:
            result = self._equilibrium_result
            ui.text(f"dh/dp: {result.dh_dp * PA_PER_PSI:.6g} m/psi")
            ui.text(f"Equilibrium residual: {result.residual_norm:.2g}")
        ui.text(f"Additional load: {self.rig._f_load:.1f} N")
        ui.text(f"Contact stiffness: {self._rig_args.kn:.1f} N/m per node")
        ui.text_wrapped("Ground changes flat support compliance. Uneven terrain and soil are not modeled.")
        ui.text(f"Fixed build pressure: {self.rig._build_pressures[0]:.0f} Pa")
        ui.text_wrapped(
            "PSI is the model's nominal cavity pressure. This learns one pressure for a static height goal."
        )
        ui.text_wrapped(f"Telemetry preparation: {self._data_source}. Vehicle motion not yet validated.")

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.rig._t)
        self.viewer.log_state(self.rig.state_0)
        # Use the standard smooth tire material/normals, as in examples 01/02.
        name = self.viewer._qualify("/model/triangles") if hasattr(self.viewer, "_qualify") else "/model/triangles"
        tire = getattr(self.viewer, "objects", {}).get(name)
        if tire is not None:
            tire.draw_edge = False
        self._render_height_markers()
        self.viewer.end_frame()

    def _render_height_markers(self) -> None:
        wp.launch(
            _position_height_markers,
            dim=len(self._marker_local_points),
            inputs=[
                self._marker_local_points,
                self.rig.state_0.body_q,
                self.rig._spindle_newton_idx,
                self.target_height,
            ],
            outputs=[self._actual_marker_points, self._target_marker_points],
            device=self._marker_local_points.device,
        )
        for name, points, color, alpha in (
            ("actual_axle", self._actual_marker_points, (0.35, 0.35, 0.75), 1.0),
            ("target_axle", self._target_marker_points, (0.65, 0.65, 0.65), 0.45),
        ):
            self.viewer.log_mesh(
                name, points, self._marker_indices, color=color, backface_culling=False, roughness=0.5, metallic=0.0
            )
            key = self.viewer._qualify(name) if hasattr(self.viewer, "_qualify") else name
            obj = getattr(self.viewer, "objects", {}).get(key)
            if obj is not None:
                obj.alpha = alpha
                obj.cast_shadow = alpha == 1.0
                obj.draw_edge = False

    def test_final(self) -> None:
        from newton.tests.ancf_diffsim_checks import check_tire_lift_result  # noqa: PLC0415

        check_tire_lift_result(self)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=960)
        parser.description = "Learn tire pressure from a target axle height; optional static calibration for vehicle telemetry (CUDA required)."
        parser.add_argument(
            "--telemetry-config",
            type=Path,
            default=vehicle_config.TELEMETRY_PRESET,
            help="Source vehicle_telemetry JSON preset.",
        )
        parser.add_argument(
            "--pressure-range",
            choices=("demo", "telemetry"),
            default=None,
            help="Default: experimental 0.25–8 nominal PSI; CSV fitting/export default to the telemetry preset limits (Warthog: 2–4 PSI).",
        )
        parser.add_argument(
            "--ground",
            choices=("telemetry", "firm"),
            default="telemetry",
            help="Flat contact: the telemetry preset settings or twice the stiffness/damping; not a soil model.",
        )
        parser.add_argument(
            "--calibration-csv",
            type=Path,
            help="Static measurements (nominal pressure, additional load, axle height, uncertainty, "
            "train/validation split); fits a shared stiffness scale.",
        )
        parser.add_argument(
            "--data-source",
            choices=("measured", "synthetic"),
            default="measured",
            help="Provenance of calibration CSV rows.",
        )
        parser.add_argument(
            "--export-config",
            type=Path,
            help="Save a new telemetry replay JSON and provenance report after convergence.",
        )
        parser.add_argument(
            "--train",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Learn demonstration pressure or fit stiffness from CSV; --no-train enables manual preview.",
        )
        parser.add_argument(
            "--target-height",
            type=_positive_float,
            default=None,
            help="Demonstration axle height [m]; default: synthetic target inside the checked pressure range.",
        )
        parser.add_argument(
            "--pressure-psi",
            type=_positive_float,
            default=None,
            help="Requested nominal cavity pressure [psi]; not calibrated gauge pressure.",
        )
        parser.add_argument(
            "--build-pressure-psi",
            type=_nonnegative_float,
            default=None,
            help="Fixed reference/build pressure [psi]; independent of the requested pressure.",
        )
        parser.add_argument(
            "--load",
            type=_nonnegative_float,
            default=None,
            help="Additional downward spindle load [N]; default: asset chassis share per wheel.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
