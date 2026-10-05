# ANCF differentiable-simulation examples

Three learning experiments built on the ANCF shell tire and the vehicle
examples one directory up (`newton/examples/ancf/`). Each one fixes a physical
setup taken from those examples, measures something a real vehicle would
expose, and learns one quantity from it with analytic or Warp gradients.
Together they are the preparation steps for calibrating the simulated Warthog
against the recorded drive used by `vehicle_telemetry` (example 07).

| Example | Learns | From | Gradient | Docker config |
|---|---|---|---|---|
| `diffsim_ancf_tire_lift` | tire pressure | static axle height of one loaded tire | analytic implicit (equilibrium) | `08_diffsim_ancf_tire_lift.json` |
| `diffsim_ancf_skid_steer` | speed and yaw-rate response gains | straight and turning manoeuvres of the full vehicle | Warp autodiff through a response model | `09_diffsim_ancf_skid_steer.json` |
| `diffsim_ancf_tire_traction` | ground friction coefficient | travel and speed of a rolling tire under a spin-and-brake command | analytic implicit through dynamic backward-Euler steps | `10_diffsim_ancf_tire_traction.json` |

All three read their tire, material, contact and pressure limits from the
telemetry preset `docker/config/07_vehicle_telemetry.json` through
`_vehicle_config.load_telemetry_preset()`, so the physics matches example 07.

## Running

From the repository root, inside the Docker image:

```bash
python -m newton.examples diffsim_ancf_tire_lift        # CUDA
python -m newton.examples diffsim_ancf_skid_steer       # CUDA
OPENBLAS_NUM_THREADS=1 python -m newton.examples diffsim_ancf_tire_traction   # CPU, float64
```

or through the launcher, which applies the Docker config and prompts for any
open choices:

```bash
./docker/run-examples.sh 08_diffsim_ancf_tire_lift
./docker/run-examples.sh 09_diffsim_ancf_skid_steer
./docker/run-examples.sh 10_diffsim_ancf_tire_traction
```

`--viewer null --test` runs an example headless and applies the acceptance
check from `newton/tests/ancf_diffsim_checks.py`. `docker/README.md` lists
the per-example options (manual pressure control, pressure ranges, report and
replay export, calibration CSV input).

## The examples

### `example_diffsim_ancf_tire_lift.py` — pressure from ride height

Reuses the single-wheel rig of example 02 (`ancf_rigid_mujoco_tires`): one
simplified Warthog ANCF tire on a MuJoCo spindle that is free to move
vertically, plus an additional downward load (by default the asset's estimated
chassis share per wheel). Learning minimises the static height error
`0.5 (h - h_target)^2` using `ANCFTireEquilibrium`, which gives analytic
pressure sensitivities through rim balance, EAS, follower pressure and contact.
MuJoCo keeps advancing the rim; its transient is not differentiated. The UI
exposes a pressure slider, Low/High PSI presets, a target-height slider and a
"Learn pressure" button; `--no-train` gives manual pressure control.

`--calibration-csv` fits measured axle heights (`_tire_calibration.py` reads the
CSV and performs the scalar modulus fit) and `--export-config` writes a replay
configuration for example 07 with provenance.

### `example_diffsim_ancf_skid_steer.py` — skid-steer response

Uses the full vehicle of example 03 (`vehicle_ancf_tires`) on flat ground with
independent left/right wheel motors; steering comes only from tire forces. The
example drives straight, gentle and stronger left/right manoeuvres, measures
body speed and yaw rate, and fits `SkidResponse` (`_skid_steer_calibration.py`):
a speed gain and an odd cubic yaw response differentiated with Warp. Gray is the
requested trajectory, orange the initial drive, purple the drive with the fitted
response. Gradients pass through the response model, not through MuJoCo or the
ANCF integration; the fitted gains absorb tire slip and drive response and are
not material constants. `--report PATH` saves the observations and the
before/after validation.

### `example_diffsim_ancf_tire_traction.py` — friction from a rolling tire

A gray reference tire and the white simulated tire receive the same
spin-and-brake command; the simulated tire's spindle translates freely, so its
travel and speed depend on traction. `ANCFTireTraction` integrates shell and
spindle together with a dense float64 backward-Euler solver and propagates the
analytic friction sensitivity through every converged step. One friction
coefficient is learned from the two histories and validated on a different
motor command. CPU only; it prioritises converged derivatives over speed.

## Shared helpers

- `_ancf_common.py` — `PA_PER_PSI` and the reference / before / learned colours.
- `_tire_calibration.py` — `Measurement`, `read_measurements()` and
  `fit_stiffness()` for the lift example's CSV path.
- `_skid_steer_calibration.py` — the `SkidResponse` model and its Warp loss kernel.

## Tests

```bash
bash docker/test-ancf-examples.sh -k test_diffsim_ancf
```

runs `newton/tests/test_diffsim_ancf_{tire_lift,skid_steer,tire_traction}.py`
and `test_ancf_differentiation.py`. The physical learning runs are opt-in:
`ANCF_SKID_CALIBRATION=1` for the skid-steer drives and `ANCF_TRACTION_TESTS=1`
for the traction dynamics.

## Scope

These are synthetic calibration exercises on flat, rigid ground with penalty
contact. They do not identify terrain materials, train a reusable controller or
replace calibration against the recorded drive; that is the next step, in
`vehicle_telemetry`.
