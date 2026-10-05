# Newton Docker Setup

Docker configuration to build and run Newton standalone (without Isaac Lab).

Adapted from the `.devcontainer` setup in the legacy
`Newton/isaac-lab/newton` project, but targeting this repo's layout.

## Layout

This `docker/` folder lives at the Newton repo root
(`third_party/newton/docker/`), parallel to `pyproject.toml` / `uv.lock`.
The build context is the parent directory (the Newton repo root) so the
Dockerfiles' `COPY pyproject.toml uv.lock ./` and friends Just Work.

## Quick Start

### Build

```bash
# Auto-detect current platform
./docker/build-docker.sh

# Or pick a platform explicitly
./docker/build-docker.sh x86     # x86_64 / amd64
./docker/build-docker.sh arm64   # aarch64 (Jetson / Grace)
./docker/build-docker.sh both    # multi-arch with buildx
```

Produces an image tagged `newton:latest` (and `newton:amd64` or
`newton:arm64`).

### Compare Warp versions on ARM64

The ARM64 build defaults to the SHA-256-verified NVIDIA Warp 1.17.0 CUDA 13
wheel, with the local automatic CUDA launch-bounds patch. The CUDA 13 base image
is retained; Warp uses the compiler shipped in its wheel. Select an alternative explicitly when comparing versions:

```bash
./docker/build-docker.sh arm64                       # Warp 1.17.0, CUDA 13
WARP_LOCAL_PATCHES=0 ./docker/build-docker.sh arm64   # Unmodified Warp 1.17/CUDA 13
WARP_VERSION=locked ./docker/build-docker.sh arm64   # uv.lock: Warp 1.15.0
```

The CUDA 13 wheel requires an R580 or newer NVIDIA driver and compute
capability 7.5 or newer. The GB10 meets these requirements. The Docker build
prints the installed versions and import paths.
The override is applied after dependency synchronization; running `uv sync`
inside the resulting container restores the lockfile version. Use `python` or
`uv run --no-sync` to retain the selected version. Each selection uses its own
kernel cache beneath the existing cache mount.

The lockfile selects the released MuJoCo **3.10.0** wheels from PyPI and keeps
the locally patched `third_party/mujoco_warp` **3.10.0.2** source. Released wheels
avoid depending on retired nightly downloads from `py.mujoco.org`. The example
launcher bind-mounts the MuJoCo Warp source from the host, so a MuJoCo Warp
upgrade must update that source and its compatible native MuJoCo dependency
together. Docker uses `uv sync --locked` to check that project metadata and the
lockfile agree. It installs the current `dev` extra; the obsolete `cudss` extra
is no longer requested.

The local Warp checkout is `third_party/warp`, pinned to upstream `v1.17.0`
(`f4c57f26f1e3936a89afd283e39fcabf6d548dc7`). Its compiler supplies CUDA
launch bounds for the actual compiled block size, preserving explicit register
limits, explicit launch bounds, and external entry points. This changes compiler
register allocation without reducing solver iterations or tire reaction torque.
A module can opt out with `wp.set_module_options({"cuda_auto_launch_bounds": False})`.

Docker applies `docker/patches/warp-1.17-auto-launch-bounds.patch` to the pinned
CUDA 13 wheel after installation. The locked-version alternative
remains unmodified. The local checkout and its native build products are excluded
from the Docker build context. Changing the checkout does not update an existing
image: export the two-file patch and rebuild after testing:

```bash
git -C third_party/warp diff -- warp/_src/codegen.py warp/_src/context.py \
  > docker/patches/warp-1.17-auto-launch-bounds.patch
./docker/build-docker.sh arm64
```

For direct library development, point `PYTHONPATH` at the checkout in an
environment with Newton's dependencies; build Warp's native libraries first with
`uv run build_lib.py --quick --cuda-path /usr/local/cuda` from that checkout.
Use a separate `WARP_CACHE_PATH` for source-built and wheel-based comparisons.
The example launcher uses the image's patched wheel by default.

In the earlier GB10 compiler comparison, three paired example 03 moving-Warthog runs with simplified tires and
the GL viewer measured 20.1 FPS median with stock Warp 1.17/CUDA 13 and 21.6 FPS
with the patch. The solver retained 10 substeps, 2 Newton iterations, 10 PCG
iterations, and full tire reaction torque. This patch alone does not reach 30 FPS.

### Run an example

```bash
# Interactive menu
./docker/run-examples.sh

# By name (flat example name; see newton/examples/*/example_*.py)
./docker/run-examples.sh basic_pendulum
./docker/run-examples.sh robot_cartpole --num-frames 500

# By number (from the menu order)
./docker/run-examples.sh 1

# Force a viewer
./docker/run-examples.sh --gl basic_pendulum      # default
./docker/run-examples.sh --usd basic_pendulum     # write output.usd
./docker/run-examples.sh --null robot_cartpole    # headless
```

Examples are invoked inside the container via
`python -m newton.examples <name>`. The live Python package is bind-mounted
from the host at `third_party/newton/newton` → `/workspace/newton/newton`,
so edits to Python sources take effect without rebuilding the image.

## Files

- **`Dockerfile.x86`** – x86_64 build on `ghcr.io/astral-sh/uv:python3.11-bookworm`.
- **`Dockerfile.arm64`** – ARM64 build on `nvidia/cuda:13.0.0-devel-ubuntu22.04`.
- **`build-docker.sh`** – Build entry point with platform auto-detection.
- **`run-examples.sh`** – Discovers examples from `config/*.json`, resolves
  CLI args via `run-example.py`, and runs the selected example in the
  `newton:latest` image.
- **`run-example.py`** – Host-side helper that lists configs and expands
  one JSON file into argparse CLI tokens (with optional interactive edit).
- **`config/`** – One JSON file per example. Each file holds a short
  description plus the example's argparse defaults (base parser + any
  args added by the example itself, pulled straight from the source).
  See `config/README.md` for the schema.

## Runtime flow

```
run-examples.sh  ──►  run-example.py list   ──►  reads config/*.json ──►  menu
                                          ▼
                      run-example.py resolve  (optional --edit / --set)
                                          ▼
              docker run … newton:latest python -m newton.examples <name> <args>
```

CLI arg precedence (last wins under argparse):

1. Values in the JSON `args` block.
2. `--set KEY=VAL` overrides passed to `run-examples.sh`.
3. Interactive edits when `-e` / `--edit` is used.
4. Raw Newton args passed after `--` on the command line.

## Requirements

- Docker with BuildKit
- NVIDIA GPU + driver (optional; CPU fallback works for most examples)
- X11 display if using the `gl` viewer

## Viewer options

Newton in this repo exposes the following viewers (see
`newton/examples/__init__.py` → `create_parser`):
`gl`, `usd`, `rerun`, `null`, `viser`.

> Note: there is **no** RTX / Vulkan viewer in this version of Newton, so
> this setup skips the Vulkan ICD plumbing that existed in the older
> `.devcontainer`. If you later need Vulkan in-container, re-add the
> `--runtime=nvidia` flag and the `/usr/share/vulkan/icd.d` mount from the
> legacy reference script.

## GPU + X11 notes

- `run-examples.sh` detects `nvidia-smi` on the host. If present, it adds
  `--gpus all` and sets `NVIDIA_DRIVER_CAPABILITIES=all`.
- `DISPLAY` is auto-detected from `/tmp/.X11-unix/`. `XAUTHORITY` is
  forwarded if set. If the viewer window won't open, try
  `xhost +local:docker` on the host.

## Dependency extras

The images install the `dev` extra, which transitively pulls in:

- `examples` – `pyglet`, `imgui_bundle`, `GitPython`, `pyyaml`, `cbor2`, `Pillow`
- `sim` – `mujoco`, `mujoco-warp`
- `importers` – USD, mesh libs (`trimesh`, `meshio`, `scipy`, …)

This mirrors the set you get locally with `uv sync --extra dev`.

### Coupled implicit vehicle solve

ANCF-specific coupling lives in `newton/_src/solvers/ancf_shell/`: `coupling.py` owns the interface
iteration and state snapshots, `kernels_coupling.py` holds its shared device kernels, and `schur.py`
/ `coupled_newton.py` provide the condensed solves. Vehicle setup and MuJoCo callbacks remain in
`newton/examples/ancf/_vehicle_simulation.py`. The public `newton.solvers.InterfaceCouplerGS` import
is unchanged; the experimental proxy/ADMM framework remains separate.

Full-vehicle examples 03 (flat ground), 05 (rigid terrain), and 07 (telemetry)
default to `coupling-method=auto` in both the Docker presets and the Python CLI.
Supported implicit vehicles select `coupled-newton`, retaining tire Newton
iterates while correcting the wheels. No override is needed:

```bash
./docker/run-examples.sh 07_vehicle_telemetry
```

The same selection applies to every vehicle asset and tire resolution. `auto`
keeps adaptive coupling for odd substep counts and partitioned coupling
for unsupported rigid models. Use `--set coupling-method=coupled-newton` to
request the method explicitly, or `--set coupling-method=adaptive` to restore
the previous terrain/telemetry preset. The drop and independent wheel rigs do
not use this full-vehicle interface. Sand retains its separate MPM defaults.

This method uses the same HHT integration, contact forces, and full reaction
torque. It shares the free-shell correction with six cached spindle responses.
Small tires whose SGS sweep fits one CUDA block use up to four PCG iterations
per joint correction; larger systems retain `pcg-iters`. The normal `gs-iters`
budget gains three reserve evaluations when the interface residual remains
above 0.025. The usual absolute/relative convergence test is unchanged, and
exhausting the budget still does not imply convergence. A singular rigid mass
matrix retains partitioned Schur coupling and the original linear budget.

The implicit solver, an even number of substeps, and at least three `gs-iters`
are required. Keep the default six-evaluation budget for the validated vehicle
profiles; `auto` retains partitioned coupling with smaller budgets, which have
not passed the moving-vehicle acceptance checks. Select
`coupling-method=schur` to use complete tire solves between interface corrections.
Acceptance probes record the active method, inner iteration budget, and whether
each counter denotes a joint Newton evaluation or a complete shell step.

### Static tire preparation for telemetry

Example 08 reads the physical setup of `config/07_vehicle_telemetry.json`. It
uses the same tire/material, fixed build pressure, contact parameters and
vehicle inflation limits. Its flat single-wheel preview keeps the 6/2/10 solver
settings; an exported replay retains the source 07 solver settings. The default
load is an **asset estimate** of the chassis share per wheel, excluding the
spindle's own weight. It does not substitute for measured corner loads.

Run the pressure demonstration and export a new replay preset:

```bash
./docker/run-examples.sh 08_diffsim_ancf_tire_lift -- \
  --export-config /workspace/newton/docker/config/07_prepared_tire.json
```

The default target is a synthetic height near the opposite end of the reachable
range from the starting pressure. For the Warthog preset, nominal pressure is
limited to 2–4 psi and build pressure is zero. An explicit `--target-height` still
selects a demonstration target; reaching it does not identify a real tire. The single-wheel load screen
uses the distributed reference contact patch; preparation also checks actual
frame-sampled penetration against 25 mm.

The main panel has one target-height slider in millimeters. Move it to start
learning automatically, then watch the large pressure readout and gauge. The
**LEARNING** phase takes bounded gradient steps once per simulated second (at
most 0.2 PSI per update with the default 2–4 PSI range). The default run raises
pressure from 2 PSI toward roughly 3.75 PSI over several visible updates. The
panel compares the starting and current pressures on a scale labeled with the
minimum, midpoint and maximum PSI. A hollow marker shows the starting pressure;
a filled marker shows the current pressure. The pressure graph has labeled PSI
axis values and stays visible after the status turns green at **TARGET REACHED**. This pacing applies
to the pressure demonstration; CSV stiffness fitting uses its own optimizer.

The physical lift is only a few millimeters. In the 3D view, the orange ghost
shows the remaining height gap magnified **50x**: a 4 mm target change moves the
ghost 20 cm. It approaches the purple axle as the error shrinks. This visual
scale is labeled next to the target slider; all height values, physics and
optimization use real units. Turn off **Magnify target motion (50x)** under
**Advanced** for the true-scale view. Startup, CSV fits and unreachable targets
use true scale. The panel also compares real axle and target heights.

After **Target reached**, click **New target** to move the
orange cylinder to a different reachable height and start learning again. You
can also choose the height with the slider. **Repeat same target** returns to the
starting pressure and repeats the same task, keeping the requested height fixed.
This optimizes one pressure for the chosen target, not a reusable controller.

The slider spans the heights supported by the vehicle's pressure limits, computed
after initial settling. An out-of-range CLI target displays an explanation; choose
an in-range height to continue. Manual pressure, visual settings and gradients are
under **Advanced**. Manual pressure pauses learning; moving the target or clicking
**Repeat same target** resumes it. With `--no-train`, use **Start learning** to begin.

The separate telemetry calibration mode remains available on the command line.
To fit one shared Young's-modulus multiplier, supply a CSV with this header:

```csv
nominal_pressure_pa,additional_load_n,axle_height_m,height_std_m,split
```

Use at least two distinct load/pressure cases with `split=train`, and reserve
independent cases with `split=validation`. Pressure is the simulator's **nominal
cavity parameter**, already converted to its reference-volume convention; do
not insert gauge-pressure readings without establishing that conversion.
Additional load excludes spindle weight. Height is the axle centre above the
flat test surface, not the rendered telemetry ghost's height. Uncertainty is a
positive height standard deviation in metres. CSV pressures must fit the
vehicle's inflation envelope; loads must pass the rig's contact-patch screen.

Place the CSV under a mounted project directory, then run:

```bash
./docker/run-examples.sh 08_diffsim_ancf_tire_lift -- \
  --calibration-csv /workspace/newton/docker/config/tire_measurements.csv \
  --export-config /workspace/newton/docker/config/07_prepared_tire.json
```

The CSV fit holds pressure, density, geometry, thickness and damping fixed.
It fits only the elastic modulus using analytic equilibrium sensitivities,
reports training and validation height errors separately, and rebuilds the
preview with the fitted material. Label generated test measurements explicitly
with `--data-source synthetic`.

After successful optimization, export writes the new 07 JSON plus a sibling
`.report.json` containing measurement provenance, input hashes and fit results.
The source 07 preset stays available for comparison. The exported preset selects
recorded-command replay and can be launched with:

```bash
./docker/run-examples.sh 07_prepared_tire
```

It requires a sequence containing recorded commands. Validate vehicle motion on
a held-out recording using the replay evaluator in `newton-rellis-3d-tool`.
Static preparation does not identify actuator response, damping, or skid slip,
and the report explicitly leaves vehicle calibration unvalidated.

### Implicit solver regression tests

Run the focused CUDA regression tests using the existing image and mounted sources:

```bash
bash docker/test-ancf-examples.sh -k TestANCFSolverRegressions \
  -k test_packed_preconditioner_preserves_csr_arithmetic
```

The tests replay captured graphs with changed inputs and dirty scratch buffers,
check deferred stiffness assembly against fresh element evaluations, and drive a
Warthog through acceleration, turning, and braking. Device counters verify that
each coupled substep skips its unused terminal stiffness assembly. The packed
preconditioner test also rejects rebuilding its unused intermediate matrix. These checks
protect numerical behavior and executed work; they do not assert a hardware-dependent
FPS target or replace viewer/Nsight benchmarks. Tests and instrumentation live in
`newton/tests/test_ancf_solver_regressions.py` and `test_ancf_shell_formulation.py`;
production examples are unchanged.

The unfiltered `bash docker/test-ancf-examples.sh` also includes these tests alongside
the existing formulation, torque-transfer, and longer physical acceptance checks.


### Solver selection

The Docker launcher uses the implicit solver for the ANCF examples.

`pcg-iters` limits the implicit linear solve; it is independent of the outer
coupling iterations. Full-vehicle presets use 6 substeps / 2 Newton iterations /
10 PCG iterations. The optimized coupled-Newton path can use fewer inner iterations.

### Single-tire pressure learning

```bash
./docker/run-examples.sh 08_diffsim_ancf_tire_lift
# Run learning without a viewer:
./docker/run-examples.sh 08_diffsim_ancf_tire_lift --set viewer=null --set num-frames=960 --set test=true
# Start with manual pressure exploration:
./docker/run-examples.sh 08_diffsim_ancf_tire_lift --set train=false
# Compare with the original 07 pressure limits:
./docker/run-examples.sh 08_diffsim_ancf_tire_lift --set pressure-range=telemetry
```

Use **Low: 0.25 PSI** / **High: 8.00 PSI** or the pressure slider to inspect
physical tire deformation. These controls pause learning. The white tire uses
the same smooth shading as examples 01/02, without black mesh edges. The actual
height cylinder is purple like the spindle. A translucent gray cylinder
marks the requested height and moves as soon as the height slider changes,
including while paused. Both cylinders use actual heights and are visual markers
only. The Low/High tread visualization stays in the main panel: blue is the
low-PSI shape, orange is the high-PSI shape, and white is the live tire, aligned
at their axles. Its vertical scale is enlarged and labeled in millimeters.

Move the target ride-height slider to learn automatically, or click **Learn
pressure** to repeat from the opposite pressure endpoint. Learning adjusts
pressure at most 10% of its allowed range per simulated second. The physical
rim settles under the tire forces. The goal is one nominal pressure that reaches
a loaded axle height, not a reusable policy or an optimum traction setting.

**Ground** switches between the source 07 flat-contact settings and a firmer
flat support with twice the contact stiffness and damping (`--ground firm`).
The solver is rebuilt and the endpoint shapes are recomputed. The target height
is preserved so pressure can be relearned for the new support; choose another
height if it is outside that support's reachable range. These are uncalibrated
penalty-contact compliance experiments. Rocks, height fields, and deformable
sand are not supported by this equilibrium differentiation model.

The experiment reuses example 02's rig with one simplified Warthog tire,
a MuJoCo rim, zero RPM, and the asset's estimated chassis load per wheel
(490.5 N for the default vehicle). Material, build pressure, and inflation
limits come from `07_vehicle_telemetry.json`; use `--load` for a known additional
corner load. The standalone demonstration expands pressure to **0.25–8 nominal PSI**,
starting at 0.25 PSI, with build pressure still zero. `--pressure-range telemetry`
restores the original 2–4 PSI limits. CSV fitting and replay export select
telemetry limits automatically; explicitly combining either with the demo
range is rejected to avoid exporting a pressure outside the vehicle envelope.

The 0.25–8 PSI interval passed inflation/deflation and upward/downward learning
checks on both flat supports at the default load. The equilibrium height ranges
are **252.94–270.22 mm** on the 07 support and **256.10–274.85 mm** on the firmer
support. Low pressures take several simulated seconds to settle after deflation;
learning reports success only when the moving rim is within 0.25 mm of the goal.
These checks establish numerical behavior for this demonstration, not a real
tire operating limit. At startup and after ground changes, 11 pressures across
the default range are checked for converged, monotonically increasing heights,
positive sensitivity, and penetration below 25 mm before publishing the height
slider bounds. Sampling uses at least five pressures, spaced by no more than a
factor of √2 to cover the softer response below 1 PSI.
Pressure uses the existing cavity convention: `p_live = p_nominal V_ref / V`;
PSI is a unit conversion, not a calibrated gauge-pressure interpretation.

`newton.solvers.ANCFTireEquilibrium` solves the static shell and vertical rim
force balance and differentiates the converged residual:
`J dz/dp = -R_p`, `dL/dp = (h - target) dh/dp`.
A bounded, backtracked update in log pressure reduces static height loss.
Successful learning also checks the moving rim is within 0.25 mm of the target.
The transient MuJoCo trajectory is not differentiated. The forward budget uses
10 substeps / 2 Newton iterations, with 20 PCG iterations for the softer demo
range. Telemetry calibration retains its original 10 PCG iterations. Set
`OPENBLAS_NUM_THREADS=1` for direct Python launches, as the Docker preset already does.

`--calibration-csv` retains static stiffness fitting with training and validation
cases, using the source 07 contact settings. `--export-config` writes a new replay
config and provenance report after convergence, including the selected contact
settings. Static fitting and synthetic pressure targets do not validate full
vehicle motion; recorded-command replay remains the next step for sim-to-real.

Gradient and optimizer checks are in `newton/tests/test_ancf_differentiation.py`.
Coupled learning, ground changes, physical deformation, and calibration checks
are in `newton/tests/test_diffsim_ancf_tire_lift.py`.

### Skid-steer response learning (09)

Example 09 is the next step from the single-tire demo toward example 07. It uses
example 03's MuJoCo vehicle and four deformable ANCF tires, with vehicle, tire,
material, contact and coupling settings read from `config/07_vehicle_telemetry.json`.
It runs on flat ground with fixed pressure and independent left/right wheel motors.

```bash
./docker/run-examples.sh 09_diffsim_ancf_skid_steer
# Complete the automatic demonstration and check its physical result:
./docker/run-examples.sh 09_diffsim_ancf_skid_steer --set viewer=null --set test=true
```

The demonstration runs automatically:

1. **Before:** drive using ideal wheel kinematics. The gray vehicle shows the
   requested speed and turn; the orange trail shows the simulated drive.
2. **Measure:** reset and run straight, gentle left/right and stronger left/right
   maneuvers. Collect body speed and yaw rate after acceleration has settled.
3. **Learn:** fit the response from those observations with Warp gradients.
4. **Learned:** repeat the original target using the fitted response. Compare
   the purple trail and target gap with the initial orange drive.

Each physical maneuver has three seconds of settling and six seconds of driving.
The entire sequence takes about 66 seconds of simulation time; wall time depends
on the GPU. Change the **Speed** or **Turn** slider to test the learned response at
a new target. **Replay before / after** repeats a comparison without fitting again;
**Learn again** repeats the measurements and fit. Positive turns go left. The
initial tested preset requests 0.6 m/s and 2 degrees/s.

The learned response is `v = g0 * v_nominal` and
`yaw = g1 * yaw_nominal + g2 * yaw_nominal**3`, where nominal speed and yaw come
from the left/right motor targets and vehicle geometry. The cubic term accounts
for the stronger skid response at larger wheel-speed differences. Warp
differentiates this small response model; the actual MuJoCo/ANCF rollout supplies
observations and validates the resulting motor commands. There is no gradient
through MuJoCo or tire time integration, and these coefficients are not identified
friction or material constants. A different terrain, tire pressure or load requires
new measurements; this is a local flat-ground response model.

Save observations and a validation report for the next calibration step:

```bash
./docker/run-examples.sh 09_diffsim_ancf_skid_steer --set viewer=null -- \
  --report /workspace/newton/docker/config/results/skid_response.json
```

The JSON labels its data as synthetic and records the coefficients, pressure,
loss history and before/after velocity and position errors. Its sibling CSV
contains commanded left/right wheel speeds and measured body speed/yaw rate.
It is a response-calibration artifact, not an example 07 replay configuration.
It does not overwrite the source preset or change the recorded ROS controller
geometry. Example 07's recorded trajectory has no measured wheel commands, so
using these coefficients with real telemetry requires a separate identification
step and real drive-command data.

Unit checks cover gradients against finite differences, synthetic coefficient
recovery, wheel-axis signs, target changes and export guards. The opt-in CUDA
test also runs the complete physical calibration, repeats it with the same fit,
and validates the opposite turn:

```bash
bash docker/test-ancf-examples.sh -k test_diffsim_ancf_skid_steer
ANCF_SKID_CALIBRATION=1 bash docker/test-ancf-examples.sh -k test_diffsim_ancf_skid_steer
```

### Dynamic tire traction learning (10)

This is the differentiation step between the static tire-lift experiment and
vehicle telemetry calibration: **friction → contact forces → dynamic tire and
spindle motion → trajectory error**. The gray target is a synthetic physical
run with hidden ground friction. The white tire receives the same wheel-speed
commands, with an initially incorrect friction estimate. Its forward travel
is free; it is determined by contact forces.

```bash
./docker/run-examples.sh 10_diffsim_ancf_tire_traction
./docker/run-examples.sh 10_diffsim_ancf_tire_traction --set viewer=null --set test=true
```

The default case learns from a 0.8 s spin-and-brake maneuver: ramp to 10 rad/s
over 0.1 s, hold until 0.4 s, then command zero spin. It fits one positive
friction coefficient from the entire travel and speed histories. After fitting,
it runs a separate 7 rad/s maneuver with an earlier brake command. Before and
after errors for that separate maneuver measure how well the fit transfers.
The reference coefficient generates observations only; the optimizer uses
motion residuals and their derivatives.

**Learn friction** repeats the fit from the initial estimate. **Replay test**
repeats the independent before/after validation. **Ground** switches between
synthetic slippery and grippy reference cases and starts a fresh experiment.
The tire stays white with a purple spindle; its reference is translucent gray.
Pressure (2 nominal PSI), rim plus carried mass (50 kg), elastic modulus
(50 MPa), density (700 kg/m³), and the motor command remain fixed during fitting.
These are controlled experiment settings, not identified real vehicle values.

`newton.solvers.ANCFTireTraction` reuses the ANCF3423 elastic/ANS/EAS and cavity
equations and integrates all free shell coordinates and the spindle's vertical
and longitudinal coordinates with backward Euler. Rim spin is prescribed by an
ideal velocity motor. Plane contact uses penalty stiffness and tanh Coulomb
friction (regularization speed 0.01 m/s). Normal contact damping and structural
Rayleigh damping are zero in this experiment. The abrupt normal-damping force
at new contact can prevent an implicit root; backward Euler supplies numerical
damping here. This dedicated solver is not the MuJoCo/HHT stepping path used by
example 07, and its fitted coefficient must not be transferred as a validated
real terrain parameter.

Every step must converge before its derivative is accepted. With one unknown,
an analytic forward sensitivity propagates through the converged implicit
equations, including previous position and velocity sensitivities. No finite
differences are used for training. Gradients are local to contact/EAS active
branches; finite-difference checks use independently rerun trajectories. The
loss normalizes travel by 0.25 m and speed by 1 m/s. Bounded updates in log
friction are accepted only when the full trajectory loss decreases.

The dense float64 solves run on CPU. Simulation playback is slower than real
time; the launcher limits BLAS threads to avoid oversubscription. The 1600-frame
headless budget includes the reference, learning attempts and validation.

```bash
./docker/run-examples.sh 10_diffsim_ancf_tire_traction --set viewer=null -- \
  --report /workspace/newton/docker/config/results/tire_traction.json

bash docker/test-ancf-examples.sh -k test_diffsim_ancf_tire_traction
ANCF_TRACTION_TESTS=1 bash docker/test-ancf-examples.sh -k test_diffsim_ancf_tire_traction
```

The full tests require the USD asset loader. They check contact and step
Jacobians, the trajectory gradient against central differences, repeatable
resets, friction recovery, a different validation command, and time-step
refinement. The report records synthetic provenance and both command histories;
it does not modify example 07 or its recordings.

### Uncoupled wheel calibration checks

Start with example 01 (`ancf_shell_drop`) before identifying a coupled vehicle.
The fixture uses one Warthog simplified tire and the implicit solver, without a
rigid body or wheel–chassis coupling. Run its short regression checks with:

```bash
bash docker/test-ancf-examples.sh -k TestWheelCalibrationSearch -k TestANCFWheelCalibration
```

The optional recovery experiment repeatedly runs the actual drop example:

```bash
ANCF_WHEEL_RECOVERY=1 ANCF_TEST_OUTPUT_DIR=/tmp/newton-wheel-calibration \
  bash docker/test-ancf-examples.sh -k test_synthetic_stiffness_recovery_and_validation
```

It fits a positive multiplier of the baked orthotropic stiffness tensor from
synthetic height and vertical-extent measurements, then validates against a
separate drop height. Density, damping, nominal/build pressure (3 psi), and contact stiffness
(20 MN/m) are fixed; these are explicit test-fixture settings, not calibrated
vehicle parameters. The reference solve uses 20 substeps / 4 Newton / 40 PCG
and is compared with 40 / 4 / 40 for timestep refinement. The experiment also
compares the unchanged 6 / 2 / 10 production budget against that reference.

`wheel_recovery/report.json` records every candidate and separate recovery,
validation, and numerical-accuracy gates. Passing the recovery test requires
less than 2% stiffness error, less than 1 mm validation error (and at least a
70% improvement over the nominal model), and less than 1 mm timestep-refinement
error. Production accuracy is reported separately; it must also be below 1 mm
for `ready_for_coupling` to be true. `traces.npz` contains observations and
predictions in SI units. Penetration is sampled once per frame, so it does not
bound penetration at every substep. Short regressions also check that material
overrides reach a single wheel, masses and repeated drops are reproducible,
and free fall alone cannot identify stiffness.

Synthetic recovery only checks the identification machinery. It does not establish
real tire accuracy, and a good trajectory fit does not certify the production
solver's accuracy. Do not advance to coupled calibration while the report's
`ready_for_coupling` gate is false. Measured wheel deformation or drop data are
still required for physical calibration.

### Sim-to-real physics checks

Rigid terrain uses the same cell triangles for tire contact, chassis collision,
rendering, and height queries. The numerical coupling screen now exercises both
RELLIS 00000 and 00004 for 45 seconds, including late chassis contact:

```bash
bash docker/test-ancf-examples.sh -k TestANCFTerrainContact \
  -k test_parallel_interface_convergence_and_reserve_budget \
  -k test_07_interface_velocity_mismatch_is_bounded
```

With `gs-iters=6`, coupled Newton retains its normal six-evaluation budget and
permits up to twelve evaluations when the residual remains above 0.025. This
extends the former nine-evaluation cap without increasing routine iterations;
6 substeps / 2 Newton / 10 PCG and full reaction torque remain unchanged.
An exhausted cap can still leave a nonconverged step: the tests retain the
independent 0.1 residual screen. Probe JSON includes rigid contact shapes and
constraint forces to distinguish chassis obstruction from motor saturation.
These checks do not certify real-world fidelity; the trajectory-accuracy and
uneven-terrain holding limits remain separate acceptance tests.
