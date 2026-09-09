# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example: Inflatable FEM Demo
#
# Four modes selected via --mode:
#   tet      (default) — tet FEM soft bodies loaded from USD assets
#   hex                — Q1 hex FEM objects with independent pressure chambers
#   glue               — tet FEM soft body coupled to a rigid body via Glue
#   showcase           — all three groups side-by-side with 7 pressure chambers
#
# Commands:
#   uv run -m newton.examples inflatable.example_inflatable
#   uv run -m newton.examples inflatable.example_inflatable --mode hex
#   uv run -m newton.examples inflatable.example_inflatable --mode glue
#   uv run -m newton.examples inflatable.example_inflatable --mode showcase
###########################################################################

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton._src.solvers.inflatable.glue import Glue
from newton._src.solvers.inflatable._glue_mujoco import GlueMuJoCo
from newton._src.solvers.inflatable.se3_tracker import (
    SE3Tracker,
    quat_compose,
    quat_rotate,
    rotvec_to_quat,
)
from newton._src.solvers.inflatable.solver_inflatable import SolverInflatable
from newton.solvers import SolverFeatherstone, SolverMuJoCo

from ._usd_asset import detect_asset_type, load_asset, load_glued_assembly, load_hex_assembly

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

# Hex mode constants
_HEX_OBJECT_NAMES = ["cube", "tire", "pyramid"]
_HEX_PRESSURE_RATE = 0.04  # pressure fraction stepped per frame


def _resolve_path(path: str) -> str:
    """Resolve an asset path: absolute paths are used as-is; relative paths
    are resolved relative to the example directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(_EXAMPLE_DIR, path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _add_surface_tris(builder, local_tris: np.ndarray, p_off: int) -> None:
    """Add local surface triangles to the builder with a global offset (zero stiffness)."""
    for a, b, c in local_tris:
        builder.add_triangle(int(a) + p_off, int(b) + p_off, int(c) + p_off, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)


def _desired_pose(
    dx: float,
    dy: float,
    dz: float,
    rx: float,
    ry: float,
    rz: float,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (desired_pos, desired_quat) from body-local offsets and angles.

    Translation is along the body's own axes at enable time (base_quat frame).
    Rotation is intrinsic (body-local): base_quat ⊗ Rx ⊗ Ry ⊗ Rz.
    Moving one slider never changes the axis direction of the others.
    """
    qx = rotvec_to_quat(np.array([rx, 0.0, 0.0]))
    qy = rotvec_to_quat(np.array([0.0, ry, 0.0]))
    qz = rotvec_to_quat(np.array([0.0, 0.0, rz]))
    des_quat = quat_compose(base_quat, quat_compose(qx, quat_compose(qy, qz)))
    des_pos = base_pos + quat_rotate(base_quat, np.array([dx, dy, dz], dtype=np.float64))
    return des_pos, des_quat


# ---------------------------------------------------------------------------
# _SoftBodyExample — shared base for tet and hex modes
# ---------------------------------------------------------------------------


class _SoftBodyExample:
    """Shared simulate / step / gui base for tet and hex inflatable modes.

    Subclasses must populate before calling any of these methods:
        self.viewer, self.solver, self.model
        self.state_0, self.state_1, self.control
        self.sim_substeps, self.sim_dt, self.frame_dt
        self.sim_time  (float, initialised to 0.0)
        self.graph     (wp.Graph | None)
        self._current_pressures, self._target_pressures, self._pressure_rates  (list[float])
        self._pressure_names  (list[str])
        self._max_pressure    (float)
    """

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        # Ramp per-chamber pressures toward their targets and push to solver.
        if self._current_pressures:
            for i in range(len(self._current_pressures)):
                delta = self._target_pressures[i] - self._current_pressures[i]
                rate = self._pressure_rates[i]
                if abs(delta) <= rate:
                    self._current_pressures[i] = self._target_pressures[i]
                else:
                    self._current_pressures[i] += rate * (1.0 if delta > 0.0 else -1.0)
            # Apply BEFORE the graph launch so the solver sees updated element poses.
            self.solver.set_chamber_pressures(list(self._current_pressures))

        # Refresh element Hessian from the current deformed positions before the graph.
        # update_preconditioner=False when a CUDA graph is active: bsr_set_from_triplets
        # reuses the same A_bsr buffer pointer (graph reads fresh values), but
        # preconditioner() would allocate a new inv_diag and freeing the old one would
        # invalidate the pointer captured by the graph.
        self.solver._rebuild_element_blocks(
            self.state_0.particle_q,
            self.sim_dt,
            update_preconditioner=(self.graph is None),
        )

        # Rebuild surface-contact BVHs from the current state before the CUDA graph
        # launches.  mesh.refit() goes through the Warp C++ runtime and may not be
        # captured in a CUDA graph; an explicit host-side call here guarantees the BVH
        # topology is up-to-date every frame.
        sc = self.solver._soft_surface_contacts
        if sc is not None:
            sc.update(self.state_0.particle_q)

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def gui(self, ui) -> None:
        """One pressure slider per chamber."""
        for i, name in enumerate(self._pressure_names):
            changed, value = ui.slider_float(
                f"pressure {name}##{i}", self._target_pressures[i], 1.0, self._max_pressure
            )
            if changed:
                self._target_pressures[i] = float(value)


# ---------------------------------------------------------------------------
# Example — public factory (defined before subclasses so they can inherit it)
# ---------------------------------------------------------------------------


class Example:
    """Inflatable FEM example.  Mode selected via ``--mode tet|hex|glue``.

    ``Example(viewer, args)`` dispatches to :class:`_TetExample`,
    :class:`_HexExample`, or :class:`_GlueExample` based on ``args.mode``
    so callers never need to import the private classes.
    """

    def __new__(cls, viewer, args):
        if cls is Example:
            mode = getattr(args, "mode", "tet")
            target = {
                "tet": _TetExample,
                "hex": _HexExample,
                "glue": _GlueExample,
                "showcase": _ShowcaseExample,
            }[mode]
            return object.__new__(target)
        return object.__new__(cls)

    @staticmethod
    def create_parser():
        """Return a fully populated argparse.ArgumentParser for all three modes."""
        parser = newton.examples.create_parser()

        # Mode
        parser.add_argument(
            "--mode",
            choices=["tet", "hex", "glue", "showcase"],
            default="tet",
            help="Simulation mode: tet FEM / hex FEM / FEM+rigid glue / showcase",
        )

        # Simulation timing
        parser.add_argument("--substeps", type=int, default=5)
        parser.add_argument("--fps", type=int, default=60)
        parser.add_argument("--gravity", type=float, default=9.81)

        # FEM material (tet and hex)
        parser.add_argument("--k-mu", type=float, default=1e5, dest="k_mu")
        parser.add_argument("--k-lambda", type=float, default=1e5, dest="k_lambda")
        parser.add_argument("--k-damp", type=float, default=1.0, dest="k_damp")
        parser.add_argument("--density", type=float, default=1.0)
        parser.add_argument("--particle-radius", type=float, default=0.008, dest="particle_radius")
        parser.add_argument("--linear-damping", type=float, default=0.0, dest="linear_damping")

        # Solver
        parser.add_argument("--solver-type", default="cg", dest="solver_type")
        parser.add_argument("--solver-maxiter", type=int, default=100, dest="solver_maxiter")
        parser.add_argument(
            "--preconditioner",
            default="diag",
            choices=["diag", "diag_abs", "block_diag", "id"],
            dest="preconditioner",
            help="CG preconditioner. 'diag' (default) is Warp point-Jacobi; "
            "'block_diag' inverts the full 3x3 per-particle block, which "
            "usually cuts the iteration count substantially.",
        )
        parser.add_argument(
            "--cg-check-every",
            type=int,
            default=0,
            dest="cg_check_every",
            help="0 (default) = fixed iteration count, CUDA-graph capturable. "
            ">0 = host-side residual checks with early exit; disables graph "
            "capture. Use to measure how many iterations the solve needs.",
        )

        # Pressure
        parser.add_argument("--max-pressure", type=float, default=3.0, dest="max_pressure")
        parser.add_argument("--pressure-rate", type=float, default=0.05, dest="pressure_rate")

        # Ground
        parser.add_argument("--ground-ke", type=float, default=1e5, dest="ground_ke")
        parser.add_argument("--ground-kd", type=float, default=1e2, dest="ground_kd")
        parser.add_argument("--ground-kf", type=float, default=1e3, dest="ground_kf")
        parser.add_argument("--ground-mu", type=float, default=0.5, dest="ground_mu")

        # Self-contact (tet and hex)
        parser.add_argument("--contact-ke", type=float, default=1e3, dest="contact_ke")
        parser.add_argument("--contact-kd", type=float, default=1e2, dest="contact_kd")

        # Tet-mode specific
        parser.add_argument(
            "--soft-objects", default="[]", dest="soft_objects", help="JSON list of per-object configs (tet mode only)"
        )
        parser.add_argument("--debug-interval", type=int, default=0, dest="debug_interval")

        # Glue-mode specific
        parser.add_argument("--asset", default=None, help="Path to glued assembly USDA (glue mode only)")
        parser.add_argument(
            "--rigid-solver", choices=["featherstone", "mujoco"], default="featherstone", dest="rigid_solver"
        )
        parser.add_argument("--coupling-alpha", type=float, default=0.05, dest="coupling_alpha")
        parser.add_argument("--coupling-kd-rigid", type=float, default=0.0, dest="coupling_kd_rigid")
        parser.add_argument("--glue-damping", type=float, default=1.0, dest="glue_damping")
        parser.add_argument("--rigid-mass", type=float, default=0.05, dest="rigid_mass")
        parser.add_argument("--rigid-color", type=float, nargs=3, default=[0.85, 0.45, 0.10], dest="rigid_color")
        parser.add_argument("--glued", action=argparse.BooleanOptionalAction, default=None)

        # Showcase-mode specific — no default; must be supplied via the JSON config
        parser.add_argument(
            "--objects",
            default=None,
            dest="objects",
            help="JSON list of per-object configs for showcase mode. "
            'Each entry: {"path": "...", "position": [x,y,z], "color": [r,g,b], '
            '"k-mu", "k-lambda", "k-damp", "density", "particle-radius"}. '
            "Set in docker/config/inflatable.json.",
        )

        return parser

    def test_final(self) -> None:
        pass


# ---------------------------------------------------------------------------
# _TetExample — tet FEM, multi-object, CUDA graph
# ---------------------------------------------------------------------------


class _TetExample(_SoftBodyExample, Example):
    """Multi-object tet FEM demo (``--mode tet``)."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        gravity = float(getattr(args, "gravity", 9.81))
        self._debug_interval = max(0, int(getattr(args, "debug_interval", 0)))
        self._frame_idx = 0
        self._rest_particle_q_np: np.ndarray | None = None

        # ── Config (from CLI args) ────────────────────────────────────────
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps

        solver_type = str(args.solver_type)
        solver_maxiter = int(args.solver_maxiter)
        ground_ke = float(args.ground_ke)
        ground_kd = float(args.ground_kd)
        ground_kf = float(args.ground_kf)
        ground_mu = float(args.ground_mu)
        contact_ke = float(args.contact_ke)
        contact_kd = float(args.contact_kd)
        linear_damping = float(args.linear_damping)

        soft_objects_raw: list = json.loads(args.soft_objects) if args.soft_objects else []
        active_items = [(obj["name"], obj) for obj in soft_objects_raw if obj.get("active", True)]
        if not active_items:
            raise ValueError('No soft objects have "active": true in --soft-objects. Pass at least one active object.')

        # ── Build model ───────────────────────────────────────────────────
        builder = newton.ModelBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=ground_ke, kd=ground_kd, kf=ground_kf, mu=ground_mu)
        )

        tet_ranges: list[tuple[int, int]] = []
        spring_ranges: list[tuple[int, int]] = []
        particle_ranges: list[tuple[int, int]] = []
        surface_triangles_list: list = []
        for name, obj in active_items:
            if "path" not in obj:
                raise ValueError(f"Object '{name}' in --soft-objects is missing a 'path' field")
            path = _resolve_path(obj["path"])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Asset not found: {path}. Regenerate with third_party/newton-mesh-tools/scripts/generate_assets.py"
                )
            asset = load_asset(path)
            verts = asset["vertices"]
            tets = asset["tets"]
            if "position" not in obj:
                raise ValueError(f"Object '{obj.get('name', path)}' is missing 'position'")
            pos = obj["position"]
            tet_start = builder.tet_count
            spring_start = builder.spring_count
            p_start = builder.particle_count
            builder.add_soft_mesh(
                pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=verts,
                indices=tets.flatten(),
                scale=1.0,
                density=float(obj.get("density", 1.0)),
                k_mu=float(obj.get("k-mu", 1.0e5)),
                k_lambda=float(obj.get("k-lambda", 1.0e5)),
                k_damp=float(obj.get("k-damp", 1.0)),
                particle_radius=float(obj.get("particle-radius", 0.008)),
            )
            tet_ranges.append((tet_start, builder.tet_count))
            spring_ranges.append((spring_start, builder.spring_count))
            particle_ranges.append((p_start, builder.particle_count - p_start))
            surface_triangles_list.append(asset.get("surface_triangles"))

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -float(gravity)))

        # Use density-value directly as per-particle mass [kg].
        # The builder computes particle_mass = density * tet_volume / 4 (micro-grams for
        # small tets), but ground_ke / contact_ke were tuned for mass ~ density_val.
        _mass_np = np.empty(self.model.particle_count, dtype=np.float32)
        for (p_start, p_count), (_, obj) in zip(particle_ranges, active_items, strict=False):
            _mass_np[p_start : p_start + p_count] = float(obj.get("density", 1.0))
        self.model.particle_mass = wp.array(_mass_np, dtype=wp.float32, device=self.model.device)

        # ── Per-object pressure bookkeeping ───────────────────────────────
        pressure_items: list[tuple[int, str, dict]] = [
            (i, name, obj) for i, (name, obj) in enumerate(active_items) if obj.get("pressure-mode", False)
        ]
        self._n_pressure = len(pressure_items)
        self._any_pressure_mode = self._n_pressure > 0

        self._pressure_names: list[str] = []
        self._max_pressures: list[float] = []
        self._pressure_rates: list[float] = []
        self._target_pressures: list[float] = []
        self._current_pressures: list[float] = []
        for _, name, obj in pressure_items:
            self._pressure_names.append(name)
            self._max_pressures.append(float(obj.get("max-pressure", 5.0)))
            self._pressure_rates.append(float(obj.get("pressure-rate", 0.05)))
            self._target_pressures.append(1.0)
            self._current_pressures.append(1.0)
        self._max_pressure = max(self._max_pressures) if self._max_pressures else 1.0

        # ── Solver ────────────────────────────────────────────────────────
        max_vol = max(self._max_pressures) if self._max_pressures else 1.0

        self.solver = newton.solvers.SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            max_volume_ratio=max_vol if self._any_pressure_mode else 1.0,
            solver_type=solver_type,
            linear_solver_maxiter=solver_maxiter,
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=ground_ke,
            ground_kd=ground_kd,
            ground_kf=ground_kf,
            ground_mu=ground_mu,
            self_contact_ke=contact_ke,
            self_contact_kd=contact_kd,
            linear_damping=linear_damping,
        )

        if len(particle_ranges) > 1:
            self.solver.set_soft_surface_contacts(
                particle_ranges=particle_ranges,
                surface_triangles_list=surface_triangles_list,
                ke=contact_ke,
                kd=contact_kd,
            )

        # Wire up per-object chamber pressures.
        if self._any_pressure_mode:
            tet_total = self.model.tet_count
            spring_total = self.model.spring_count
            tet_mask = np.full(tet_total, -1, dtype=np.int32)
            spring_mask = np.full(spring_total, -1, dtype=np.int32)
            for chamber_id, (obj_idx, _, _) in enumerate(pressure_items):
                t0, t1 = tet_ranges[obj_idx]
                tet_mask[t0:t1] = chamber_id
                s0, s1 = spring_ranges[obj_idx]
                spring_mask[s0:s1] = chamber_id
            self.solver.set_chamber_mask(
                wp.array(tet_mask, dtype=wp.int32, device=self.model.device),
                spring_chamber_mask=wp.array(spring_mask, dtype=wp.int32, device=self.model.device),
                num_chambers=self._n_pressure,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True

        if self._debug_interval > 0:
            self._rest_particle_q_np = self.state_0.particle_q.numpy().copy()

        self.capture()

    def capture(self) -> None:
        """Warm up kernels, snapshot initial state, then capture a CUDA graph.

        Pressure updates are applied in :meth:`step` *before* the graph
        launch — not inside :meth:`simulate` — so the graph captures only
        the solver substep loop and never bakes stale pressure values.

        When inter-body surface contacts are active, graph capture is skipped.
        ``mesh.refit()`` goes through the Warp C++ runtime and is not reliably
        captured; using a graph would freeze the BVH at frame-start positions
        across all substeps, causing objects to pass through each other.
        Eager mode calls ``solver.step()`` (and therefore ``sc.update()``) once
        per substep, so the BVH is always current.
        """
        self.graph = None
        if not wp.get_device().is_cuda:
            return
        if self.solver._soft_surface_contacts is not None:
            # Warm up kernels but do not capture — see docstring above.
            snap_0 = self.model.state()
            snap_0.assign(self.state_0)
            snap_1 = self.model.state()
            snap_1.assign(self.state_1)
            try:
                self.simulate()
                wp.synchronize_device()
            except Exception as exc:
                print(f"   [capture] warmup failed ({exc}).", flush=True)
            if self.sim_substeps % 2 == 1:
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_0.assign(snap_0)
            self.state_1.assign(snap_1)
            return
        snap_0 = self.model.state()
        snap_0.assign(self.state_0)
        snap_1 = self.model.state()
        snap_1.assign(self.state_1)
        try:
            self.simulate()
            wp.synchronize_device()
        except Exception as exc:
            print(f"   [capture] warmup failed ({exc}); running eager.", flush=True)
            self.state_0.assign(snap_0)
            self.state_1.assign(snap_1)
            return
        if self.sim_substeps % 2 == 1:
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.state_0.assign(snap_0)
        self.state_1.assign(snap_1)
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            if self.sim_substeps % 2 == 1:
                self.state_0, self.state_1 = self.state_1, self.state_0
        except Exception as exc:
            print(f"   [capture] ScopedCapture failed ({exc}); running eager.", flush=True)
            self.state_0.assign(snap_0)
            self.state_1.assign(snap_1)
            self.graph = None

    def step(self) -> None:
        super().step()
        self._frame_idx += 1
        if self._debug_interval > 0 and self._frame_idx % self._debug_interval == 0:
            self._dump_debug_state()

    def _dump_debug_state(self) -> None:
        """Print one line of particle bbox / NaN / drift / pressure state."""
        try:
            q = self.state_0.particle_q.numpy()
            nan_n = int(np.isnan(q).any(axis=1).sum())
            parts = [f"[dbg t={self.sim_time:7.3f}s f={self._frame_idx:6d}]"]
            if nan_n:
                parts.append(f"NaN={nan_n}/{q.shape[0]}")
            else:
                qmin = q.min(axis=0)
                qmax = q.max(axis=0)
                bbox = qmax - qmin
                parts.append(
                    f"qmin=({qmin[0]:+.3f},{qmin[1]:+.3f},{qmin[2]:+.3f}) qmax=({qmax[0]:+.3f},{qmax[1]:+.3f},{qmax[2]:+.3f}) "
                    f"bbox=({bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f})"
                )
                if self._rest_particle_q_np is not None:
                    drift = np.linalg.norm(q - self._rest_particle_q_np, axis=1)
                    parts.append(f"drift max={float(drift.max()):.4f} mean={float(drift.mean()):.4f}")
            for name, cur, tgt in zip(
                self._pressure_names, self._current_pressures, self._target_pressures, strict=False
            ):
                parts.append(f"{name} cur={cur:.3f} tgt={tgt:.3f}")
            print(" ".join(parts), flush=True)
        except Exception as exc:
            print(f"[inflatable tet][dbg] dump failed: {exc}", flush=True)

    def gui(self, ui) -> None:
        """ImGui sliders — one per pressure-mode object (with per-object max)."""
        for i, (name, max_p) in enumerate(zip(self._pressure_names, self._max_pressures, strict=False)):
            changed, value = ui.slider_float(f"pressure {name}##{i}", self._target_pressures[i], 1.0, max_p)
            if changed:
                print(
                    f"[gui] slider[{i}] {name!r} → {value:.4f}  (targets before: {self._target_pressures})", flush=True
                )
                self._target_pressures[i] = float(value)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        p_lower = wp.vec3(-5.0, -5.0, -0.5)
        p_upper = wp.vec3(5.0, 5.0, 10.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles stay in workspace",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )


# ---------------------------------------------------------------------------
# _HexExample — hex FEM, no CUDA graph
# ---------------------------------------------------------------------------


class _HexExample(_SoftBodyExample, Example):
    """Three Q1 hex FEM objects with independent per-chamber pressure control (``--mode hex``)."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        k_mu = args.k_mu
        k_lambda = args.k_lambda
        k_damp = args.k_damp
        density = args.density
        max_p = args.max_pressure
        ground_ke = args.ground_ke
        ground_kd = args.ground_kd
        ground_kf = args.ground_kf
        ground_mu = args.ground_mu
        contact_ke = args.contact_ke
        contact_kd = args.contact_kd
        cg_maxiter = args.solver_maxiter

        # mass floor: ensures kd_ground * dt / m < 2 for all nodes (safety factor 1.5)
        _m_floor = 10.0 * self.sim_dt * 1.5  # ≈ 0.05 kg at substeps=5

        # ── Load hex objects; type auto-detected from each USDA path ─────
        if not args.objects:
            raise ValueError("--objects is required for hex mode. Set it in docker/config/inflatable.json.")
        all_objects = json.loads(args.objects)
        hex_cfgs, hex_objects = [], []
        for o in all_objects:
            if "path" not in o:
                raise ValueError(f"Object entry missing 'path': {o}")
            path = _resolve_path(o["path"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Asset not found: {path}")
            kind = o.get("type") or detect_asset_type(path)
            if kind != "hex":
                continue
            objs = load_hex_assembly(path)
            if not objs:
                raise RuntimeError(f"No hex mesh prim found in {path}")
            usd_obj = objs[0]
            usd_obj["_cfg"] = o
            hex_cfgs.append(o)
            hex_objects.append(usd_obj)
        if not hex_objects:
            raise ValueError("--objects contains no hex assets")

        # ── Assemble Newton model ─────────────────────────────────────────
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        hex_ranges: list[tuple[int, int]] = []
        particle_ranges: list[tuple[int, int]] = []
        surface_triangles_list: list[np.ndarray] = []
        hex_cursor = 0

        for obj in hex_objects:
            verts = obj["vertices"]  # (V,3) float32 — world-space
            hexes = obj["hex_indices"]  # (H,8) int32   — local
            local_tris = obj["surface_triangles"]  # (T,3) int32 — local

            H_obj = hexes.shape[0]
            n_verts = verts.shape[0]

            # Lumped mass: approximate element volume from node-0 / node-6
            # bounding box.  Overestimates ~2-3× for curved tire elements
            # (safe: more mass → more stable).
            node_mass = np.zeros(n_verts, dtype=np.float64)
            for e in range(H_obj):
                n0, n6 = int(hexes[e, 0]), int(hexes[e, 6])
                dx = abs(float(verts[n6, 0]) - float(verts[n0, 0]))
                dy = abs(float(verts[n6, 1]) - float(verts[n0, 1]))
                dz = abs(float(verts[n6, 2]) - float(verts[n0, 2]))
                em = density * max(dx * dy * dz, 1e-10)
                for a in range(8):
                    node_mass[int(hexes[e, a])] += em / 8.0

            p_off = builder.particle_count
            p_radius = float(obj["_cfg"].get("particle-radius", 0.008))
            for i in range(n_verts):
                pos = verts[i]
                builder.add_particle(
                    pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    mass=float(max(node_mass[i], _m_floor)),
                    radius=p_radius,
                )

            shifted_hexes = hexes + p_off
            builder.add_soft_hex_mesh(
                hex_indices=shifted_hexes,
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=k_damp,
            )
            _add_surface_tris(builder, local_tris, p_off)
            particle_ranges.append((p_off, n_verts))
            surface_triangles_list.append(local_tris)
            hex_ranges.append((hex_cursor, hex_cursor + H_obj))
            hex_cursor += H_obj

        # Chamber mask: each object keeps its chamber_id from the USD
        total_H = hex_cursor
        hex_chamber_mask = np.full(total_H, -1, dtype=np.int32)
        for obj, (h0, h1) in zip(hex_objects, hex_ranges, strict=False):
            hex_chamber_mask[h0:h1] = obj["chamber_id"]

        self.model = builder.finalize()

        # ── Create solver ─────────────────────────────────────────────────
        self.solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            max_volume_ratio=max_p,
            preconditioner_type="diag",
            solver_type="cg",
            linear_solver_maxiter=cg_maxiter,
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=ground_ke,
            ground_kd=ground_kd,
            ground_kf=ground_kf,
            ground_mu=ground_mu,
            self_contact_ke=0.0,
            self_contact_kd=0.0,
        )

        self.solver.set_soft_surface_contacts(
            particle_ranges=particle_ranges,
            surface_triangles_list=surface_triangles_list,
            ke=contact_ke,
            kd=contact_kd,
        )

        self.solver.set_chamber_mask(
            element_chamber_mask=wp.array(hex_chamber_mask, dtype=wp.int32, device=self.model.device),
            spring_chamber_mask=None,
            num_chambers=3,
        )

        # Pressure bookkeeping
        self._pressure_names = list(_HEX_OBJECT_NAMES)
        self._target_pressures = [1.0, 1.0, 1.0]
        self._current_pressures = [1.0, 1.0, 1.0]
        self._pressure_rates = [_HEX_PRESSURE_RATE, _HEX_PRESSURE_RATE, _HEX_PRESSURE_RATE]
        self._max_pressure = float(max_p)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Hex mode always runs eager — no CUDA graph.
        self.graph = None

        self.viewer.set_model(self.model)

    def gui(self, ui) -> None:
        for i, name in enumerate(_HEX_OBJECT_NAMES):
            _, self._target_pressures[i] = ui.slider_float(
                f"pressure {name}##{i}", self._target_pressures[i], 1.0, self._max_pressure
            )
        for i, name in enumerate(_HEX_OBJECT_NAMES):
            ui.text(f"  {name}: p={self._current_pressures[i]:.3f}")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.model.contacts(), self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        p_lower = wp.vec3(-2.0, -2.0, -0.5)
        p_upper = wp.vec3(2.0, 2.0, 15.0)
        newton.examples.test_particle_state(
            self.state_0,
            "hex-demo particles stay within bounds",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )


# ---------------------------------------------------------------------------
# _GlueExample — tet FEM soft body coupled to a rigid body via Glue
# ---------------------------------------------------------------------------


class _GlueExample(Example):
    """Rigid box glued to an inflatable box via Dirichlet pin + Featherstone coupling.

    The coupling is force-based: the FEM reaction is written into ``State.body_f``
    before each Featherstone step so the articulated-body solver integrates gravity,
    ground contact, and inertia together with the elastic coupling force.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        # ── Config (from CLI args) ───────────────────────────────────────
        gravity = float(getattr(args, "gravity", 9.81))
        fps = int(args.fps)
        substeps = int(args.substeps)
        solver_type = str(args.solver_type)
        solver_iter = int(args.solver_maxiter)
        density = float(args.density)
        k_mu = float(args.k_mu)
        k_lambda = float(args.k_lambda)
        k_damp = float(args.k_damp)
        particle_r = float(args.particle_radius)
        self_contact_ke = float(args.contact_ke)
        self_contact_kd = float(args.contact_kd)
        max_pressure = float(args.max_pressure)
        pressure_rate = float(args.pressure_rate)
        self._particle_r = particle_r

        rigid_mass = float(args.rigid_mass)
        rigid_color = wp.vec3(*args.rigid_color)

        coupling_alpha = float(args.coupling_alpha)
        kd_rigid = float(args.coupling_kd_rigid)
        glue_damping = float(getattr(args, "glue_damping", 1.0))

        gnd_ke = float(args.ground_ke)
        gnd_kd = float(args.ground_kd)
        gnd_kf = float(args.ground_kf)
        gnd_mu = float(args.ground_mu)

        self._rigid_solver_type: str = getattr(args, "rigid_solver", None) or "featherstone"
        self._use_mujoco = self._rigid_solver_type == "mujoco"

        self.fps = fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # ── Load GluedAssembly ───────────────────────────────────────────
        if getattr(args, "asset", None):
            asset_path = args.asset
        elif args.objects:
            glue_cfgs = [
                o
                for o in json.loads(args.objects)
                if (o.get("type") or detect_asset_type(_resolve_path(o["path"]))) == "glue"
            ]
            asset_path = _resolve_path(glue_cfgs[0]["path"]) if glue_cfgs else _GLUE_ASSET
        else:
            asset_path = _GLUE_ASSET
        glued = load_glued_assembly(asset_path)
        soft_names = glued["soft_names"]
        rigids = glued["rigids"]
        glue_sets = glued["glue_sets"]

        # Minimum rigid COM z for manual-teleport clamping.
        self._rigid_hz = particle_r

        # Particle index offset per soft body (for glue index arrays).
        soft_particle_start: dict[str, int] = {}
        _off = 0
        for n in soft_names:
            soft_particle_start[n] = _off
            _off += len(glued["softs"][n]["vertices"])

        # ── Build model ──────────────────────────────────────────────────
        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=gnd_ke, kd=gnd_kd, kf=gnd_kf, mu=gnd_mu))

        # Add every soft body at its canonical placement position from the asset.
        for sname in soft_names:
            s = glued["softs"][sname]
            sv = np.asarray(s["vertices"], dtype=np.float32)
            st = np.asarray(s["tets"], dtype=np.int32)
            spos = np.array(s["placement"]["pos"], dtype=np.float32)
            builder.add_soft_mesh(
                pos=wp.vec3(*spos),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=sv,
                indices=st.flatten(),
                scale=1.0,
                density=density,
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=k_damp,
                particle_radius=particle_r,
            )

        # Rigid bodies: canonical placement from asset, ground collision enabled
        # as a safety net.  Particle collision disabled — the FEM supports the
        # rigid via the glue coupling, not direct particle contact.
        rigid_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=gnd_ke, kd=gnd_kd, kf=gnd_kf, mu=gnd_mu)
        rigid_shape_cfg.has_particle_collision = False
        rigid_body_ids: dict[str, int] = {}
        for rname, r in rigids.items():
            rpos = np.array(r["placement"]["pos"], dtype=np.float32)
            body_id = builder.add_body(
                xform=wp.transform(p=wp.vec3(*rpos), q=wp.quat_identity()),
                mass=rigid_mass,
                label=rname,
            )
            rigid_body_ids[rname] = body_id
            r_mesh = newton.Mesh(
                vertices=np.asarray(r["vertices"], dtype=np.float32),
                indices=np.asarray(r["triangles"], dtype=np.int32).flatten(),
                compute_inertia=False,
            )
            builder.add_shape_mesh(
                body=body_id,
                mesh=r_mesh,
                cfg=rigid_shape_cfg,
                color=rigid_color,
                label=rname,
            )

        # Use the first rigid body for the Glue / SE3 tracker.
        first_rigid_name = next(iter(rigid_body_ids.keys()))
        self._rigid_body_id = rigid_body_ids[first_rigid_name]

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -gravity))

        # Override per-particle mass so the system matrix diagonal M = density,
        # matching the scale of ke/kd contact springs (same fix as the demo example).
        _mass_np = np.full(self.model.particle_count, density, dtype=np.float32)
        self.model.particle_mass = wp.array(_mass_np, dtype=wp.float32, device=self.model.device)

        # ── Solvers ──────────────────────────────────────────────────────
        _solver_kwargs = {
            "model": self.model,
            "dt": self.sim_dt,
            "mass": density,
            "solver_type": solver_type,
            "linear_solver_maxiter": solver_iter,
            "ground_plane": (0.0, 0.0, 1.0, 0.0),
            "ground_ke": gnd_ke,
            "ground_kd": gnd_kd,
            "ground_kf": gnd_kf,
            "ground_mu": gnd_mu,
            "self_contact_ke": self_contact_ke,
            "self_contact_kd": self_contact_kd,
        }
        self.soft_solver = SolverInflatable(**_solver_kwargs, max_volume_ratio=max_pressure)
        self._pressure = 1.0
        self._target_pressure = 1.0
        self._pressure_rate = pressure_rate
        self._max_pressure = max_pressure

        # Locate the free joint that owns the rigid body (needed for joint_q teleport).
        _jchild = self.model.joint_child.numpy()
        _jqs = self.model.joint_q_start.numpy()
        _jqds = self.model.joint_qd_start.numpy()
        _rigid_jnt = int(np.where(_jchild == self._rigid_body_id)[0][0])
        self._rigid_joint_q_start = int(_jqs[_rigid_jnt])
        self._rigid_joint_qd_start = int(_jqds[_rigid_jnt])

        if self._use_mujoco:
            # MuJoCo split-step: collision handled internally, no separate pipeline.
            # update_data_interval=1 ensures joint_q teleports are synced every step.
            self.rigid_solver = SolverMuJoCo(self.model, update_data_interval=1)
            self.collision_pipeline = None
            self.contacts = None
            # Reverse-lookup MuJoCo body index from Newton body index.
            _mjc_to_newton = self.rigid_solver.mjc_body_to_newton.numpy()[0]
            self._mj_body_id = int(np.where(_mjc_to_newton == self._rigid_body_id)[0][0])
        else:
            self.rigid_solver = SolverFeatherstone(self.model)
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
            self._mj_body_id = -1
        self._n_particles = self.model.particle_count

        # ── States ────────────────────────────────────────────────────────
        self.state_0 = self.model.state()
        self.state_soft = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ── Dirichlet pins: union of all GlueSet soft_indices (with per-soft offset) ──
        rest_q = self.state_0.particle_q.numpy()
        _all_si: list[np.ndarray] = []
        for gs in glue_sets:
            off = soft_particle_start[gs["soft_name"]]
            _all_si.append(np.asarray(gs["soft_indices"], dtype=np.int32) + off)
        if not _all_si:
            raise ValueError(f"Asset at '{asset_path}' has no glue_sets.")
        pin_idx = np.unique(np.concatenate(_all_si)).astype(np.int32)
        body_com = np.array(
            rigids[first_rigid_name]["placement"]["pos"],
            dtype=np.float32,
        )
        device = self.model.device
        self._glue_indices_wp = wp.array(pin_idx, dtype=wp.int32, device=device)
        self._glue_local_offsets = wp.array(
            (rest_q[pin_idx] - body_com).astype(np.float32),
            dtype=wp.vec3,
            device=device,
        )
        self._inv_sim_dt = 1.0 / self.sim_dt

        GlueCls = GlueMuJoCo if self._use_mujoco else Glue
        self.glue = GlueCls(
            solver=self.soft_solver,
            glue_indices=self._glue_indices_wp,
            local_offsets=self._glue_local_offsets,
            coupling_gain=coupling_alpha,
            device=device,
            substep_dt=self.sim_dt,
            kd_rigid=kd_rigid,
            glue_damping=glue_damping,
        )
        self._coupling_alpha = coupling_alpha
        self._kd_rigid = kd_rigid
        cli_glued = getattr(args, "glued", None)
        self._glued = True if cli_glued is None else bool(cli_glued)
        if not self._glued:
            self.glue.toggle(False)
        print(
            f"[inflatable_glue] CFL-derived max_corr_vel = {self.glue._max_corr_vel:.2f} m/s"
            f"  (h_min/dt = {self.glue._max_corr_vel * self.sim_dt * 1e3:.2f} mm / {self.sim_dt * 1e3:.2f} ms)",
            flush=True,
        )

        # ── SE(3) manual rigid-body control ──────────────────────────────────
        self._manual_control = False
        self._pose_dirty = False
        _bq0 = self.state_0.body_q.numpy()
        _tf0 = _bq0[self._rigid_body_id]  # [px,py,pz, qx,qy,qz,qw] Warp
        _pos0 = np.array(_tf0[:3], dtype=np.float64)
        _q0 = np.array([_tf0[6], _tf0[3], _tf0[4], _tf0[5]], dtype=np.float64)
        # Body-local slider state — all relative to _base_pos/_base_quat.
        self._base_pos = _pos0.copy()
        self._base_quat = _q0.copy()
        self._dx = self._dy = self._dz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self._se3_tracker = SE3Tracker(pos=_pos0, quat=_q0)

        self._dbg_frame = 0
        self._dbg_step = 0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True

        # ── Geometry sanity check (t=0) ──────────────────────────────────
        pq0 = self.state_0.particle_q.numpy()
        bq0 = self.state_0.body_q.numpy()
        pins = self._glue_indices_wp.numpy()
        lo = self._glue_local_offsets.numpy()
        pin_mean_z = float(pq0[pins, 2].mean())
        rigid_z = float(bq0[self._rigid_body_id][2])
        _rv_local = np.asarray(rigids[first_rigid_name]["vertices"], dtype=np.float32)
        print(
            f"[glue geometry]"
            f"  soft z=[{float(pq0[:, 2].min()):.4f}, {float(pq0[:, 2].max()):.4f}]"
            f"  particle_radius={particle_r:.4f}"
            f"  pin_mean_z={pin_mean_z:.4f}  n_glue={self.glue.n_glue}"
            f"  rigid_com_z={rigid_z:.4f}"
            f"  rigid_mesh_world_z=[{rigid_z + float(_rv_local[:, 2].min()):.4f},"
            f" {rigid_z + float(_rv_local[:, 2].max()):.4f}]"
            f"  local_offset_z=[{float(lo[:, 2].min()):.4f},{float(lo[:, 2].max()):.4f}]",
            flush=True,
        )

        # Oscillation tracker: ring buffer of recent gap values → peak-to-peak metric.
        _buf_len = fps * 4  # 4 seconds of history
        self._gap_buf = [0.0] * _buf_len
        self._gap_buf_idx = 0
        self._gap_pp = 0.0  # peak-to-peak amplitude (the stability metric)
        self._soft_v_buf = [0.0] * _buf_len
        self._soft_v_rms = 0.0  # RMS particle velocity [m/s] — FEM oscillation metric
        self._soft_v_pp = 0.0  # peak-to-peak of RMS vel over 4 s window

        self.capture()

    # ------------------------------------------------------------------

    def _teleport_to(self, pos: np.ndarray, quat: np.ndarray, zero_velocities: bool) -> None:
        """Co-move soft particles by SE(3) delta and teleport the rigid body.

        Args:
            pos:              Target body COM position [3].
            quat:             Target body quaternion ``[qw, qx, qy, qz]``.
            zero_velocities:  If True, zero all rigid + particle velocities (call on
                              explicit slider change to suppress Baumgarte spike).
                              If False, only rigid-body velocity is zeroed so physics
                              does not push the body away from the held pose while
                              FEM particle dynamics continue uninterrupted.
        """
        # Clamp Z: body COM cannot go below the floor half-height.
        clamped = pos.copy()
        clamped[2] = max(clamped[2], self._rigid_hz)

        # SE(3) delta applied to particles in-place on the GPU (no D↔H copy).
        moved = self._se3_tracker.update_device(self.state_0.particle_q, clamped, quat, self.model.device)
        if moved:
            wp.copy(self.state_soft.particle_q, self.state_0.particle_q)

        # Teleport rigid body. MUST write joint_q — Featherstone reads it, not body_q.
        qw, qx, qy, qz = quat
        tf7 = np.array([clamped[0], clamped[1], clamped[2], qx, qy, qz, qw], dtype=np.float32)

        bq_np = self.state_0.body_q.numpy()
        bq_np[self._rigid_body_id] = tf7
        self.state_0.body_q.assign(bq_np)
        self.state_rigid.body_q.assign(bq_np)

        jq_np = self.state_0.joint_q.numpy()
        s = self._rigid_joint_q_start
        jq_np[s : s + 7] = tf7
        self.state_0.joint_q.assign(jq_np)
        self.state_rigid.joint_q.assign(jq_np)

        # Always zero rigid-body velocity so physics cannot push the body away.
        bqd_np = self.state_0.body_qd.numpy()
        bqd_np[self._rigid_body_id] = 0.0
        self.state_0.body_qd.assign(bqd_np)
        self.state_rigid.body_qd.assign(bqd_np)

        jqd_np = self.state_0.joint_qd.numpy()
        sd = self._rigid_joint_qd_start
        jqd_np[sd : sd + 6] = 0.0
        self.state_0.joint_qd.assign(jqd_np)
        self.state_rigid.joint_qd.assign(jqd_np)

        if zero_velocities:
            pqd_np = self.state_0.particle_qd.numpy()
            pqd_np[:] = 0.0
            self.state_0.particle_qd.assign(pqd_np)
            self.state_soft.particle_qd.assign(pqd_np)

    # ------------------------------------------------------------------

    def capture(self) -> None:
        self.graph = None
        if not wp.get_device().is_cuda:
            return
        # Restore BSR to clean (unfiltered) state and reset the filter snapshot.
        _sg = getattr(self.soft_solver, "_solver_glue", None)
        if _sg is not None and _sg._A_bsr_clean_values is not None:
            wp.copy(dest=_sg._solver.A_bsr.values, src=_sg._A_bsr_clean_values)
            _sg._A_bsr_clean_values = None
        snap_0 = self.model.state()
        snap_0.assign(self.state_0)
        snap_soft = self.model.state()
        snap_soft.assign(self.state_soft)
        snap_rigid = self.model.state()
        snap_rigid.assign(self.state_rigid)
        try:
            self.simulate()
            wp.synchronize_device()
        except Exception as exc:
            print(f"[inflatable_glue] warmup failed ({exc}); running eager.", flush=True)
            self.state_0.assign(snap_0)
            self.state_soft.assign(snap_soft)
            self.state_rigid.assign(snap_rigid)
            return
        self.state_0.assign(snap_0)
        self.state_soft.assign(snap_soft)
        self.state_rigid.assign(snap_rigid)
        try:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        except Exception as exc:
            print(f"[inflatable_glue] ScopedCapture failed ({exc}); running eager.", flush=True)
            self.state_0.assign(snap_0)
            self.state_soft.assign(snap_soft)
            self.state_rigid.assign(snap_rigid)
            self.graph = None

    # ------------------------------------------------------------------

    def simulate(self) -> None:
        """Pure-GPU substep loop — capturable as a CUDA graph (Featherstone mode only)."""
        if self._use_mujoco:
            self._simulate_mujoco()
        else:
            self._simulate_featherstone()

    def _merge_states(self) -> None:
        """Copy particle and rigid state from substep outputs back into state_0."""
        wp.copy(self.state_0.particle_q, self.state_soft.particle_q)
        wp.copy(self.state_0.particle_qd, self.state_soft.particle_qd)
        wp.copy(self.state_0.body_q, self.state_rigid.body_q)
        wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
        wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
        wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)

    def _simulate_featherstone(self) -> None:
        """Featherstone substep loop.

        Sequence each substep:
          1. glue.update_target_dv  — KKT: drive FEM pins to predicted body pose
          2. soft_solver.step       — FEM with Dirichlet pin boundary
          3. glue.apply_reaction    — elastic reaction → body_f
          4. collision_pipeline     — rigid ground contact detection
          5. rigid_solver.step      — Featherstone dynamics (gravity + contacts + body_f)
          6. Merge                  — copy particle_q/qd + body_q/qd back into state_0
        """
        for _ in range(self.sim_substeps):
            if self._glued:
                self.glue.update_target_dv(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._rigid_body_id,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self._inv_sim_dt,
                )

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.soft_solver.step(self.state_0, self.state_soft, self.control, None, self.sim_dt)

            if self._glued:
                self.glue.apply_reaction(
                    self.state_0.body_f,
                    self._rigid_body_id,
                    body_qd=self.state_0.body_qd,
                    body_q=self.state_0.body_q,
                    particle_q=self.state_soft.particle_q,
                )
            else:
                self.state_0.body_f.zero_()

            self.collision_pipeline.collide(self.state_0, self.contacts)

            self.model.particle_count = 0
            self.rigid_solver.step(self.state_0, self.state_rigid, self.control, self.contacts, self.sim_dt)
            self.model.particle_count = self._n_particles

            self._merge_states()

    def _simulate_mujoco(self) -> None:
        """MuJoCo split-step substep loop.

        Sequence each substep:
          1. rigid_solver.step_kinematics — pre-advance body to predicted pose
          2. glue.update_target_dv        — KKT target from predicted xpos/cvel
          3. soft_solver.step             — FEM with Dirichlet pin boundary
          4. xfrc_applied.zero_()         — clear external forces
          5. glue.apply_reaction           — elastic reaction → xfrc_applied
          6. rigid_solver.step_dynamics   — MuJoCo integrate (gravity + contacts + xfrc)
          7. Merge                        — copy particle_q/qd + body_q/qd back to state_0

        Implicit velocity damping comes from step_kinematics' pre-advance — no manual
        body_p_pred is needed.  Ground collision is handled by MuJoCo internally.
        """
        for _ in range(self.sim_substeps):
            # 1. Pre-advance body to predicted next-step position (provides implicit damping).
            self.rigid_solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self.sim_dt)

            # 2. KKT pin target from MuJoCo's predicted body pose.
            if self._glued:
                self.glue.update_target_dv(
                    self.rigid_solver.xpos,
                    self.rigid_solver.xquat,
                    self.rigid_solver.cvel,
                    0,
                    self._mj_body_id,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self._inv_sim_dt,
                )

            # 3. FEM step (Dirichlet pin drives pins to predicted body pose).
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.soft_solver.step(self.state_0, self.state_soft, self.control, None, self.sim_dt)

            # 4–5. Elastic reaction → xfrc_applied (MuJoCo's external force slot).
            self.rigid_solver.xfrc_applied.zero_()
            if self._glued:
                self.glue.apply_reaction(
                    self.rigid_solver.xfrc_applied,
                    0,
                    self._mj_body_id,
                    cvel=self.rigid_solver.cvel,
                    xpos=self.rigid_solver.xpos,
                    particle_q=self.state_soft.particle_q,
                )

            # 6. MuJoCo dynamics: gravity (internal) + contacts + xfrc_applied → integrate.
            self.rigid_solver.step_dynamics(self.state_rigid)

            # 7. Merge outputs back into shared state_0.
            self._merge_states()

    def step(self) -> None:
        delta = self._target_pressure - self._pressure
        if abs(delta) <= self._pressure_rate:
            self._pressure = self._target_pressure
        else:
            self._pressure += self._pressure_rate * (1.0 if delta > 0.0 else -1.0)
        self.soft_solver.set_pressure(self._pressure)

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        if self._manual_control:
            _bq = self.state_0.body_q.numpy()
            _tf = _bq[self._rigid_body_id]  # [px,py,pz, qx,qy,qz,qw]
            _phys_pos = np.array(_tf[:3], dtype=np.float64)
            _phys_quat = np.array([_tf[6], _tf[3], _tf[4], _tf[5]], dtype=np.float64)

            if self._pose_dirty:
                # Slider moved: teleport rigid + co-move particles via SE3 tracker.
                des_pos, des_quat = _desired_pose(
                    self._dx,
                    self._dy,
                    self._dz,
                    self._rx,
                    self._ry,
                    self._rz,
                    self._base_pos,
                    self._base_quat,
                )
                self._teleport_to(des_pos, des_quat, zero_velocities=True)
                self._pose_dirty = False
            else:
                # Physics running freely: update slider values to track the body.
                _q_bc = np.array([self._base_quat[0], -self._base_quat[1], -self._base_quat[2], -self._base_quat[3]])
                _local = quat_rotate(_q_bc, _phys_pos - self._base_pos)
                self._dx = float(_local[0])
                self._dy = float(_local[1])
                self._dz = float(_local[2])
                _q_rel = quat_compose(_q_bc, _phys_quat)
                _qw, _qx, _qy, _qz = float(_q_rel[0]), float(_q_rel[1]), float(_q_rel[2]), float(_q_rel[3])
                _sy = max(-1.0, min(1.0, 2.0 * (_qw * _qy - _qz * _qx)))
                self._ry = math.asin(_sy)
                self._rx = math.atan2(2.0 * (_qw * _qx + _qy * _qz), 1.0 - 2.0 * (_qx * _qx + _qy * _qy))
                self._rz = math.atan2(2.0 * (_qw * _qz + _qx * _qy), 1.0 - 2.0 * (_qy * _qy + _qz * _qz))

        self.sim_time += self.frame_dt

        # Record gap and soft-body oscillation metrics every frame.
        bq = self.state_0.body_q.numpy()
        pq = self.state_0.particle_q.numpy()
        pqd = self.state_0.particle_qd.numpy()
        pins = self._glue_indices_wp.numpy()
        body_z = float(bq[self._rigid_body_id][2])
        pin_mean_z = float(pq[pins, 2].mean())
        gap = pin_mean_z - body_z
        # RMS velocity of ALL particles — high = FEM oscillating internally.
        self._soft_v_rms = float(np.sqrt((pqd**2).sum(axis=1).mean()))

        idx = self._gap_buf_idx % len(self._gap_buf)
        self._gap_buf[idx] = gap
        self._soft_v_buf[idx] = self._soft_v_rms
        self._gap_buf_idx += 1
        if self._gap_buf_idx >= len(self._gap_buf):
            self._gap_pp = max(self._gap_buf) - min(self._gap_buf)
            self._soft_v_pp = max(self._soft_v_buf) - min(self._soft_v_buf)

        # Debug: print every 2 s of sim time.
        self._dbg_frame += 1
        if self._dbg_frame % (self.fps * 2) == 0:
            print(
                f"[glue t={self.sim_time:.1f}s]"
                f"  rigid_z={body_z:.4f}  pin_mean_z={pin_mean_z:.4f}"
                f"  gap={gap:.4f}  gap_pp={self._gap_pp:.4f}"
                f"  soft_v_rms={self._soft_v_rms:.4f}  soft_v_pp={self._soft_v_pp:.4f}"
                f"  glued={self._glued}",
                flush=True,
            )

    # ------------------------------------------------------------------

    def gui(self, ui) -> None:
        ui.text(f"rigid solver: {self._rigid_solver_type}")

        changed, glued = ui.checkbox("glued##glue", self._glued)
        if changed:
            self._glued = bool(glued)
            self.glue.toggle(self._glued)
            if not self._glued:
                self.state_0.body_f.zero_()
            self.capture()

        ui.text(f"pinned particles: {self.glue.n_glue}")
        ui.text(f"gap peak-to-peak:    {self._gap_pp:.4f} m  (rigid↔FEM, lower = better)")
        ui.text(f"soft v_rms:          {self._soft_v_rms:.4f} m/s  (FEM kinetic energy)")
        ui.text(f"soft v_rms peak-peak:{self._soft_v_pp:.4f} m/s  (FEM oscillation amplitude)")

        alpha_changed, new_alpha = ui.slider_float("alpha##coupling", self._coupling_alpha, 0.001, 0.5)
        if alpha_changed:
            self._coupling_alpha = float(new_alpha)
            self.glue._coupling_gain = self._coupling_alpha
            self.capture()

        kd_changed, new_kd = ui.slider_float("kd rigid##coupling", self._kd_rigid, 0.0, 30.0)
        if kd_changed:
            self._kd_rigid = float(new_kd)
            self.glue._kd_rigid = self._kd_rigid
            self.capture()

        _, self._target_pressure = ui.slider_float("pressure##inflate", self._target_pressure, 1.0, self._max_pressure)
        ui.text(f"pressure: {self._pressure:.3f} / {self._target_pressure:.3f}")

        ui.text("---")
        mc_changed, mc_val = ui.checkbox("manual SE3 control##se3", self._manual_control)
        if mc_changed:
            self._manual_control = bool(mc_val)
            if self._manual_control:
                # On enable: snapshot current body pose; all sliders start at zero.
                _bq = self.state_0.body_q.numpy()
                _tf = _bq[self._rigid_body_id]
                _p = np.array(_tf[:3], dtype=np.float64)
                _q = np.array([_tf[6], _tf[3], _tf[4], _tf[5]], dtype=np.float64)
                self._base_pos = _p.copy()
                self._base_quat = _q.copy()
                self._dx = self._dy = self._dz = 0.0
                self._rx = self._ry = self._rz = 0.0
                self._se3_tracker = SE3Tracker(pos=_p, quat=_q)
        if self._manual_control:
            pxc, dx = ui.slider_float("pos X##se3", self._dx, -1.0, 1.0)
            pyc, dy = ui.slider_float("pos Y##se3", self._dy, -1.0, 1.0)
            pzc, dz = ui.slider_float("pos Z##se3", self._dz, -1.0, 1.0)
            rxc, rx_deg = ui.slider_float("rot X [°]##se3", math.degrees(self._rx), -180.0, 180.0)
            ryc, ry_deg = ui.slider_float("rot Y [°]##se3", math.degrees(self._ry), -180.0, 180.0)
            rzc, rz_deg = ui.slider_float("rot Z [°]##se3", math.degrees(self._rz), -180.0, 180.0)
            if pxc or pyc or pzc or rxc or ryc or rzc:
                self._dx = float(dx)
                self._dy = float(dy)
                self._dz = float(dz)
                self._rx = math.radians(rx_deg)
                self._ry = math.radians(ry_deg)
                self._rz = math.radians(rz_deg)
                self._pose_dirty = True

    # ------------------------------------------------------------------

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self._glued:
            pin_pos_wp = self.glue.get_pin_positions(self.state_0.particle_q)
            tgt_pos_wp = self.glue.get_target_positions(self.state_0.body_q, self._rigid_body_id)
            pin_np = pin_pos_wp.numpy()
            tgt_np = tgt_pos_wp.numpy()
            n = len(pin_np)

            # Green→yellow→red gradient by constraint violation.
            violations = np.linalg.norm(pin_np - tgt_np, axis=1)
            t = np.clip(violations / 0.01, 0.0, 1.0)  # 1 cm = fully red
            colors_np = np.stack([t, 1.0 - t, np.zeros(n)], axis=1).astype(np.float32)
            colors_wp = wp.array(colors_np, dtype=wp.vec3, device=self.model.device)

            r_arr = wp.array(
                np.full(n, self._particle_r * 1.4, dtype=np.float32),
                dtype=wp.float32,
                device=self.model.device,
            )
            magenta = wp.array(
                np.tile([1.0, 0.0, 1.0], (n, 1)).astype(np.float32), dtype=wp.vec3, device=self.model.device
            )
            green = wp.array(
                np.tile([0.0, 1.0, 0.4], (n, 1)).astype(np.float32), dtype=wp.vec3, device=self.model.device
            )
            self.viewer.log_lines(
                "glue/constraints",
                pin_pos_wp,
                tgt_pos_wp,
                colors_wp,
                width=0.004,
            )
            self.viewer.log_points("glue/particles", pin_pos_wp, radii=r_arr, colors=magenta)
            self.viewer.log_points("glue/targets", tgt_pos_wp, radii=r_arr, colors=green)

        self.viewer.end_frame()

    # ------------------------------------------------------------------

    def test_post_step(self) -> None:
        self._dbg_step += 1
        if self._dbg_step % self.fps != 0:
            return
        bq = self.state_0.body_q.numpy()
        pq = self.state_0.particle_q.numpy()
        pins = self._glue_indices_wp.numpy()
        tdv = self.glue._target_dv.numpy()
        body_z = float(bq[self._rigid_body_id][2])
        pin_mean_z = float(pq[pins, 2].mean())
        tdv_mean_z = float(tdv[pins, 2].mean())
        print(
            f"[glue t={self.sim_time:.1f}s]  "
            f"rigid_z={body_z:.4f}  pin_mean_z={pin_mean_z:.4f}  "
            f"gap={pin_mean_z - body_z:.4f}  tdv_z={tdv_mean_z:.4f}",
            flush=True,
        )

    def test_final(self) -> None:
        p_lower = wp.vec3(-5.0, -5.0, -0.5)
        p_upper = wp.vec3(5.0, 5.0, 10.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles stay in workspace",
            lambda q, _: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )


# ---------------------------------------------------------------------------
# _ShowcaseExample — three groups side by side with 7 pressure chambers
# ---------------------------------------------------------------------------


class _ShowcaseExample(Example):
    """Showcase placing tet objects, a glue assembly, and hex objects side by side.

    Chamber layout (shared int32 id space):
        0 = tet box       (X = -4 m)
        1 = tet sphere    (X = -4 m)
        2 = tet baymax    (X = -4 m)
        3 = glue soft     (X =  0 m)
        4 = hex cube      (X = +4 m)
        5 = hex tire      (X = +4 m)
        6 = hex pyramid   (X = +4 m)
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        max_p = args.max_pressure

        # Parse per-object config; filter by "active", auto-detect type
        if not args.objects:
            raise ValueError("--objects is required for showcase mode. Set it in docker/config/inflatable.json.")
        all_objects = json.loads(args.objects)
        tet_objs, glue_objs, hex_objs = [], [], []
        for o in all_objects:
            if not o.get("active", True):
                continue
            if "path" not in o:
                raise ValueError(f"Object entry missing 'path': {o}")
            path = _resolve_path(o["path"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Asset not found: {path}")
            kind = o.get("type") or detect_asset_type(path)
            name = o.get("name") or os.path.splitext(os.path.basename(path))[0]
            obj = {**o, "type": kind, "name": name, "_resolved_path": path}
            if kind == "tet":
                tet_objs.append(obj)
            elif kind == "hex":
                hex_objs.append(obj)
            elif kind == "glue":
                glue_objs.append(obj)
            else:
                raise ValueError(f"Unknown asset type '{kind}' for {path}")

        if not tet_objs and not glue_objs and not hex_objs:
            raise ValueError('No active objects in --objects. Set at least one entry to "active": true')

        # ----- load tet assets -----
        tet_assets = [load_asset(o["_resolved_path"]) for o in tet_objs]

        # ----- load hex assets -----
        hex_objects = []
        for o in hex_objs:
            ho = load_hex_assembly(o["_resolved_path"])
            if not ho:
                raise RuntimeError(f"No hex mesh prim in {o['_resolved_path']}")
            hex_objects.append(ho[0])

        # ----- load glue assembly (optional) -----
        glue_soft = glue_rigid = glue_set = None
        if glue_objs:
            glue_path = getattr(args, "asset", None) or glue_objs[0]["_resolved_path"]
            glue_assembly = load_glued_assembly(glue_path)
            glue_soft = glue_assembly["softs"][glue_assembly["soft_names"][0]]
            glue_rigid = glue_assembly["rigids"][glue_assembly["rigid_names"][0]]
            glue_set = glue_assembly["glue_sets"][0]

        # ----- build unified model -----
        builder = newton.ModelBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=args.ground_ke,
                kd=args.ground_kd,
                kf=args.ground_kf,
                mu=args.ground_mu,
            )
        )

        # --- Tet objects at X = -4 m ---
        tet_ranges = []
        tet_particle_ranges = []
        tet_surface_tris = []
        # (name, tri_start, tri_end, color) for per-object mesh coloring
        _mesh_color_ranges: list[tuple[str, int, int, tuple]] = []
        for asset, obj in zip(tet_assets, tet_objs, strict=False):
            p0 = builder.particle_count
            t0 = builder.tet_count
            tri0 = builder.tri_count
            builder.add_soft_mesh(
                pos=wp.vec3(*[float(v) for v in obj["position"]]),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                scale=1.0,
                vertices=asset["vertices"],
                indices=asset["tets"].reshape(-1).tolist(),
                density=float(obj.get("density", 1.0)),
                k_mu=float(obj.get("k-mu", 1e5)),
                k_lambda=float(obj.get("k-lambda", 1e5)),
                k_damp=float(obj.get("k-damp", 1.0)),
                particle_radius=float(obj.get("particle-radius", 0.008)),
            )
            t1 = builder.tet_count
            p1 = builder.particle_count
            tet_ranges.append((t0, t1))
            tet_particle_ranges.append((p0, p1 - p0))
            surf_tris = asset.get("surface_triangles")
            tet_surface_tris.append(surf_tris)
            # Add surface triangles to builder so they appear in model.tri_indices
            if surf_tris is not None:
                _add_surface_tris(builder, surf_tris, p0)
            c = obj.get("color", [0.7, 0.6, 0.4])
            _mesh_color_ranges.append(
                (
                    obj.get("name", f"tet_{len(tet_ranges) - 1}"),
                    tri0,
                    builder.tri_count,
                    (float(c[0]), float(c[1]), float(c[2])),
                )
            )

        # --- Glue assembly at X = 0 m (only if active) ---
        go = glue_objs[0] if glue_objs else None
        glue_p0 = glue_t0 = glue_t1 = glue_p1 = 0
        glue_rigid_body_id = -1
        if go and glue_soft is not None:
            glue_p0 = builder.particle_count
            glue_t0 = builder.tet_count
            _glue_tri0 = builder.tri_count
            builder.add_soft_mesh(
                pos=wp.vec3(*[float(v) for v in go["position"]]),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                scale=1.0,
                vertices=glue_soft["vertices"],
                indices=np.asarray(glue_soft["tets"], dtype=np.int32).reshape(-1).tolist(),
                density=float(go.get("density", 1.0)),
                k_mu=float(go.get("k-mu", 1e5)),
                k_lambda=float(go.get("k-lambda", 1e5)),
                k_damp=float(go.get("k-damp", 200.0)),
                particle_radius=float(go.get("particle-radius", 0.008)),
            )
            glue_t1 = builder.tet_count
            glue_p1 = builder.particle_count
            _glue_surf = glue_soft.get("surface_triangles")
            if _glue_surf is not None:
                _add_surface_tris(builder, _glue_surf, glue_p0)
            gc = go.get("color", [0.85, 0.45, 0.10])
            _mesh_color_ranges.append(
                (go.get("name", "glue"), _glue_tri0, builder.tri_count, (float(gc[0]), float(gc[1]), float(gc[2])))
            )

            # Add rigid hub
            glue_rigid_body_id = builder.body_count
            glue_rigid_placement = glue_rigid.get("placement", {})
            rigid_pos = glue_rigid_placement.get("pos", (0.0, 0.0, 0.8))
            rigid_quat = glue_rigid_placement.get("quat", (0.0, 0.0, 0.0, 1.0))
            builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(float(rigid_pos[0]), float(rigid_pos[1]), float(rigid_pos[2])),
                    q=wp.quat(float(rigid_quat[0]), float(rigid_quat[1]), float(rigid_quat[2]), float(rigid_quat[3])),
                ),
                mass=args.rigid_mass,
            )
            rigid_shape_cfg = newton.ModelBuilder.ShapeConfig(
                ke=args.ground_ke,
                kd=args.ground_kd,
                kf=args.ground_kf,
                mu=args.ground_mu,
            )
            rigid_shape_cfg.has_particle_collision = False
            r_mesh = newton.Mesh(
                vertices=np.asarray(glue_rigid["vertices"], dtype=np.float32),
                indices=np.asarray(glue_rigid["triangles"], dtype=np.int32).flatten(),
                compute_inertia=False,
            )
            builder.add_shape_mesh(
                body=glue_rigid_body_id,
                mesh=r_mesh,
                cfg=rigid_shape_cfg,
            )

        # --- Hex objects at X = +4 m ---
        _m_floor = 10.0 * self.sim_dt * 1.5
        hex_ranges = []
        hex_particle_ranges = []
        hex_surface_tris = []
        hex_cursor = builder.hex_count

        hex_usda_by_name = {o["name"]: o for o in hex_objects}
        ordered_hex_usd = [hex_usda_by_name.get(o["name"], hex_objects[i]) for i, o in enumerate(hex_objs)]

        for i, (cfg, usd_obj) in enumerate(zip(hex_objs, ordered_hex_usd, strict=False)):
            if "position" not in cfg:
                raise ValueError(f"Hex object '{cfg.get('name', cfg['path'])}' is missing 'position'")
            pos = cfg["position"]
            verts = usd_obj["vertices"].copy()
            # Apply X and Y from position; Z: if specified use it as floor, else place on ground
            verts[:, 0] += float(pos[0]) - verts[:, 0].mean()
            verts[:, 1] += float(pos[1]) - verts[:, 1].mean()
            # z in position is the floor height (bottom of mesh). 0.0 = rest on ground.
            verts[:, 2] += float(pos[2]) - verts[:, 2].min()
            hexes = usd_obj["hex_indices"]
            local_tris = usd_obj["surface_triangles"]
            H_obj = hexes.shape[0]
            n_verts = verts.shape[0]
            obj_density = float(cfg.get("density", 1000.0))

            node_mass = np.zeros(n_verts, dtype=np.float64)
            for e in range(H_obj):
                n0, n6 = int(hexes[e, 0]), int(hexes[e, 6])
                dx = abs(float(verts[n6, 0]) - float(verts[n0, 0]))
                dy = abs(float(verts[n6, 1]) - float(verts[n0, 1]))
                dz = abs(float(verts[n6, 2]) - float(verts[n0, 2]))
                em = obj_density * max(dx * dy * dz, 1e-10)
                for a in range(8):
                    node_mass[int(hexes[e, a])] += em / 8.0

            p_off = builder.particle_count
            for vi in range(n_verts):
                pos = verts[vi]
                builder.add_particle(
                    pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    mass=float(max(node_mass[vi], _m_floor)),
                    radius=float(cfg.get("particle-radius", 0.008)),
                )
            shifted_hexes = hexes + p_off
            builder.add_soft_hex_mesh(
                hex_indices=shifted_hexes,
                k_mu=float(cfg.get("k-mu", 5e5)),
                k_lambda=float(cfg.get("k-lambda", 5e5)),
                k_damp=float(cfg.get("k-damp", 0.01)),
            )
            _hex_tri0 = builder.tri_count
            # Flip winding so outward normals face the viewer (CCW convention).
            _add_surface_tris(builder, local_tris[:, [0, 2, 1]], p_off)
            hc = cfg.get("color", [0.5, 0.5, 0.5])
            _mesh_color_ranges.append(
                (cfg.get("name", f"hex_{i}"), _hex_tri0, builder.tri_count, (float(hc[0]), float(hc[1]), float(hc[2])))
            )
            hex_particle_ranges.append((p_off, n_verts))
            hex_surface_tris.append(local_tris)
            hex_ranges.append((hex_cursor, hex_cursor + H_obj))
            hex_cursor += H_obj

        model = builder.finalize()
        model.set_gravity((0.0, 0.0, -float(getattr(args, "gravity", 9.81))))
        self.model = model

        # ----- Chamber masks (only for active element types) -----
        # Tet chambers: 0..N-1 for tet objs, then N for glue (if active)
        n_tet_chambers = len(tet_ranges)
        glue_chamber_id = n_tet_chambers if glue_objs else -1
        n_hex_base = n_tet_chambers + (1 if glue_objs else 0)
        total_chambers = n_hex_base + len(hex_ranges)

        tet_mask_np = np.full(model.tet_count, -1, dtype=np.int32) if model.tet_count > 0 else None
        if tet_mask_np is not None:
            for ch, (t0, t1) in enumerate(tet_ranges):
                tet_mask_np[t0:t1] = ch
            if glue_objs and glue_t1 > glue_t0:
                tet_mask_np[glue_t0:glue_t1] = glue_chamber_id

        hex_mask_np = np.full(model.hex_count, -1, dtype=np.int32) if model.hex_count > 0 else None
        if hex_mask_np is not None:
            for i, (h0, h1) in enumerate(hex_ranges):
                hex_mask_np[h0:h1] = n_hex_base + i

        # ----- Create solver -----
        self.solver = SolverInflatable(
            model=model,
            dt=self.sim_dt,
            max_volume_ratio=max_p,
            preconditioner_type=args.preconditioner,
            solver_type=args.solver_type,
            linear_solver_maxiter=args.solver_maxiter,
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=args.ground_ke,
            ground_kd=args.ground_kd,
            ground_kf=args.ground_kf,
            ground_mu=args.ground_mu,
            self_contact_ke=args.contact_ke,
            self_contact_kd=args.contact_kd,
            linear_damping=args.linear_damping,
        )

        # Surface contacts for all active particle groups
        glue_surface_tris = [glue_soft.get("surface_triangles")] if glue_soft else []
        glue_p_ranges = [(glue_p0, glue_p1 - glue_p0)] if glue_objs else []
        all_particle_ranges = tet_particle_ranges + glue_p_ranges + hex_particle_ranges
        all_surface_tris = tet_surface_tris + glue_surface_tris + hex_surface_tris
        # ke <= 0 means inter-body contact is off: skip the BVH build/refit and
        # the per-substep surface-contact kernel entirely rather than computing
        # forces that are identically zero (~5.7% of GPU time in the showcase).
        if all_particle_ranges and args.contact_ke > 0.0:
            self.solver.set_soft_surface_contacts(
                particle_ranges=all_particle_ranges,
                surface_triangles_list=all_surface_tris,
                ke=args.contact_ke,
                kd=args.contact_kd,
            )

        if total_chambers > 0:
            self.solver.set_chamber_mask(
                tet_chamber_mask=wp.array(tet_mask_np, dtype=wp.int32, device=model.device)
                if tet_mask_np is not None
                else None,
                hex_chamber_mask=wp.array(hex_mask_np, dtype=wp.int32, device=model.device)
                if hex_mask_np is not None
                else None,
                num_chambers=total_chambers,
            )

        # ----- Rigid solver + Glue coupling (only if glue active) -----
        self.rigid_solver = None
        self.collision_pipeline = None
        self.contacts = None
        self.glue = None
        self._glue_rigid_body_id = -1

        # ----- States -----
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.state_rigid = model.state()
        self.control = model.control()
        if model.body_count > 0:
            newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)

        # ----- Debug -----
        self._debug_interval = getattr(args, "debug_interval", 0)
        self._frame_idx = 0
        # Store per-object particle index ranges for per-object reporting
        self._particle_ranges_labeled = (
            [
                (o.get("name", f"tet_{i}"), p0, p0 + n)
                for i, (o, (p0, n)) in enumerate(zip(tet_objs, tet_particle_ranges, strict=False))
            ]
            + ([(glue_objs[0].get("name", "glue"), glue_p0, glue_p1)] if glue_objs and glue_p1 > glue_p0 else [])
            + [
                (o.get("name", f"hex_{i}"), p0, p0 + n)
                for i, (o, (p0, n)) in enumerate(zip(hex_objs, hex_particle_ranges, strict=False))
            ]
        )

        if glue_objs and glue_set is not None:
            self.rigid_solver = SolverFeatherstone(model)
            self.collision_pipeline = newton.CollisionPipeline(model)
            self.contacts = self.collision_pipeline.contacts()
            self._glue_rigid_body_id = glue_rigid_body_id
            glue_pins = np.asarray(glue_set["soft_indices"], dtype=np.int32) + glue_p0
            rest_q = self.state_0.particle_q.numpy()
            local_offsets = rest_q[glue_pins].astype(np.float32) - np.array(rigid_pos, dtype=np.float32)
            self.glue = Glue(
                solver=self.solver,
                glue_indices=wp.array(glue_pins, dtype=wp.int32, device=model.device),
                local_offsets=wp.array(local_offsets, dtype=wp.vec3, device=model.device),
                coupling_gain=args.coupling_alpha,
                device=model.device,
                substep_dt=self.sim_dt,
                kd_rigid=args.coupling_kd_rigid,
                glue_damping=args.glue_damping,
            )
        self._use_glue = bool(glue_objs) and self.glue is not None

        # ----- Per-object render meshes -----
        # Build each object's triangle index buffer ONCE as an owned GPU array.
        # Slicing a per-frame temporary here would hand the viewer views that
        # dangle as soon as the temporary is freed (CUDA 700 on the next frame).
        self._render_meshes: list[tuple[str, wp.array, tuple]] = []
        if model.tri_count > 0 and model.tri_indices is not None:
            _tri_np = model.tri_indices.numpy().reshape(-1)
            for name, tri0, tri1, color in _mesh_color_ranges:
                if tri1 <= tri0:
                    continue
                self._render_meshes.append(
                    (
                        name,
                        wp.array(_tri_np[tri0 * 3 : tri1 * 3].copy(), dtype=wp.int32, device=model.device),
                        color,
                    )
                )
        # Suppress the viewer's default single-color unified triangle mesh;
        # render() draws per-object colored meshes instead.
        self.viewer.show_triangles = False
        # The GL viewer's spotlight is camera-anchored with a fixed 30/45 degree
        # cone, so objects at the ends of the showcase row fall outside it and
        # render with only the (very dim) ambient term, i.e. black.
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.spotlight_enabled = False
        self._n_particles = model.particle_count

        # ----- Pressure state -----
        # Build ordered label list matching chamber IDs so gui() can iterate generically
        # Entry: (label, chamber_id, rate, group)
        self._chamber_labels: list[tuple[str, int, float, str]] = []
        for i, o in enumerate(tet_objs):
            name = o.get("name", f"tet_{i}")
            self._chamber_labels.append((name, i, 0.1, "tet"))
        if glue_objs:
            go_name = (glue_objs[0].get("name") or "glue").removeprefix("tet_")
            self._chamber_labels.append((go_name, glue_chamber_id, 0.1, "glue"))
        for i, o in enumerate(hex_objs):
            name = o.get("name", f"hex_{i}")
            self._chamber_labels.append((name, n_hex_base + i, 0.04, "hex"))

        self._target_pressures = [1.0] * total_chambers
        self._current_pressures = [1.0] * total_chambers
        self._max_pressure = float(max_p)
        self._pressure_rates = [entry[2] for entry in self._chamber_labels]

        self.solver.linear_solver_check_every = int(getattr(args, "cg_check_every", 0))

        self.graph = None
        self.viewer.set_model(self.model)
        self.capture()

    def capture(self) -> None:
        """Capture the FEM substep loop as a CUDA graph.

        Only ``simulate()`` is captured — ``_rebuild_element_blocks``
        (``bsr_set_from_triplets``) runs outside the graph each frame.
        Skipped when the rigid glue solver is active.
        """
        self.graph = None
        if not wp.get_device().is_cuda:
            return
        if self.rigid_solver is not None:
            print("   [showcase] rigid solver active — skipping graph capture.", flush=True)
            return
        if self.solver.linear_solver_check_every > 0:
            print(
                "   [showcase] --cg-check-every > 0 — host-side residual checks are not capturable; running eager.",
                flush=True,
            )
            return
        try:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
            print("   [showcase] CUDA graph captured.", flush=True)
        except Exception as exc:
            print(f"   [showcase] ScopedCapture failed: {exc}", flush=True)
            self.graph = None

    def simulate(self) -> None:
        """Substep loop: ping-pong state_0 ↔ state_1 for FEM, state_rigid for rigid hub."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self._use_glue:
                self.glue.update_target_dv(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._glue_rigid_body_id,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    1.0 / self.sim_dt,
                )

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            if self.rigid_solver is not None:
                if self._use_glue:
                    self.state_0.body_f.zero_()
                    self.glue.apply_reaction(
                        self.state_0.body_f,
                        self._glue_rigid_body_id,
                        self.state_0.body_qd,
                        self.state_0.body_q,
                        particle_q=self.state_1.particle_q,
                    )
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.model.particle_count = 0
                self.rigid_solver.step(self.state_0, self.state_rigid, self.control, self.contacts, self.sim_dt)
                self.model.particle_count = self._n_particles
                wp.copy(self.state_1.body_q, self.state_rigid.body_q)
                wp.copy(self.state_1.body_qd, self.state_rigid.body_qd)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def _dump_debug(self) -> None:
        """Print per-object particle bounding box, max velocity, and NaN count."""
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        cg_n = self.solver.last_cg_iters
        cg_txt = f"  cg_iters={cg_n}/{self.solver.linear_solver_maxiter}" if cg_n is not None else ""
        print(f"\n[showcase frame {self._frame_idx}  t={self.sim_time:.3f}s]{cg_txt}")
        for name, p0, p1 in self._particle_ranges_labeled:
            pos = q[p0:p1]
            vel = qd[p0:p1]
            nan_pos = int(np.isnan(pos).any(axis=1).sum())
            nan_vel = int(np.isnan(vel).any(axis=1).sum())
            if nan_pos > 0 or nan_vel > 0:
                print(f"  {name:20s}  NaN pos={nan_pos}/{p1 - p0}  NaN vel={nan_vel}/{p1 - p0}  *** DIVERGED ***")
            else:
                lo = pos.min(axis=0)
                hi = pos.max(axis=0)
                vmax = float(np.linalg.norm(vel, axis=1).max()) if len(vel) else 0.0
                print(f"  {name:20s}  z=[{lo[2]:.3f},{hi[2]:.3f}]  vmax={vmax:.3f} m/s")

    def step(self) -> None:
        for i in range(len(self._current_pressures)):
            delta = self._target_pressures[i] - self._current_pressures[i]
            rate = self._pressure_rates[i]
            if abs(delta) <= rate:
                self._current_pressures[i] = self._target_pressures[i]
            else:
                self._current_pressures[i] += math.copysign(rate, delta)
        self.solver.set_chamber_pressures(self._current_pressures)

        # Rebuild Hessian blocks + BSR assembly (must stay outside the graph —
        # bsr_set_from_triplets uses CUB temp alloc that can't be captured).
        # Skip preconditioner rebuild when graph is active: preconditioner()
        # allocates a new inv-diag buffer, which would invalidate the captured
        # graph's pointer.  The graph carries the preconditioner from capture.
        self.solver._rebuild_element_blocks(
            self.state_0.particle_q,
            self.sim_dt,
            update_preconditioner=(self.graph is None),
        )

        if self.graph is not None:
            wp.capture_launch(self.graph)
            if self.sim_substeps % 2 == 1:
                self.state_0, self.state_1 = self.state_1, self.state_0
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frame_idx += 1
        if self._debug_interval > 0 and self._frame_idx % self._debug_interval == 0:
            self._dump_debug()

    def gui(self, ui) -> None:
        current_group = None
        for label, ch_id, _rate, group in self._chamber_labels:
            if group != current_group:
                current_group = group
                if group == "tet":
                    ui.text("-- Tet objects --")
                elif group == "glue":
                    ui.text("-- Glue assembly --")
                elif group == "hex":
                    ui.text("-- Hex objects --")

            changed, val = ui.slider_float(
                f"pressure {label}##{group}{ch_id}",
                self._current_pressures[ch_id],
                1.0,
                self._max_pressure,
            )
            if changed:
                self._target_pressures[ch_id] = val

            if group == "glue":
                glued_changed, new_glued = ui.checkbox("glued##glue", self._use_glue)
                if glued_changed and self.glue is not None:
                    self._use_glue = new_glued
                    self.glue.toggle(self._use_glue)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Draw per-object colored meshes (show_triangles=False suppresses the default).
        for name, indices, color in self._render_meshes:
            self.viewer.log_mesh(
                f"/model/mesh/{name}",
                points=self.state_0.particle_q,
                indices=indices,
                color=color,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def test_final(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
