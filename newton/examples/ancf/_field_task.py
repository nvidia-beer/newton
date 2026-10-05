# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Terrain loader shared by the Newton vehicle examples and the Isaac Lab environment.

:class:`FieldTerrain` loads a ``newton_terrain_tool.field/2`` bundle - the one format of every
terrain asset: the synthetic ``boulders`` / ``craters`` of newton-terrain-tool and the RELLIS-3D
reconstructions of newton-rellis-3d-tool. The stage weight ``w`` scales the one baked layer
(``heights = w * I_N``), which is how the relief is changed at run time. A bundle may carry a
:class:`ReferenceTrack` (``<name>_reference_track.json``): the drive line as vehicle poses in the
scene frame - a baked corridor centreline, the flower manoeuvre or a recorded drive - whose pose
0 is where the vehicle spawns. Host side, NumPy only (PIL for the PNG).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from newton.examples.ancf._terrain_common import (
    triangulated_sample,
)

REFERENCE_TRACK_FORMATS = ("rellis.reference_track/1", "newton_terrain_tool.reference_track/1")


class ReferenceTrack:
    """The drive line of a terrain bundle, as geometry the examples steer by.

    The recorded / baked poses are kept as they are (``poses``, ``yaw``, ``ground``, display
    ``segments``); the steering geometry is the plan-view polyline resampled at :attr:`SPACING`
    with arc length ``s``, unit tangent and a smoothed signed curvature (+ = bending left), so a
    0.1 s LiDAR pose log and a 0.5 m baked centreline give the same kind of answers. ``closed``
    tracks wrap; open ones clamp at their ends.
    """

    SPACING = 0.5
    """Resampling step of the steering polyline [m]."""
    CURVATURE_WINDOW = 3.0
    """Box-filter length for the signed curvature [m] (recorded tracks are noisy at 0.1 m)."""
    TURN_MIN_LENGTH = 5.0
    """A stretch bending one way shorter than this is not a turn of its own [m]."""
    TURN_MIN_RADIUS = 50.0
    """Bends of larger radius count as straight [m]."""

    def __init__(self, data: dict):
        if data.get("format") not in REFERENCE_TRACK_FORMATS or data.get("frame") != "scene":
            raise ValueError(f"unsupported reference track format {data.get('format')!r} / frame {data.get('frame')!r}")
        self.closed = bool(data.get("closed", False))
        self.poses = np.asarray(data["position_m"], dtype=np.float32).reshape(-1, 3)
        self.yaw = np.asarray(data["yaw_rad"], dtype=np.float64).reshape(-1)
        self.ground = np.asarray(data["ground_center_m"], dtype=np.float32).reshape(-1, 3)
        self.segments = np.asarray(data.get("segments", []), dtype=np.int32).reshape(-1, 2)
        if not np.isfinite(self.poses).all() or len(self.poses) != len(self.yaw) or len(self.poses) < 2:
            raise ValueError("reference track needs >= 2 finite poses with a yaw each")
        if self.segments.size and (self.segments.min() < 0 or self.segments.max() >= len(self.poses)):
            raise ValueError("reference track segments index missing poses")
        # steering polyline: dedupe standing poses, resample at SPACING
        xy = self.poses[:, :2].astype(np.float64)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3])
        xy = xy[keep]
        if self.closed and np.linalg.norm(xy[-1] - xy[0]) > 1e-3:
            xy = np.vstack([xy, xy[:1]])
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(cum[-1])
        n = max(4, int(self.length // self.SPACING))
        s = np.arange(n) * (self.length / n) if self.closed else np.linspace(0.0, self.length, n)
        self.xy = np.stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])], axis=1)
        self.s = s
        self._ds = float(self.length / n) if self.closed else float(self.length / (n - 1))
        if self.closed:
            d = np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0)
        else:
            d = np.gradient(self.xy, axis=0)
        self.tangent = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        dpsi = (np.diff(heading, append=heading[:1] if self.closed else heading[-1:]) + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        kappa = dpsi / self._ds
        k = max(1, int(round(self.CURVATURE_WINDOW / self._ds)))
        box = np.ones(k) / k
        if self.closed:
            pad = np.concatenate([kappa[-k:], kappa, kappa[:k]])
            self.curvature = np.convolve(pad, box, mode="same")[k:-k]
        else:
            self.curvature = np.convolve(kappa, box, mode="same")
        self._turn_of = self._split_turns()

    # ── spawn ──────────────────────────────────────────────────────────────────

    @property
    def start_pose(self) -> tuple[float, float, float]:
        """Pose 0 as ``(x, y, yaw)`` [m, m, rad]: where the vehicle starts."""
        return float(self.poses[0, 0]), float(self.poses[0, 1]), float(self.yaw[0])

    # ── steering geometry ──────────────────────────────────────────────────────

    def wrap(self, s: float) -> float:
        """Arc length brought into the track: modulo the length when closed, clamped when open."""
        return s % self.length if self.closed else min(max(s, 0.0), self.length)

    def _index(self, s: float) -> int:
        return int(min(round(self.wrap(s) / self._ds), len(self.xy) - 1)) % len(self.xy)

    def nearest(
        self, x: float, y: float, s_hint: float | None = None, window: float = 15.0
    ) -> tuple[int, float, float]:
        """Nearest polyline sample to (x, y): ``(index, s, lateral)``, lateral + = left of the track.

        With ``s_hint`` (the arc length found last time) only samples within ``window`` [m] of it are
        considered, so a track that runs back over itself (a recorded out-and-back drive) cannot
        snap the vehicle onto another pass of the same road.
        """
        d = self.xy - (x, y)
        d2 = np.einsum("ij,ij->i", d, d)
        if s_hint is not None:
            ds = np.abs(self.s - s_hint)
            if self.closed:
                ds = np.minimum(ds, self.length - ds)
            d2 = np.where(ds <= window, d2, np.inf)
        i = int(np.argmin(d2))
        t = self.tangent[i]
        lateral = float(-(x - self.xy[i, 0]) * t[1] + (y - self.xy[i, 1]) * t[0])
        return i, float(self.s[i]), lateral

    def point_at(self, s: float) -> np.ndarray:
        """Track point (3,) at arc length ``s`` (z = 0; the callers drape on the terrain)."""
        p = self.xy[self._index(s)]
        return np.array([p[0], p[1], 0.0], dtype=np.float64)

    def tangent_at(self, s: float) -> np.ndarray:
        return self.tangent[self._index(s)]

    def curvature_at(self, s: float) -> float:
        """Unsigned curvature [1/m] at ``s`` (the drivers' speed cap)."""
        return float(abs(self.curvature[self._index(s)]))

    # ── turns ──────────────────────────────────────────────────────────────────

    def _split_turns(self) -> np.ndarray:
        """Turn id per sample (-1 = straight): a turn is a stretch bending one way for at least
        TURN_MIN_LENGTH with a radius under TURN_MIN_RADIUS; shorter stretches join the previous
        turn, and on a closed track the two pieces at the seam are one turn."""
        sign = np.sign(self.curvature) * (np.abs(self.curvature) > 1.0 / self.TURN_MIN_RADIUS)
        n = len(sign)
        # run-length encode
        starts = np.concatenate([[0], np.flatnonzero(np.diff(sign) != 0) + 1])
        ends = np.concatenate([starts[1:], [n]])
        runs = [[int(a), int(b), float(sign[a])] for a, b in zip(starts, ends, strict=True)]
        # merge stretches shorter than the minimum turn length into the previous run
        min_n = int(round(self.TURN_MIN_LENGTH / self._ds))
        merged: list[list] = []
        for r in runs:
            if merged and r[1] - r[0] < min_n:  # too short to be a turn (or a straight) of its own
                merged[-1][1] = r[1]
            elif merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        if self.closed and len(merged) > 1 and merged[0][2] == merged[-1][2]:
            # the seam pieces are one turn, and it is turn 0 so the ids run 0, 1, .. along the loop
            # (the callers count a lap when the id wraps back down)
            merged[-1][1] = merged[0][1] + n
            merged.insert(0, merged.pop())
            merged.pop(1)
        turn_of = np.full(n, -1, dtype=np.int32)
        k = 0
        for a, b, sgn in merged:
            if sgn == 0.0:
                continue
            idx = np.arange(a, b) % n
            turn_of[idx] = k
            k += 1
        self.n_turns = k
        self.turn_sign = np.array([sgn for _, _, sgn in merged if sgn != 0.0], dtype=np.float64)
        return turn_of

    def turn_at(self, s: float) -> int:
        """Turn id under arc length ``s`` (-1 on a straight)."""
        return int(self._turn_of[self._index(s)])

    def turn_radius(self) -> float:
        """Typical bend radius [m]: median 1 / |curvature| over the turning samples."""
        k = np.abs(self.curvature[self._turn_of >= 0])
        return float(1.0 / np.median(k)) if len(k) else math.inf


def load_height_png(path: str) -> np.ndarray:
    """Grayscale PNG -> [0, 1] float32 (16-bit if the file is, else 8-bit); row 0 = y = -hy."""
    from PIL import Image  # noqa: PLC0415  (optional dependency, only needed to load a terrain)

    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        return np.asarray(img, dtype=np.float32) / 65535.0
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def downsample(h: np.ndarray, k: int, mode: str = "max") -> np.ndarray:
    """Aggregate ``h`` over ``k x k`` blocks so node (r, c) of the result sits on fine node (r k, c k).

    ``"max"`` is the upper envelope of the rocks: rigid parts colliding with it can never be
    inside a visible boulder. ``"mean"`` (the track example's choice) sits below the rock tops
    and lets rims, arms and chassis sink into visible rock.
    """
    if k <= 1:
        return h
    agg = np.max if mode == "max" else np.mean
    nrow, ncol = h.shape
    nr, nc = (nrow - 1) // k + 1, (ncol - 1) // k + 1
    out = np.empty((nr, nc), dtype=np.float32)
    for r in range(nr):
        r0, r1 = max(r * k - k // 2, 0), min(r * k + k // 2 + 1, nrow)
        out[r] = [float(agg(h[r0:r1, max(c * k - k // 2, 0) : min(c * k + k // 2 + 1, ncol)])) for c in range(nc)]
    return out


class FieldTerrain:
    """A ``newton-terrain-tool`` field: the rugged layer ``I_N`` and the arena rectangle.

    :attr:`origin` is the world (x, y) of the grid centre, so ``field = world - origin``. All
    methods take world coordinates. ``heights(w)`` is stage ``w``.
    """

    def __init__(self, terrain_dir: str, name: str, difficulty: float):
        d = os.path.join(terrain_dir, name)
        meta_path = os.path.join(d, f"{name}_terrain.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"terrain '{name}' not found ({meta_path}); bake it with third_party/newton-terrain-tool/regenerate_all.sh --name {name} --no-trackgen"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)
        if self.meta.get("kind") != "field":
            raise ValueError(
                f"terrain '{name}' is a {self.meta.get('kind', 'track')} terrain; this example needs a field (kind: field)"
            )
        g = self.meta["grid"]
        self.name = name
        self.cell = float(g["cell"])
        self.hx, self.hy = float(g["hx"]), float(g["hy"])
        self.max_h = float(g["max_h"])
        self.field = (load_height_png(os.path.join(d, self.meta["files"]["height_png"])) * self.max_h).astype(
            np.float32
        )
        assert self.field.shape == (int(g["nrow"]), int(g["ncol"])), "height PNG does not match terrain.json"
        # Cells worth drawing (baked by the terrain tool: measured ground + margin); None = draw all.
        self.draw_mask: np.ndarray | None = None
        draw_file = self.meta["files"].get("draw_png")
        if draw_file:
            self.draw_mask = load_height_png(os.path.join(d, draw_file)) > 0.5
            assert self.draw_mask.shape == self.field.shape, "draw PNG does not match terrain.json"
        a = self.meta["arena"]
        self.arena_half_x, self.arena_half_y = float(a["half_x"]), float(a["half_y"])
        self.stages = [float(s) for s in self.meta.get("stage", {}).get("stages", (0.2, 0.4, 0.6, 0.8, 1.0))]
        self.vehicle = dict(self.meta.get("vehicle", {}))
        self.w = float(difficulty)
        self.origin = (0.0, 0.0)
        self.corridor_half_width = float(self.meta.get("corridor", {}).get("half_width", 0.0)) or None
        self.track: ReferenceTrack | None = None
        self.reference_track = np.empty((0, 3), dtype=np.float32)
        self.reference_track_indices = np.empty((0, 2), dtype=np.int32)
        track_file = self.meta["files"].get("reference_track_json")
        if track_file:
            with open(os.path.join(d, track_file)) as f:
                self.track = ReferenceTrack(json.load(f))
            self.reference_track, self.reference_track_indices = self.track.ground, self.track.segments

    @property
    def start_pose(self) -> tuple[float, float, float]:
        """Where the vehicle spawns ``(x, y, yaw)`` [m, m, rad]: pose 0 of the reference track, else
        the scene origin heading +x."""
        return self.track.start_pose if self.track is not None else (0.0, 0.0, 0.0)

    def reference_track_segments(self, w: float | None = None, lift: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Return recorded path segment starts and ends in world coordinates [m].

        Args:
            w: Terrain difficulty; use the current stage when omitted.
            lift: Display offset above the terrain [m].
        """
        points = self.reference_track.copy()
        points[:, :2] += np.asarray(self.origin)
        points[:, 2] = points[:, 2] * (self.w if w is None else float(w)) + lift
        return points[self.reference_track_indices[:, 0]], points[self.reference_track_indices[:, 1]]

    @property
    def nrow(self) -> int:
        return int(self.field.shape[0])

    @property
    def ncol(self) -> int:
        return int(self.field.shape[1])

    def heights(self, w: float | None = None) -> np.ndarray:
        """Stage ``w * I_N`` [m], float32."""
        w = self.w if w is None else float(w)
        return (w * self.field).astype(np.float32)

    def height_at(self, x: float, y: float, w: float | None = None) -> float:
        """Rendered triangle height [m] at world (x, y) for stage ``w``."""
        w = self.w if w is None else float(w)
        return w * triangulated_sample(self, self.field, x, y)

    def downsample(self, h: np.ndarray, k: int, mode: str = "max") -> np.ndarray:
        """Block-aggregate ``h`` by ``k``; see :func:`downsample` (kept as a method for the Isaac Lab task)."""
        return downsample(h, k, mode)
