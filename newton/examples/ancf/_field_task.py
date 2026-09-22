# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Boulder-field terrain loader shared by the Newton example and the Isaac Lab environment.

:class:`FieldTerrain` loads a ``newton-terrain-tool`` field (``kind: field``); the stage
weight ``w`` scales the one baked layer (``heights = w * I_N``), which is how the relief is
changed at run time. Host side, NumPy only (PIL for the PNG).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np


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
        a = self.meta["arena"]
        self.arena_half_x, self.arena_half_y = float(a["half_x"]), float(a["half_y"])
        self.stages = [float(s) for s in self.meta.get("stage", {}).get("stages", (0.2, 0.4, 0.6, 0.8, 1.0))]
        self.vehicle = dict(self.meta.get("vehicle", {}))
        self.w = float(difficulty)
        self.origin = (0.0, 0.0)

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
        """Bilinear height [m] at world (x, y) for stage ``w``."""
        w = self.w if w is None else float(w)
        x, y = x - self.origin[0], y - self.origin[1]
        fx = min(max((x + self.hx) / self.cell, 0.0), self.ncol - 1 - 1e-6)
        fy = min(max((y + self.hy) / self.cell, 0.0), self.nrow - 1 - 1e-6)
        c, r = int(fx), int(fy)
        tx, ty = fx - c, fy - r
        h = self.field
        return w * float(
            (h[r, c] * (1 - tx) + h[r, c + 1] * tx) * (1 - ty) + (h[r + 1, c] * (1 - tx) + h[r + 1, c + 1] * tx) * ty
        )

    def max_height_in_discs(self, centres_xy: np.ndarray, radius: float, w: float | None = None) -> float:
        """Highest grid point under any of the discs (K, 2) [world m] of ``radius`` [m]: what a
        dropped vehicle must clear."""
        w = self.w if w is None else float(w)
        best = -math.inf
        k = int(math.ceil(radius / self.cell)) + 1
        for cx, cy in np.asarray(centres_xy, dtype=np.float64) - np.array(self.origin):
            c0 = int(round((cx + self.hx) / self.cell))
            r0 = int(round((cy + self.hy) / self.cell))
            cs = slice(max(c0 - k, 0), min(c0 + k + 1, self.ncol))
            rs = slice(max(r0 - k, 0), min(r0 + k + 1, self.nrow))
            if cs.start >= cs.stop or rs.start >= rs.stop:
                continue
            xs = -self.hx + np.arange(cs.start, cs.stop) * self.cell
            ys = -self.hy + np.arange(rs.start, rs.stop) * self.cell
            XX, YY = np.meshgrid(xs, ys)
            win = self.field[rs, cs]
            inside = (XX - cx) ** 2 + (YY - cy) ** 2 <= radius * radius
            if inside.any():
                best = max(best, float(win[inside].max()))
        return w * (best if math.isfinite(best) else 0.0)

    def max_height_in_rect(self, x0: float, x1: float, y0: float, y1: float, w: float | None = None) -> float:
        """Highest grid point inside the world rectangle [x0, x1] x [y0, y1] [m]."""
        w = self.w if w is None else float(w)
        c0 = max(int(math.floor((x0 - self.origin[0] + self.hx) / self.cell)), 0)
        c1 = min(int(math.ceil((x1 - self.origin[0] + self.hx) / self.cell)) + 1, self.ncol)
        r0 = max(int(math.floor((y0 - self.origin[1] + self.hy) / self.cell)), 0)
        r1 = min(int(math.ceil((y1 - self.origin[1] + self.hy) / self.cell)) + 1, self.nrow)
        if c0 >= c1 or r0 >= r1:
            return 0.0
        return w * float(self.field[r0:r1, c0:c1].max())

    def downsample(self, h: np.ndarray, k: int, mode: str = "max") -> np.ndarray:
        """Block-aggregate ``h`` by ``k``; see :func:`downsample` (kept as a method for the Isaac Lab task)."""
        return downsample(h, k, mode)
