# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA graph capture and state snapshot helpers shared by the ANCF examples.

A capture warm-up steps the simulation, so the examples save every array the step
mutates, capture, and put the saved values back.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp


def try_capture(fn: Callable[[], None], failure: str, device: str = "cuda:0") -> wp.Graph | None:
    """Capture ``fn`` as a CUDA graph on ``device``; ``None`` if the capture fails.

    On failure the capture is ended, and ``failure`` is printed with ``{error}``
    replaced by the exception (``{error!r}`` for its repr).
    """
    try:
        wp.capture_begin(device=device)
        fn()
        return wp.capture_end(device=device)
    except Exception as error:
        try:
            wp.capture_end(device=device)
        except Exception:
            pass
        print(failure.format(error=error))
        return None


def snapshot_arrays(arrays: dict[str, wp.array]) -> dict[str, np.ndarray]:
    """Host copies of ``arrays``, keyed like the input (its order is the restore order)."""
    return {name: array.numpy().copy() for name, array in arrays.items()}


def restore_arrays(saved: dict[str, np.ndarray], live: dict[str, wp.array]) -> None:
    """Assign every snapshot in ``saved`` back to the array of the same name in ``live``."""
    for name, value in saved.items():
        live[name].assign(value)
