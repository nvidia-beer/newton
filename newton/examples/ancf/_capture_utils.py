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


def reset_ancf_state(solver, ancf_model, node_x: np.ndarray, node_D: np.ndarray) -> None:
    """Put a shell solver back at rest after a warm-up step.

    Positions and directors come from ``node_x`` / ``node_D``; every velocity,
    acceleration, force and EAS array is zeroed.  ``set_cavity(p, p)`` gives
    p_gauge = 0 at V_ref, so the reference configuration has no internal force.
    """
    solver.node_x.assign(node_x)
    solver.node_D.assign(node_D)
    for array in (
        solver.node_xd,
        solver.node_xdd,
        solver.node_Dd,
        solver.node_Ddd,
        solver.global_f_int,
        solver.global_f_int0,
        solver.global_f_ext,
        solver.global_f_ext0,
        solver.node_f_ext_persistent,
        ancf_model.elem_eas_alpha,
    ):
        array.zero_()
