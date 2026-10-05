# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Restore rigid kinematic buffers between responses at the same time level."""

import warp as wp

from .kernels_coupled import copy_kinematics


class KinematicStateCache:
    """The caller supplies every kinematic buffer modified by trial responses.

    Contact geometry, mass matrices, and controls must remain valid at the
    saved time level. Rebuild them once before saving each new substep.

    Exactly six contiguous arrays of float32-based dtypes (MuJoCo ``qpos``,
    ``qvel``, ``xpos``, ``xquat``, ``cvel``, ``subtree_com``) are copied by the
    single :func:`copy_kinematics` launch; any other list uses one
    :func:`warp.copy` per array.
    """

    def __init__(self, arrays):
        self.pairs = [(live, wp.clone(live)) for live in arrays]
        self.views = None
        if len(self.pairs) == 6 and all(live.is_contiguous for live, _ in self.pairs):
            # Vector dtypes view as float; a non-float dtype is a caller error and raises here.
            self.views = tuple(tuple(pair[column].view(float).flatten() for pair in self.pairs) for column in (0, 1))

    def __bool__(self):
        return bool(self.pairs)

    def copy(self, restore):
        if self.views is not None:
            live, saved = self.views
            source, destination = (saved, live) if restore else (live, saved)
            wp.launch(
                copy_kinematics, dim=max(a.size for a in live), inputs=[*source, *destination], device=live[0].device
            )
        else:
            for live, saved in self.pairs:
                source, destination = (saved, live) if restore else (live, saved)
                wp.copy(destination, source)
