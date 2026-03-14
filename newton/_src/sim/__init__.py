# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from . import box_topology

# glue_utils is imported lazily to avoid circular import
_glue_utils_loading = False


def __getattr__(name: str):
    if name == "glue_utils":
        global _glue_utils_loading
        if _glue_utils_loading:
            raise AttributeError(
                "module %r is loading glue_utils; circular import detected."
                % (__name__,)
            )
        _glue_utils_loading = True
        try:
            from . import glue_utils as _glue_utils
            return _glue_utils
        finally:
            _glue_utils_loading = False
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from . import ik
from .articulation import eval_fk, eval_ik, eval_jacobian, eval_mass_matrix
from .builder import ModelBuilder
from .collide import CollisionPipeline
from .contacts import Contacts
from .control import Control
from .enums import (
    BodyFlags,
    EqType,
    JointTargetMode,
    JointType,
)
from .model import Model
from .state import State
from .tetra_sphere import TetraSphere, create_tetra_sphere
from .tetra_cylinder import TetraCylinder, create_tetra_cylinder
from .tetra_box import TetraBox, create_tetra_box, get_axis_aligned_springs
from .surface_box import SurfaceBox, create_surface_box

__all__ = [
    "BodyFlags",
    "CollisionPipeline",
    "Contacts",
    "Control",
    "EqType",
    "JointTargetMode",
    "JointType",
    "Model",
    "ModelBuilder",
    "State",
    "TetraSphere",
    "TetraCylinder",
    "TetraBox",
    "SurfaceBox",
    "box_topology",
    "glue_utils",
    "create_surface_box",
    "color_graph",
    "create_tetra_sphere",
    "create_tetra_cylinder",
    "create_tetra_box",
    "get_axis_aligned_springs",
    "eval_fk",
    "eval_ik",
    "eval_jacobian",
    "eval_mass_matrix",
]
