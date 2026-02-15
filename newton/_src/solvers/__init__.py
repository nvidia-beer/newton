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

from .featherstone import SolverFeatherstone
from .flags import SolverNotifyFlags
from .implicit_mpm import SolverImplicitMPM
from .bend import SolverBend
from .inflatable import SolverInflatable
from .mujoco import SolverMuJoCo
from .semi_implicit import SolverSemiImplicit
from .soft import SolverSoft
from .solver import SolverBase
from .style3d import SolverStyle3D
from .vbd import SolverVBD
from .xpbd import SolverXPBD

# Re-export from sim for backward compatibility
from ..sim import (
    SurfaceBox,
    create_surface_box,
    TetraSphere,
    create_tetra_sphere,
    TetraCylinder,
    create_tetra_cylinder,
    TetraBox,
    create_tetra_box,
)

__all__ = [
    "SolverBase",
    "SolverBend",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverInflatable",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSemiImplicit",
    "SolverSoft",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
    "TetraSphere",
    "TetraCylinder",
    "TetraBox",
    "SurfaceBox",
    "create_surface_box",
    "create_tetra_sphere",
    "create_tetra_cylinder",
    "create_tetra_box",
]
