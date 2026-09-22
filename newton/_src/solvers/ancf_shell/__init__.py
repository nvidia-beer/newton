# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ANCF3423 shell element solver package.

- Tire mesh geometry loaded from USD assets baked by ``newton-tire-tool``
  (``third_party/newton-tire-tool/``) — see :func:`load_ancf_tire_usd`
- Full ANS (ε_zz at corners, γ_13/γ_23 at mid-edges) + β-transform; EAS (5 modes) in the
  single-tire element kernel only, the batched N-env kernels run with α ≡ 0
- K_eff in 6x6 node-block CSR, refilled in place (graph-capture safe)
- Penalty ground contact (kn=2e6, kd=13, mu=0.9) inside the solver; optional
  heightfield / soil terrain via :class:`TerrainSCM`
- 6x6 block-Jacobi PCG with a fused SpMV kernel
- N-env batched mode: flat [N*per_env] arrays, shared sparsity, PcgSolverBatched
- :class:`SolverANCFShellRigid` couples each tire to a rigid spindle body
"""

from .model_ancf_shell import (
    ANCFMaterial,
    ANCFShellModel,
    SpindleAsset,
    TireAssetMeta,
    ancf_material_from_engineering,
    isotropic_ancf_material,
    load_ancf_tire_usd,
)
from .solver_ancf_shell import SolverANCFShell
from .solver_ancf_shell_rigid import SolverANCFShellRigid
from .terrain_scm import TerrainSCM

__all__ = [
    "ANCFMaterial",
    "ANCFShellModel",
    "SolverANCFShell",
    "SolverANCFShellRigid",
    "SpindleAsset",
    "TerrainSCM",
    "TireAssetMeta",
    "ancf_material_from_engineering",
    "isotropic_ancf_material",
    "load_ancf_tire_usd",
]
