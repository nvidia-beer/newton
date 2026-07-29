# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ANCF3423 shell element solver package.

- 3-section orthotropic Polaris materials (bead / sidewall / tread)
- 71-point Polaris cross-section profile
- Full ANS (ε_zz at corners, γ_13/γ_23 at mid-edges) + EAS (5 modes)
- BSR sparse K_eff with scatter_map for graph-capture-safe in-place updates
- Penalty ground contact (kn=2e6, kd=13, mu=0.9) inside the solver
- Diagonal-preconditioned PCG with custom SpMV kernel (no wps.bsr_mv)
- N-env batched mode: flat [N*per_env] arrays, shared sparsity, PcgSolverBatched
"""

from .model_ancf_shell import (
    ANCFMaterial,
    ANCFShellModel,
    ancf_material_from_engineering,
    build_ancf_parabolic_mesh,
    build_ancf_tire_mesh,
    isotropic_ancf_material,
    polaris_bead_material,
    polaris_sidewall_material,
    polaris_tread_material,
)
from .solver_ancf_shell import SolverANCFShell

# Backward-compat alias
ANCFModel = ANCFShellModel

__all__ = [
    "ANCFMaterial",
    "ANCFModel",
    "ANCFShellModel",
    "SolverANCFShell",
    "ancf_material_from_engineering",
    "build_ancf_parabolic_mesh",
    "build_ancf_tire_mesh",
    "isotropic_ancf_material",
    "polaris_bead_material",
    "polaris_sidewall_material",
    "polaris_tread_material",
]
