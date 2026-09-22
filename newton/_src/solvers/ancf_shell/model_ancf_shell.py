# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
ANCF shell model: mesh data, orthotropic material, and tire USD asset loader.

Tire mesh geometry is authored offline by ``newton-tire-tool``
(``third_party/newton-tire-tool/``) and loaded here via :func:`load_ancf_tire_usd` —
this module does not generate mesh geometry.

All physical quantities in SI units (metres, kg, Pa, s).
"""

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.struct
class ANCFMaterial:
    """Orthotropic ANCF3423 shell material in Voigt stiffness form.

    The 6×6 stiffness matrix C (S = C·e) for an orthotropic material has
    the block structure::

        C = | C11 C12 C13  0   0   0  |
            | C12 C22 C23  0   0   0  |
            | C13 C23 C33  0   0   0  |
            |  0   0   0  G23  0   0  |
            |  0   0   0   0  G13  0  |
            |  0   0   0   0   0  G12 |

    These 9 independent entries are computed from (E1, E2, E3, nu12, nu13,
    nu23, G12, G13, G23) on the Python side via :func:`ancf_material_from_engineering`.

    Voigt strain ordering: [E11, E22, E33, 2·E12, 2·E13, 2·E23]
    (directions: 1=circumferential ξ, 2=radial η, 3=thickness ζ)
    """

    C11: float
    C22: float
    C33: float
    C12: float
    C13: float
    C23: float
    G23: float
    G13: float
    G12: float
    rho: float  # density [kg/m³]
    alpha_damp: float  # Rayleigh stiffness damping coefficient


def ancf_material_from_engineering(
    E1: float,
    E2: float,
    E3: float,
    nu12: float,
    nu13: float,
    nu23: float,
    G12: float,
    G13: float,
    G23: float,
    rho: float,
    alpha_damp: float = 0.0,
) -> ANCFMaterial:
    """Build an :class:`ANCFMaterial` from engineering constants.

    Inverts the 3×3 compliance block analytically, following the symmetry
    conditions nu12/E1 = nu21/E2 etc.
    """
    # 3×3 compliance (upper-left block of 6×6 S)
    S3 = np.array(
        [
            [1.0 / E1, -nu12 / E1, -nu13 / E1],
            [-nu12 / E1, 1.0 / E2, -nu23 / E2],
            [-nu13 / E1, -nu23 / E2, 1.0 / E3],
        ],
        dtype=np.float64,
    )
    C3 = np.linalg.inv(S3)

    mat = ANCFMaterial()
    mat.C11 = float(C3[0, 0])
    mat.C22 = float(C3[1, 1])
    mat.C33 = float(C3[2, 2])
    mat.C12 = float(C3[0, 1])
    mat.C13 = float(C3[0, 2])
    mat.C23 = float(C3[1, 2])
    mat.G23 = float(G23)
    mat.G13 = float(G13)
    mat.G12 = float(G12)
    mat.rho = float(rho)
    mat.alpha_damp = float(alpha_damp)
    return mat


def isotropic_ancf_material(
    E: float,
    nu: float,
    rho: float,
    alpha_damp: float = 0.15,
) -> ANCFMaterial:
    """Build an isotropic :class:`ANCFMaterial` from a single Young's modulus.

    Convenience wrapper for quick per-tire parameterization.  Sets
    E1=E2=E3=E, nu12=nu13=nu23=nu, G12=G13=G23=E/(2*(1+nu)).
    """
    G = E / (2.0 * (1.0 + nu))
    return ancf_material_from_engineering(
        E1=E,
        E2=E,
        E3=E,
        nu12=nu,
        nu13=nu,
        nu23=nu,
        G12=G,
        G13=G,
        G23=G,
        rho=rho,
        alpha_damp=alpha_damp,
    )


# ---------------------------------------------------------------------------
# ANCFShellModel: mesh topology + per-element material/thickness arrays
# ---------------------------------------------------------------------------


@dataclass
class ANCFShellModel:
    """Static mesh data for the ANCF shell solver.

    Arrays are Warp arrays on *device*.  Topology (connectivity, reference
    geometry) does not change after construction.

    Attributes:
        n_nodes:      total node count
        n_elems:      total element count
        node_x0:      reference positions  [n_nodes, vec3]
        node_D0:      reference gradients  [n_nodes, vec3]  (through-thickness)
        elem_nodes:   4 node indices per element  [n_elems, 4] int32
        elem_h:       element thickness [m]  [n_elems] float32
        elem_mat:     material per element  [n_elems, 11] float32, columns in
                      :class:`ANCFMaterial` field order
                      (C11, C22, C33, C12, C13, C23, G23, G13, G12 [Pa], rho [kg/m³], alpha_damp)
        elem_eas_alpha: EAS internal DOFs  [n_elems, 5] float32 (zeros initially)
        elem_fiber_cos / elem_fiber_sin: cos/sin of the fiber angle  [n_elems] float32
        device:       Warp device
    """

    n_nodes: int
    n_elems: int
    node_x0: wp.array[wp.vec3]
    node_D0: wp.array[wp.vec3]
    elem_nodes: wp.array2d[wp.int32]
    elem_h: wp.array[float]
    elem_mat: wp.array2d[float]
    elem_eas_alpha: wp.array2d[float]
    elem_fiber_cos: wp.array[float]
    elem_fiber_sin: wp.array[float]
    device: str = "cuda:0"


@dataclass
class SpindleAsset:
    """Rigid spindle (hub) baked next to the shell mesh (``/Tire/Spindle``), tire frame: axle along X.

    Everything a consumer needs to build the rigid wheel body the bead rings are
    pinned to — no MJCF/URDF rig required. Produced by ``newton-tire-tool``
    (``newton_tire_tool.spindle``).
    """

    points: np.ndarray
    """Visual triangle-mesh vertices [m], shape [n_pts, 3], float32."""
    triangle_indices: np.ndarray
    """Triangle vertex indices, shape [n_tri, 3], int32."""
    mass: float
    """Lumped rigid wheel mass [kg]."""
    com: np.ndarray
    """Centre of mass [m], shape [3]."""
    inertia: np.ndarray
    """Inertia tensor about :attr:`com` [kg m^2], shape [3, 3]."""


@dataclass
class TireAssetMeta:
    """Bake-time provenance of a :func:`load_ancf_tire_usd` asset (``customData`` on the mesh prim).

    Topology (node/element counts, connectivity) is fully determined by the loaded
    mesh itself; these fields are the build parameters the mesh was baked from, for
    callers that need to derive section boundaries (e.g. bead-ring node ranges).
    """

    n_circ: int
    n_bead_rows: int
    R_outer: float
    R_inner: float
    width: float
    pressure: float
    spindle: SpindleAsset | None = None
    """Rigid hub baked with the tire, or None if the asset has no ``/Tire/Spindle`` prim."""
    contact_kn: float | None = None
    """Recommended ground-contact stiffness per node [N/m] (scaled with node area at bake time), or None."""
    pcg_iters: int | None = None
    """Recommended PCG iterations per Newton step for this mesh's element size, or None."""
    shell_thickness: float | None = None
    """Uniform shell thickness [m] to run this mesh at instead of its baked per-section bands, or None."""


def load_ancf_tire_usd(
    path: str, mesh_path: str = "/Tire/Mesh", device: str = "cuda:0"
) -> tuple[ANCFShellModel, TireAssetMeta]:
    """Load a pre-baked ANCF3423 shell tire mesh from a USD asset.

    The asset is produced offline by ``newton-tire-tool``
    (``third_party/newton-tire-tool/scripts/bake_tire.py``), which owns the
    Polaris/parabolic profile sampling and orthotropic material layout that
    used to be computed at runtime inside Newton. This loader does no geometry
    math — it only reads back the mesh, primvars and bake metadata.

    Args:
        path:      USD file path.
        mesh_path: prim path of the baked ``UsdGeom.Mesh``.
        device:    Warp target device.

    Returns:
        Tuple of the :class:`ANCFShellModel` (arrays on *device*) and its :class:`TireAssetMeta`.
    """
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))

    positions = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    connectivity = face_indices.reshape(-1, 4)

    primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    gradients = np.array(primvars_api.GetPrimvar("D0").Get(), dtype=np.float32)
    elem_h_np = np.array(primvars_api.GetPrimvar("shellThickness").Get(), dtype=np.float32)
    material_fields = list(mesh.GetPrim().GetCustomDataByKey("materialFields"))
    mat_flat = np.array(primvars_api.GetPrimvar("material").Get(), dtype=np.float32)
    mat_array_np = mat_flat.reshape(-1, len(material_fields))

    spindle = None
    spindle_prim = stage.GetPrimAtPath(str(Sdf.Path(mesh_path).GetParentPath().AppendChild("Spindle")))
    if spindle_prim.IsValid():
        sp_mesh = UsdGeom.Mesh(spindle_prim)
        sp_custom = spindle_prim.GetCustomData()
        spindle = SpindleAsset(
            points=np.array(sp_mesh.GetPointsAttr().Get(), dtype=np.float32),
            triangle_indices=np.array(sp_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3),
            mass=float(sp_custom["mass"]),
            com=np.array(sp_custom["com"], dtype=np.float64),
            inertia=np.array(sp_custom["inertia"], dtype=np.float64).reshape(3, 3),
        )

    custom = mesh.GetPrim().GetCustomData()
    meta = TireAssetMeta(
        n_circ=int(custom["nCirc"]),
        n_bead_rows=int(custom["nBeadRows"]),
        R_outer=float(custom["rOuter"]),
        R_inner=float(custom["rInner"]),
        width=float(custom["width"]),
        pressure=float(custom["pressure"]),
        spindle=spindle,
        contact_kn=float(custom["contactKn"]) if "contactKn" in custom else None,
        pcg_iters=int(custom["pcgIters"]) if "pcgIters" in custom else None,
        shell_thickness=float(custom["shellThickness"]) if "shellThickness" in custom else None,
    )

    total_nodes = positions.shape[0]
    n_elems = connectivity.shape[0]
    eas_alpha_np = np.zeros((n_elems, 5), dtype=np.float32)
    fiber_cos_np = np.ones(n_elems, dtype=np.float32)
    fiber_sin_np = np.zeros(n_elems, dtype=np.float32)

    ancf_model = ANCFShellModel(
        n_nodes=total_nodes,
        n_elems=n_elems,
        node_x0=wp.array(positions, dtype=wp.vec3, device=device),
        node_D0=wp.array(gradients, dtype=wp.vec3, device=device),
        elem_nodes=wp.array(connectivity, dtype=wp.int32, device=device),
        elem_h=wp.array(elem_h_np, dtype=float, device=device),
        elem_mat=wp.array(mat_array_np, dtype=float, device=device),
        elem_eas_alpha=wp.array(eas_alpha_np, dtype=float, device=device),
        elem_fiber_cos=wp.array(fiber_cos_np, dtype=float, device=device),
        elem_fiber_sin=wp.array(fiber_sin_np, dtype=float, device=device),
        device=device,
    )
    return ancf_model, meta
