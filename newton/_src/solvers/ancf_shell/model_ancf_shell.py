# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
ANCF shell model: mesh data, orthotropic material, and tire mesh builder.

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
# Polaris ANCF4 tire material presets (from Chrono Polaris_ANCF4Tire_Lumped.json)
# ---------------------------------------------------------------------------


def polaris_bead_material() -> ANCFMaterial:
    return ancf_material_from_engineering(
        E1=5e7,
        E2=1.3e8,
        E3=5e7,
        nu12=0.02,
        nu13=0.74,
        nu23=0.45,
        G12=1.63e6,
        G13=1.63e6,
        G23=1.63e6,
        rho=1080.0,
        alpha_damp=0.15,
    )


def polaris_sidewall_material() -> ANCFMaterial:
    return ancf_material_from_engineering(
        E1=5e7,
        E2=6.87e8,
        E3=5e7,
        nu12=0.01,
        nu13=0.48,
        nu23=0.45,
        G12=1.63e6,
        G13=1.63e6,
        G23=1.63e6,
        rho=1000.0,
        alpha_damp=0.15,
    )


def polaris_tread_material() -> ANCFMaterial:
    # E3 = 0.08e8 = 8e6 Pa, G13 = G23 = 0.02e7 = 2e5 Pa  (from Chrono JSON)
    return ancf_material_from_engineering(
        E1=7.5e7,
        E2=1.85e8,
        E3=8e6,
        nu12=0.01,
        nu13=0.70,
        nu23=0.65,
        G12=5.4e7,
        G13=2e5,
        G23=2e5,
        rho=1150.0,
        alpha_damp=0.15,
    )


# ---------------------------------------------------------------------------
# Polaris tire cross-section profile (from Chrono Polaris_ANCF4Tire_Lumped.json)
#
# 71 points [t, x_prf, y_prf]:
#   t      ∈ [0, 1]      — profile parameter (0 = left bead, 1 = right bead)
#   x_prf  ∈ [0, 0.199]  — radial offset from rim [m]  (0 at bead, max at tread)
#   y_prf  ∈ [-0.115, 0.115] — axial half-position [m] (±0.115 at beads, 0 at tread center)
# ---------------------------------------------------------------------------
_POLARIS_PROFILE = np.array(
    [
        [0.000000e00, 0.000000e00, -1.150000e-01],
        [1.428571e-02, 1.166670e-02, -1.164180e-01],
        [2.857143e-02, 2.333330e-02, -1.192300e-01],
        [4.285714e-02, 3.500000e-02, -1.230200e-01],
        [5.714286e-02, 4.666670e-02, -1.273710e-01],
        [7.142857e-02, 5.833330e-02, -1.318700e-01],
        [8.571429e-02, 7.000000e-02, -1.361330e-01],
        [1.000000e-01, 8.166670e-02, -1.399910e-01],
        [1.142857e-01, 9.333330e-02, -1.433510e-01],
        [1.285714e-01, 1.050000e-01, -1.461240e-01],
        [1.428571e-01, 1.166670e-01, -1.482160e-01],
        [1.571429e-01, 1.283330e-01, -1.495390e-01],
        [1.714286e-01, 1.400000e-01, -1.500000e-01],
        [1.857143e-01, 1.475000e-01, -1.486380e-01],
        [2.000000e-01, 1.550000e-01, -1.457860e-01],
        [2.142857e-01, 1.625000e-01, -1.419760e-01],
        [2.285714e-01, 1.700000e-01, -1.360000e-01],
        [2.428571e-01, 1.768970e-01, -1.288420e-01],
        [2.571429e-01, 1.831090e-01, -1.216840e-01],
        [2.714286e-01, 1.883940e-01, -1.145260e-01],
        [2.857143e-01, 1.925100e-01, -1.073680e-01],
        [3.000000e-01, 1.953230e-01, -1.002110e-01],
        [3.142857e-01, 1.970380e-01, -9.305260e-02],
        [3.285714e-01, 1.979260e-01, -8.589470e-02],
        [3.428571e-01, 1.982580e-01, -7.873680e-02],
        [3.571429e-01, 1.983020e-01, -7.157890e-02],
        [3.714286e-01, 1.983090e-01, -6.442110e-02],
        [3.857143e-01, 1.983540e-01, -5.726320e-02],
        [4.000000e-01, 1.984290e-01, -5.010530e-02],
        [4.142857e-01, 1.985240e-01, -4.294740e-02],
        [4.285714e-01, 1.986300e-01, -3.578950e-02],
        [4.428571e-01, 1.987380e-01, -2.863160e-02],
        [4.571429e-01, 1.988390e-01, -2.147370e-02],
        [4.714286e-01, 1.989220e-01, -1.431580e-02],
        [4.857143e-01, 1.989790e-01, -7.157890e-03],
        [5.000000e-01, 1.990000e-01, 0.000000e00],
        [5.142857e-01, 1.989790e-01, 7.157890e-03],
        [5.285714e-01, 1.989220e-01, 1.431580e-02],
        [5.428571e-01, 1.988390e-01, 2.147370e-02],
        [5.571429e-01, 1.987380e-01, 2.863160e-02],
        [5.714286e-01, 1.986300e-01, 3.578950e-02],
        [5.857143e-01, 1.985240e-01, 4.294740e-02],
        [6.000000e-01, 1.984290e-01, 5.010530e-02],
        [6.142857e-01, 1.983540e-01, 5.726320e-02],
        [6.285714e-01, 1.983090e-01, 6.442110e-02],
        [6.428571e-01, 1.983020e-01, 7.157890e-02],
        [6.571429e-01, 1.982580e-01, 7.873680e-02],
        [6.714286e-01, 1.979260e-01, 8.589470e-02],
        [6.857143e-01, 1.970380e-01, 9.305260e-02],
        [7.000000e-01, 1.953230e-01, 1.002110e-01],
        [7.142857e-01, 1.925100e-01, 1.073680e-01],
        [7.285714e-01, 1.883940e-01, 1.145260e-01],
        [7.428571e-01, 1.831090e-01, 1.216840e-01],
        [7.571429e-01, 1.768970e-01, 1.288420e-01],
        [7.714286e-01, 1.700000e-01, 1.360000e-01],
        [7.857143e-01, 1.625000e-01, 1.419760e-01],
        [8.000000e-01, 1.550000e-01, 1.457860e-01],
        [8.142857e-01, 1.475000e-01, 1.486380e-01],
        [8.285714e-01, 1.400000e-01, 1.500000e-01],
        [8.428571e-01, 1.283330e-01, 1.495390e-01],
        [8.571429e-01, 1.166670e-01, 1.482160e-01],
        [8.714286e-01, 1.050000e-01, 1.461240e-01],
        [8.857143e-01, 9.333330e-02, 1.433510e-01],
        [9.000000e-01, 8.166670e-02, 1.399910e-01],
        [9.142857e-01, 7.000000e-02, 1.361330e-01],
        [9.285714e-01, 5.833330e-02, 1.318700e-01],
        [9.428571e-01, 4.666670e-02, 1.273710e-01],
        [9.571429e-01, 3.500000e-02, 1.230200e-01],
        [9.714286e-01, 2.333330e-02, 1.192300e-01],
        [9.857143e-01, 1.166670e-02, 1.164180e-01],
        [1.000000e00, 0.000000e00, 1.150000e-01],
    ],
    dtype=np.float64,
)

# Normalization constants derived from the Polaris profile
_POLARIS_X_MAX = 0.199  # max radial offset [m]  (at tread crown, t=0.5)
_POLARIS_Y_HALF = 0.115  # half axial span   [m]  (at bead edges, t=0 and t=1)


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
        elem_h:       element thickness  [n_elems] float32
        elem_mat:     material per element  [n_elems] ANCFMaterial (struct array)
        surf_tris:    surface triangle connectivity for contact  [n_tris, 3] int32
        device:       Warp device
    """

    n_nodes: int
    n_elems: int
    node_x0: wp.array  # dtype=wp.vec3
    node_D0: wp.array  # dtype=wp.vec3
    elem_nodes: wp.array  # dtype=wp.int32, shape (n_elems, 4)
    elem_h: wp.array  # dtype=float
    elem_mat: wp.array  # dtype=ANCFMaterial
    surf_tris: wp.array  # dtype=wp.int32, shape (n_tris, 3)
    elem_eas_alpha: wp.array  # dtype=float, shape (n_elems, 5) — EAS internal DOFs (zeros initially)
    elem_fiber_cos: wp.array  # dtype=float, shape (n_elems,) — cos(fiber_angle), defaults to 1.0
    elem_fiber_sin: wp.array  # dtype=float, shape (n_elems,) — sin(fiber_angle), defaults to 0.0
    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# Tire mesh builder
# ---------------------------------------------------------------------------


def build_ancf_tire_mesh(
    R_outer: float = 0.33,
    R_inner: float = 0.13,
    width: float = 0.23,
    n_circ: int = 40,
    section_divs: tuple[int, int, int] = (1, 2, 3),
    section_mats: tuple[ANCFMaterial, ...] | None = None,
    section_h: tuple[float, float, float] = (9e-3, 3.1e-3, 8.1e-3),
    pressure: float = 110e3,
    device: str = "cuda:0",
) -> ANCFShellModel:
    """Generate an ANCF3423 shell tire mesh in reference (undeformed) config.

    Matches the Chrono ``ChANCFTire`` / ``Polaris_ANCF4Tire_Lumped.json`` mesh
    layout:

    * **Profile shape**: the Polaris 71-point cross-section spline is sampled at
      each node row and scaled to ``(R_inner, R_outer, width)``.  This gives the
      correct bead–sidewall–tread bulge instead of a perfect cylinder.
    * **Symmetric section layout**: following Chrono's
      ``div_width = 2*(bead + sidewall + tread)`` rule, each section is mirrored
      about the tread centre so the mesh reads bead → sidewall → tread →
      sidewall → bead.

    Newton is Y-up; the tire axle is along X (horizontal) so the tread contacts
    the ``Y = 0`` ground plane when the tire falls under gravity.

    Args:
        R_outer:      outer (tread crown) radius [m]
        R_inner:      inner (bead / rim attach) radius [m]
        width:        bead-to-bead axial width [m]
        n_circ:       elements in the circumferential direction
        section_divs: element count per half-section ``(bead, sidewall, tread_half)``;
                      total axial elements = ``2 * sum(section_divs)``
        section_mats: :class:`ANCFMaterial` per section ``(bead, sidewall, tread)``;
                      defaults to Polaris presets
        section_h:    shell thickness per section ``(bead, sidewall, tread)`` [m]
        pressure:     initial inflation pressure [Pa] (stored for reference only)
        device:       Warp target device

    Returns:
        :class:`ANCFShellModel` with all arrays on *device*.
    """
    if section_mats is None:
        section_mats = (
            polaris_bead_material(),
            polaris_sidewall_material(),
            polaris_tread_material(),
        )

    n_bead, n_side, n_tread = section_divs

    # Chrono: div_width = 2*(bead + sidewall + tread) — symmetric about tread centre
    n_ax_divs = 2 * (n_bead + n_side + n_tread)
    n_ax_nodes = n_ax_divs + 1

    # ------------------------------------------------------------------
    # Node positions — profile-shaped cylindrical mid-surface.
    #
    # Profile parameter t = j / n_ax_divs for each node row j.
    # The Polaris profile is scaled so that:
    #   R(t)    = R_inner  +  x_prf_norm(t) * (R_outer - R_inner)
    #   x_ax(t) = y_prf_norm(t) * (width / 2)
    #
    # Newton axle along X, tread radius in YZ:
    #   x_node  = x_ax(t)           axial (wheel width)
    #   y_node  = -R(t)*cos(theta)  vertical  (theta=0 → bottom of tire)
    #   z_node  =  R(t)*sin(theta)  lateral
    #
    # Through-thickness director D = outward radial = (-cos θ, sin θ) in YZ.
    #
    # Node grid:
    #   i_circ in [0, n_circ)     — periodic around circumference
    #   j_ax   in [0, n_ax_nodes) — bead-to-bead along X
    # Total nodes = n_circ * n_ax_nodes
    # ------------------------------------------------------------------

    t_prf = _POLARIS_PROFILE[:, 0]
    x_prf = _POLARIS_PROFILE[:, 1]
    y_prf = _POLARIS_PROFILE[:, 2]

    # Node row profile parameters — Chrono formula (ChANCFTire::CreateMeshANCF4):
    #   R     = rim_radius + x_prf          (additive radial offset from profile)
    #   x_ax  = y_prf                       (axial position directly from profile)
    # Scale factors stretch the profile to the caller's (R_inner, R_outer, width).
    # x_prf is scaled so the tread crown reaches R_outer;
    # y_prf is scaled so the bead edges reach ±width/2.
    r_scale = (R_outer - R_inner) / _POLARIS_X_MAX  # 1.0 for exact Polaris dims
    y_scale = (width / 2.0) / _POLARIS_Y_HALF  # 1.0 for exact Polaris dims

    t_nodes = np.linspace(0.0, 1.0, n_ax_nodes)
    R_nodes = R_inner + np.interp(t_nodes, t_prf, x_prf) * r_scale  # [n_ax_nodes]
    x_ax = np.interp(t_nodes, t_prf, y_prf) * y_scale  # [n_ax_nodes]

    total_nodes = n_circ * n_ax_nodes
    positions = np.zeros((total_nodes, 3), dtype=np.float32)
    gradients = np.zeros((total_nodes, 3), dtype=np.float32)

    for j in range(n_ax_nodes):
        R = float(R_nodes[j])
        xa = float(x_ax[j])
        for i in range(n_circ):
            theta = 2.0 * np.pi * i / n_circ
            nid = j * n_circ + i
            positions[nid, 0] = xa
            positions[nid, 1] = -R * np.cos(theta)
            positions[nid, 2] = R * np.sin(theta)
            gradients[nid, 0] = 0.0
            gradients[nid, 1] = -np.cos(theta)
            gradients[nid, 2] = np.sin(theta)

    # ------------------------------------------------------------------
    # Symmetric section assignment (matches Chrono ChANCFTire::CreateMeshANCF4)
    #
    #   b1 = n_bead,               b2 = n_ax_divs - n_bead
    #   s1 = n_bead + n_side,      s2 = n_ax_divs - n_bead - n_side
    #
    #   j <  b1  or  j >= b2  →  bead     (section 0)
    #   j <  s1  or  j >= s2  →  sidewall (section 1)
    #   else                   →  tread    (section 2)
    # ------------------------------------------------------------------

    b1 = n_bead
    b2 = n_ax_divs - n_bead
    s1 = n_bead + n_side
    s2 = n_ax_divs - n_bead - n_side

    def _section(ja: int) -> int:
        if ja < b1 or ja >= b2:
            return 0  # bead
        if ja < s1 or ja >= s2:
            return 1  # sidewall
        return 2  # tread

    # ------------------------------------------------------------------
    # Element connectivity — (i_circ, j_ax) quad, i periodic
    #   node0 = j_ax   * n_circ + i_circ
    #   node1 = j_ax   * n_circ + (i_circ+1) % n_circ
    #   node2 = (j_ax+1) * n_circ + (i_circ+1) % n_circ
    #   node3 = (j_ax+1) * n_circ + i_circ
    # ------------------------------------------------------------------

    n_elems = n_circ * n_ax_divs
    connectivity = np.zeros((n_elems, 4), dtype=np.int32)
    elem_h_np = np.zeros(n_elems, dtype=np.float32)
    elem_sec = np.zeros(n_elems, dtype=np.int32)

    eid = 0
    for ja in range(n_ax_divs):
        sec = _section(ja)
        for ic in range(n_circ):
            n0 = ja * n_circ + ic
            n1 = ja * n_circ + (ic + 1) % n_circ
            n2 = (ja + 1) * n_circ + (ic + 1) % n_circ
            n3 = (ja + 1) * n_circ + ic
            connectivity[eid] = [n0, n1, n2, n3]
            elem_h_np[eid] = section_h[sec]
            elem_sec[eid] = sec
            eid += 1

    # ------------------------------------------------------------------
    # Material array  — shape (n_elems, 11) float32
    # ------------------------------------------------------------------
    mat_fields = {
        f: np.zeros(n_elems, dtype=np.float32)
        for f in [
            "C11",
            "C22",
            "C33",
            "C12",
            "C13",
            "C23",
            "G23",
            "G13",
            "G12",
            "rho",
            "alpha_damp",
        ]
    }
    for eid_i in range(n_elems):
        mat = section_mats[elem_sec[eid_i]]
        for fname, arr in mat_fields.items():
            arr[eid_i] = getattr(mat, fname)

    mat_array = wp.array(
        np.stack(
            [
                mat_fields[f]
                for f in [
                    "C11",
                    "C22",
                    "C33",
                    "C12",
                    "C13",
                    "C23",
                    "G23",
                    "G13",
                    "G12",
                    "rho",
                    "alpha_damp",
                ]
            ],
            axis=1,
        ),
        dtype=float,
        device=device,
    )

    # ------------------------------------------------------------------
    # Surface triangles — all quads triangulated for contact/rendering
    # ------------------------------------------------------------------
    tris = []
    for ja in range(n_ax_divs):
        for ic in range(n_circ):
            n0 = ja * n_circ + ic
            n1 = ja * n_circ + (ic + 1) % n_circ
            n2 = (ja + 1) * n_circ + (ic + 1) % n_circ
            n3 = (ja + 1) * n_circ + ic
            tris.append([n0, n1, n2])
            tris.append([n0, n2, n3])
    surf_tris_np = np.array(tris, dtype=np.int32)

    eas_alpha_np = np.zeros((n_elems, 5), dtype=np.float32)
    fiber_cos_np = np.ones(n_elems, dtype=np.float32)
    fiber_sin_np = np.zeros(n_elems, dtype=np.float32)

    return ANCFShellModel(
        n_nodes=total_nodes,
        n_elems=n_elems,
        node_x0=wp.array(positions, dtype=wp.vec3, device=device),
        node_D0=wp.array(gradients, dtype=wp.vec3, device=device),
        elem_nodes=wp.array(connectivity, dtype=wp.int32, device=device),
        elem_h=wp.array(elem_h_np, dtype=float, device=device),
        elem_mat=mat_array,
        surf_tris=wp.array(surf_tris_np, dtype=wp.int32, device=device),
        elem_eas_alpha=wp.array(eas_alpha_np, dtype=float, device=device),
        elem_fiber_cos=wp.array(fiber_cos_np, dtype=float, device=device),
        elem_fiber_sin=wp.array(fiber_sin_np, dtype=float, device=device),
        device=device,
    )


def build_ancf_parabolic_mesh(
    R_outer: float = 0.150,
    R_inner: float = 0.120,
    width: float = 0.100,
    n_circ: int = 16,
    n_ax: int = 4,
    material: ANCFMaterial = None,
    h_shell: float = 0.008,
    pressure: float = 0.0,
    device: str = "cuda:0",
) -> ANCFShellModel:
    """Generate an ANCF3423 shell mesh with a parabolic cross-section profile.

    Profile: ``r(t) = R_inner + (R_outer - R_inner) * 4t(1-t)``, ``t`` in ``[0, 1]``.
    Bead rings at ``j=0`` (left) and ``j=n_ax`` (right) sit at ``r=R_inner``
    (always above ground when hub is at ``z=R_outer``).
    Crown at ``j=n_ax//2`` reaches ``r=R_outer`` and contacts the ground.

    Node ordering: ``j * n_circ + i``, ``j`` in ``[0, n_ax]``, ``i`` in ``[0, n_circ)``.
    This matches the bead-indexing convention of :func:`build_ancf_tire_mesh`
    (left bead = nodes ``[0, n_circ)``, right bead = nodes ``[n_ax*n_circ, (n_ax+1)*n_circ)``).

    Newton is Y-up; the tire axle is along X so the tread contacts ``Y = 0``.

    Args:
        R_outer:  crown (tread) radius [m]
        R_inner:  bead / rim-seat radius [m]; must be < R_outer
        width:    bead-to-bead axial width [m]
        n_circ:   circumferential element count
        n_ax:     axial element count (total, not per section)
        material: :class:`ANCFMaterial`; defaults to isotropic rubber-like material
        h_shell:  shell thickness [m]
        pressure: stored for reference only [Pa]
        device:   Warp target device

    Returns:
        :class:`ANCFShellModel` with all arrays on *device*.
    """
    if material is None:
        material = isotropic_ancf_material(E=1e6, nu=0.45, rho=700.0)

    n_nodes = n_circ * (n_ax + 1)
    n_elems = n_circ * n_ax

    positions = np.zeros((n_nodes, 3), dtype=np.float32)
    gradients = np.zeros((n_nodes, 3), dtype=np.float32)
    elems = np.zeros((n_elems, 4), dtype=np.int32)
    tris_list = []

    for j in range(n_ax + 1):
        t = float(j) / n_ax
        # parabolic: R_inner at edges (t=0,1), R_outer at crown (t=0.5)
        r = R_inner + (R_outer - R_inner) * 4.0 * t * (1.0 - t)
        x = -width / 2.0 + t * width
        for i in range(n_circ):
            theta = 2.0 * np.pi * i / n_circ
            nid = j * n_circ + i
            positions[nid] = [x, -r * np.cos(theta), r * np.sin(theta)]
            gradients[nid] = [0.0, -np.cos(theta), np.sin(theta)]

    for j in range(n_ax):
        for i in range(n_circ):
            i_n = (i + 1) % n_circ
            n00 = j * n_circ + i
            n10 = j * n_circ + i_n
            n11 = (j + 1) * n_circ + i_n
            n01 = (j + 1) * n_circ + i
            eid = j * n_circ + i
            elems[eid] = [n00, n10, n11, n01]
            tris_list.append([n00, n10, n11])
            tris_list.append([n00, n11, n01])

    # Build per-element material array — same layout as build_ancf_tire_mesh: (n_elems, 11) float
    _MAT_FIELDS = ["C11", "C22", "C33", "C12", "C13", "C23", "G23", "G13", "G12", "rho", "alpha_damp"]
    mat_np = np.zeros((n_elems, len(_MAT_FIELDS)), dtype=np.float32)
    for col, fname in enumerate(_MAT_FIELDS):
        mat_np[:, col] = getattr(material, fname)
    mat_array = wp.array(mat_np, dtype=float, device=device)

    h_np = np.full(n_elems, h_shell, dtype=np.float32)
    eas_alpha_np = np.zeros((n_elems, 5), dtype=np.float32)
    fiber_cos_np = np.ones(n_elems, dtype=np.float32)
    fiber_sin_np = np.zeros(n_elems, dtype=np.float32)

    return ANCFShellModel(
        n_nodes=n_nodes,
        n_elems=n_elems,
        node_x0=wp.array(positions, dtype=wp.vec3, device=device),
        node_D0=wp.array(gradients, dtype=wp.vec3, device=device),
        elem_nodes=wp.array(elems, dtype=wp.int32, device=device),
        elem_h=wp.array(h_np, dtype=float, device=device),
        elem_mat=mat_array,
        surf_tris=wp.array(np.array(tris_list, dtype=np.int32), dtype=wp.int32, device=device),
        elem_eas_alpha=wp.array(eas_alpha_np, dtype=float, device=device),
        elem_fiber_cos=wp.array(fiber_cos_np, dtype=float, device=device),
        elem_fiber_sin=wp.array(fiber_sin_np, dtype=float, device=device),
        device=device,
    )
