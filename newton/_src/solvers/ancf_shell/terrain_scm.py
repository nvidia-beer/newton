# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Heightfield terrain contact for ANCF shell nodes, rigid or deformable (Soil Contact Model).

Replaces the flat analytic plane of :mod:`kernels_contact` with a 2D elevation grid.  The
grid follows :class:`newton.Heightfield`: ``h[row, col]`` with row = world Y and col = world X,
spanning ``[-hx, hx] x [-hy, hy]`` (Z-up).  ANCF nodes are Y-up ``(x_lat, y_up, z_fwd)`` and
are mapped to Z-up with ``(x, y, z) -> (z, x, y)`` before the lookup.

Why a heightfield and not a mesh: the tire is a point cloud (one contact per node), so the
surface query is one bilinear lookup and its gradient - O(1), no BVH, graph-capturable - and
the plastic soil state lives on the same grid.

Soil model
----------
Port of Project Chrono ``SCMTerrain`` (``src/chrono_vehicle/terrain/SCMTerrain.cpp``, force
loop lines 1370-1521), node-centric instead of grid-centric.  Per node, with ``pen`` the
penetration below the *current* (rutted) surface and ``z`` the sinkage below the *undeformed*
surface, both measured along the local normal:

* elastic branch (every NR iteration, in the tangent):  ``fn = kn*pen + kd*max(-v_n, 0)``
  - identical to the flat-plane kernel; Chrono's ``elastic_K`` is ``kn / A_node``.
* yield (once per substep, before the predictor):  if ``kn*pen/A_node > sigma_yield(cell)``
  the soil yields to Bekker-Wong ``sigma = (Kc/b + Kphi) * z**n``, the cell stores
  ``sigma_yield = sigma`` and lowers its surface by ``z_p = z - sigma*A_node/kn`` (the plastic
  sinkage).  Both are monotone (``wp.atomic_max``) so ruts persist and unloading is elastic.
* shear:  Janosi-Hanamoto ``tau = (c + sigma*tan(phi)) * (1 - exp(-j/K_j))`` with ``j`` the
  shear displacement accumulated per node while in contact; ``sigma*tan(phi)*A = mu*fn``, so
  the regularised Coulomb friction of the flat kernel is the ``j >> K_j`` limit.

With the soil disabled (``rigid=True``) the plastic update is skipped, ``z_p == 0`` and the
kernel is a pure penalty contact against the heightfield.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.struct
class TerrainGrid:
    nrow: wp.int32
    ncol: wp.int32
    hx: wp.float32  # half-extent X [m]
    hy: wp.float32  # half-extent Y [m]
    dx: wp.float32  # cell size X [m]
    dy: wp.float32  # cell size Y [m]


@wp.struct
class SoilParams:
    kphi: wp.float32  # Bekker frictional modulus [Pa/m^n]
    kc: wp.float32  # Bekker cohesive modulus [Pa/m^(n-1)]
    n_exp: wp.float32  # Bekker sinkage exponent
    cohesion: wp.float32  # Mohr-Coulomb cohesion [Pa]
    mu: wp.float32  # tan(friction angle) for soil, rubber-ground mu for rigid terrain
    janosi_k: wp.float32  # Janosi shear displacement scale [m]; <= 0 disables (pure Coulomb)
    inv_b: wp.float32  # 1/b, b = contact patch width (tire width) [1/m]


@wp.func
def _cell(grid: TerrainGrid, origin: wp.array[wp.vec2], x: float, y: float):
    """Clamped fractional grid coordinates (col_f, row_f) and integer cell (c0, r0) with weights.

    ``origin[0]`` is the world (x, y) of the grid centre; a device array (not a struct field) so
    it can be updated in place without re-capturing the graph."""
    fx = wp.clamp((x - origin[0][0] + grid.hx) / grid.dx, 0.0, float(grid.ncol - 1) - 1.0e-4)
    fy = wp.clamp((y - origin[0][1] + grid.hy) / grid.dy, 0.0, float(grid.nrow - 1) - 1.0e-4)
    c0 = int(fx)
    r0 = int(fy)
    return c0, r0, fx - float(c0), fy - float(r0)


@wp.func
def _surface(
    grid: TerrainGrid,
    origin: wp.array[wp.vec2],
    h0: wp.array2d[float],
    z_p: wp.array2d[float],
    x: float,
    y: float,
):
    """Bilinear current surface height h0 - z_p, its undeformed height h0, and the unit normal."""
    c0, r0, tx, ty = _cell(grid, origin, x, y)
    c1 = c0 + 1
    r1 = r0 + 1
    a00 = h0[r0, c0] - z_p[r0, c0]
    a10 = h0[r0, c1] - z_p[r0, c1]
    a01 = h0[r1, c0] - z_p[r1, c0]
    a11 = h0[r1, c1] - z_p[r1, c1]
    h = (a00 * (1.0 - tx) + a10 * tx) * (1.0 - ty) + (a01 * (1.0 - tx) + a11 * tx) * ty
    dhdx = ((a10 - a00) * (1.0 - ty) + (a11 - a01) * ty) / grid.dx
    dhdy = ((a01 - a00) * (1.0 - tx) + (a11 - a10) * tx) / grid.dy
    n = wp.normalize(wp.vec3(-dhdx, -dhdy, 1.0))
    h_undeformed = (h0[r0, c0] * (1.0 - tx) + h0[r0, c1] * tx) * (1.0 - ty) + (
        h0[r1, c0] * (1.0 - tx) + h0[r1, c1] * tx
    ) * ty
    return h, h_undeformed, n


@wp.kernel
def _node_area(
    node_x0: wp.array[wp.vec3],
    elem_nodes: wp.array2d[wp.int32],
    area: wp.array[float],
):
    """dim = n_elems.  Lumped rest surface area per node (quad area / 4 to each corner)."""
    e = wp.tid()
    p0 = node_x0[elem_nodes[e, 0]]
    p1 = node_x0[elem_nodes[e, 1]]
    p2 = node_x0[elem_nodes[e, 2]]
    p3 = node_x0[elem_nodes[e, 3]]
    a = 0.5 * (wp.length(wp.cross(p1 - p0, p2 - p0)) + wp.length(wp.cross(p2 - p0, p3 - p0)))
    for k in range(4):
        wp.atomic_add(area, elem_nodes[e, k], 0.25 * a)


@wp.kernel
def _update_plastic(
    node_x: wp.array[wp.vec3],  # ANCF Y-up, flat [N*n_nodes]
    node_xd: wp.array[wp.vec3],
    n_nodes: int,
    grid: TerrainGrid,
    origin: wp.array[wp.vec2],
    soil: SoilParams,
    kn: float,
    dt: float,
    h0: wp.array2d[float],
    node_area: wp.array[float],  # [n_nodes]
    z_p: wp.array2d[float],  # plastic sinkage per cell [m], monotone
    sigma_yield: wp.array2d[float],  # yield pressure per cell [Pa], monotone
    kshear: wp.array[float],  # Janosi shear displacement per node [m]
):
    """Once per substep from the state at step begin (Chrono ``ComputeInternalForces`` order)."""
    tid = wp.tid()
    p = node_x[tid]
    x = p[2]
    y = p[0]
    z = p[1]
    h, h_und, n = _surface(grid, origin, h0, z_p, x, y)
    pen = (h - z) * n[2]
    if pen <= 0.0:
        kshear[tid] = 0.0
        return

    v = node_xd[tid]
    v_zu = wp.vec3(v[2], v[0], v[1])
    vt = v_zu - wp.dot(v_zu, n) * n
    kshear[tid] = kshear[tid] + wp.length(vt) * dt

    a = node_area[tid % n_nodes]
    sigma_e = kn * pen / a  # elastic trial pressure
    c0, r0, tx, ty = _cell(grid, origin, x, y)
    c = c0 + int(tx + 0.5)
    r = r0 + int(ty + 0.5)
    if sigma_e > sigma_yield[r, c]:
        z_tot = wp.max((h_und - z) * n[2], 0.0)
        sigma_b = (soil.kc * soil.inv_b + soil.kphi) * wp.pow(z_tot, soil.n_exp)
        wp.atomic_max(sigma_yield, r, c, sigma_b)
        wp.atomic_max(z_p, r, c, z_tot - sigma_b * a / kn)


@wp.kernel
def _apply_terrain_contact(
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    n_nodes: int,
    grid: TerrainGrid,
    origin: wp.array[wp.vec2],
    soil: SoilParams,
    kn: float,
    kd: float,
    v_reg: float,
    c_v: float,  # gamma/(beta*dt): velocity -> displacement factor for the HHT tangent
    h0: wp.array2d[float],
    z_p: wp.array2d[float],
    node_area: wp.array[float],
    kshear: wp.array[float],
    global_f: wp.array[float],  # flat [N*n_nodes*6] DOF vector (Y-up)
    K_contact_diag: wp.array[float],
    node_f: wp.array[wp.vec3],  # per-node terrain force (Z-up), diagnostics
):
    """dim = N*n_nodes.  Penalty normal + Janosi/Coulomb tangential force against the current surface."""
    tid = wp.tid()
    p = node_x[tid]
    x = p[2]
    y = p[0]
    z = p[1]
    h, _h_und, n = _surface(grid, origin, h0, z_p, x, y)
    pen = (h - z) * n[2]
    if pen <= 0.0:
        node_f[tid] = wp.vec3(0.0)
        return

    v = node_xd[tid]
    v_zu = wp.vec3(v[2], v[0], v[1])
    vn = -wp.dot(v_zu, n)  # positive = closing
    fn = kn * pen + kd * wp.max(vn, 0.0)

    vt = v_zu - wp.dot(v_zu, n) * n
    vt_mag = wp.sqrt(wp.dot(vt, vt) + 1.0e-12)
    g = wp.tanh(vt_mag / v_reg)
    u = vt / vt_mag

    a = node_area[tid % n_nodes]
    ft_max = soil.mu * fn + soil.cohesion * a
    if soil.janosi_k > 0.0:
        ft_max = ft_max * (1.0 - wp.exp(-kshear[tid] / soil.janosi_k))
    ft = -ft_max * g * u

    f_zu = fn * n + ft
    node_f[tid] = f_zu
    base = tid * 6
    # Z-up -> Y-up: (x, y, z) -> (y, z, x)
    wp.atomic_add(global_f, base + 0, f_zu[1])
    wp.atomic_add(global_f, base + 1, f_zu[2])
    wp.atomic_add(global_f, base + 2, f_zu[0])

    # Diagonal of -df/du (see kernels_contact._contact_tangent_diag for the flat-plane case):
    #   normal   : (kn + kd*c_v*[closing]) * n_i^2
    #   friction : ft_max*c_v*( g/|vt| * (1 - n_i^2 - u_i^2) + g' * u_i^2 ),  g' = sech^2/v_reg
    k_n = kn
    if vn > 0.0:
        k_n = k_n + kd * c_v
    g_over_v = g / vt_mag
    g_prime = (1.0 - g * g) / v_reg
    for i in range(3):
        ni2 = n[i] * n[i]
        ui2 = u[i] * u[i]
        k_i = k_n * ni2 + ft_max * c_v * (g_over_v * wp.max(1.0 - ni2 - ui2, 0.0) + g_prime * ui2)
        # Z-up axis i lands on Y-up DOF (i + 2) % 3 ... explicitly: x->2, y->0, z->1
        j = 2
        if i == 1:
            j = 0
        elif i == 2:
            j = 1
        wp.atomic_add(K_contact_diag, base + j, k_i)


class TerrainSCM:
    """Heightfield terrain with optional Bekker-Wong / Janosi-Hanamoto soil for :class:`SolverANCFShell`.

    Attach with ``solver.terrain = TerrainSCM(...)`` before ``solver.capture_graph()``.

    Args:
        heights: Undeformed surface ``(nrow, ncol)`` in metres, row = Y, col = X (Z-up).
        hx: Half-extent along X [m].
        hy: Half-extent along Y [m].
        node_x0: Rest node positions of one tire (Y-up), for the lumped node areas.
        elem_nodes: Element connectivity ``(n_elems, 4)`` of one tire.
        n_envs: Number of tires (ANCF environments).
        kphi: Bekker frictional modulus [Pa/m^n].
        kc: Bekker cohesive modulus [Pa/m^(n-1)].
        n_exp: Bekker sinkage exponent.
        cohesion: Mohr-Coulomb cohesion [Pa].
        friction_angle_deg: Soil internal friction angle [deg]; ``tan`` of it is the shear coefficient.
        janosi_k: Janosi shear displacement scale [m]; 0 disables the shear build-up.
        patch_width: Contact patch width ``b`` in Bekker's ``Kc/b`` [m]; the tire width.
        rigid: Skip the plastic update: heightfield penalty contact only.
        mu_rigid: Rubber-ground friction used when ``rigid`` is True.
        origin: World (x, y) of the grid centre [m]; ``(0, 0)`` keeps the grid centred on the origin.
        device: CUDA device.
    """

    def __init__(
        self,
        heights: np.ndarray,
        hx: float,
        hy: float,
        node_x0: wp.array[wp.vec3],
        elem_nodes: wp.array2d[wp.int32],
        n_envs: int,
        kphi: float = 2.0e6,
        kc: float = 0.0,
        n_exp: float = 1.1,
        cohesion: float = 0.0,
        friction_angle_deg: float = 30.0,
        janosi_k: float = 0.01,
        patch_width: float = 0.3,
        rigid: bool = False,
        mu_rigid: float = 0.9,
        origin: tuple[float, float] = (0.0, 0.0),
        device: str = "cuda:0",
    ):
        heights = np.asarray(heights, dtype=np.float32)
        if heights.ndim != 2 or min(heights.shape) < 2:
            raise ValueError(f"heights must be (nrow>=2, ncol>=2), got {heights.shape}")
        nrow, ncol = heights.shape
        self.nrow, self.ncol = int(nrow), int(ncol)
        self.hx, self.hy = float(hx), float(hy)
        self.rigid = bool(rigid)
        self.device = device

        self.grid = TerrainGrid()
        self.grid.nrow = self.nrow
        self.grid.ncol = self.ncol
        self.grid.hx = self.hx
        self.grid.hy = self.hy
        self.grid.dx = 2.0 * self.hx / (self.ncol - 1)
        self.grid.dy = 2.0 * self.hy / (self.nrow - 1)
        self.origin = (float(origin[0]), float(origin[1]))
        self._origin = wp.array([wp.vec2(self.origin[0], self.origin[1])], dtype=wp.vec2, device=device)

        self.soil = SoilParams()
        self.soil.kphi = float(kphi)
        self.soil.kc = float(kc)
        self.soil.n_exp = float(n_exp)
        self.soil.cohesion = 0.0 if rigid else float(cohesion)
        self.soil.mu = float(mu_rigid) if rigid else math.tan(math.radians(friction_angle_deg))
        self.soil.janosi_k = 0.0 if rigid else float(janosi_k)
        self.soil.inv_b = 1.0 / float(patch_width)

        n_nodes = node_x0.shape[0]
        self.n_nodes = int(n_nodes)
        self.n_envs = int(n_envs)
        self.h0 = wp.array(heights, dtype=float, device=device)
        self.z_p = wp.zeros((self.nrow, self.ncol), dtype=float, device=device)
        self.sigma_yield = wp.zeros((self.nrow, self.ncol), dtype=float, device=device)
        self.kshear = wp.zeros(self.n_envs * self.n_nodes, dtype=float, device=device)
        self.node_f = wp.zeros(self.n_envs * self.n_nodes, dtype=wp.vec3, device=device)
        self.node_area = wp.zeros(self.n_nodes, dtype=float, device=device)
        wp.launch(_node_area, dim=elem_nodes.shape[0], inputs=[node_x0, elem_nodes, self.node_area], device=device)

    # -- solver hooks -------------------------------------------------------------------------

    def update_plastic(self, node_x: wp.array[wp.vec3], node_xd: wp.array[wp.vec3], kn: float, dt: float) -> None:
        """Yield test and rut update from the state at substep begin.  No-op for rigid terrain."""
        if self.rigid:
            return
        wp.launch(
            _update_plastic,
            dim=self.n_envs * self.n_nodes,
            inputs=[
                node_x,
                node_xd,
                self.n_nodes,
                self.grid,
                self._origin,
                self.soil,
                float(kn),
                float(dt),
                self.h0,
                self.node_area,
                self.z_p,
                self.sigma_yield,
                self.kshear,
            ],
            device=self.device,
        )

    def apply_contact(
        self,
        node_x: wp.array[wp.vec3],
        node_xd: wp.array[wp.vec3],
        kn: float,
        kd: float,
        v_reg: float,
        c_v: float,
        global_f: wp.array[float],
        K_contact_diag: wp.array[float],
    ) -> None:
        """Add the terrain contact force and its diagonal tangent (called every NR iteration)."""
        wp.launch(
            _apply_terrain_contact,
            dim=self.n_envs * self.n_nodes,
            inputs=[
                node_x,
                node_xd,
                self.n_nodes,
                self.grid,
                self._origin,
                self.soil,
                float(kn),
                float(kd),
                float(v_reg),
                float(c_v),
                self.h0,
                self.z_p,
                self.node_area,
                self.kshear,
                global_f,
                K_contact_diag,
                self.node_f,
            ],
            device=self.device,
        )
