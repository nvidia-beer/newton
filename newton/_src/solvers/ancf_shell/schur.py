# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linearized shell response to a small rigid interface velocity vector.

The shell interior is eliminated with its existing SPD linear solver. Boundary
position and velocity maps remain separate: prescribed beads do not use the
interior Newmark velocity/displacement relation. This module changes the
interface iteration matrix, never the physical mass or transferred wrench.
"""

import numpy as np
import warp as wp

from .kernels_assembly import add_diag_to_blk_values_batched
from .solver_ancf_shell import HHT_ALPHA, HHT_BETA, HHT_GAMMA, PcgSolverBatched, pointwise_scale


@wp.kernel(enable_backward=False)
def _prepare_systems(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    free: wp.array[float],
    n: int,
    nnz: int,
    nrhs: int,
    systems: wp.array[float],
):
    index = wp.tid()
    system = index // nnz
    env = system // nrhs
    local = index % nnz
    block = local // 36
    column = columns[block] * 6 + local % 6
    # Binary search is only used during preparation, not in PCG iterations.
    lo, hi = int(0), n // 6
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= block:
            lo = mid
        else:
            hi = mid
    row = lo * 6 + (local % 36) // 6
    value = float(0.0)
    if free[env * n + row] != 0.0 and free[env * n + column] != 0.0:
        value = values[env * nnz + local]
    elif row == column:
        value = 1.0
    systems[index] = value


@wp.kernel(enable_backward=False)
def _prepare_rhs(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    free: wp.array[float],
    boundary: wp.array[float],
    n: int,
    nnz: int,
    nrhs: int,
    boundary_scale: float,
    rhs: wp.array[float],
):
    index = wp.tid()
    system, row = index // n, index % n
    env = system // nrhs
    value = float(0.0)
    if free[env * n + row] != 0.0:
        for block in range(offsets[row // 6], offsets[row // 6 + 1]):
            for col in range(6):
                other = columns[block] * 6 + col
                if free[env * n + other] == 0.0:
                    value -= values[env * nnz + block * 36 + row % 6 * 6 + col] * boundary[system * n + other]
    rhs[index] = boundary_scale * value


@wp.kernel(enable_backward=False)
def _response_velocities(
    displacement: wp.array[float],
    boundary: wp.array[float],
    free: wp.array[float],
    n: int,
    nrhs: int,
    dt: float,
    velocity_factor: float,
    velocity: wp.array[float],
):
    i = wp.tid()
    env = (i // n) // nrhs
    row = i % n
    if free[env * n + row] != 0.0:
        velocity[i] = velocity_factor * displacement[i]
    else:
        displacement[i] = dt * boundary[i]
        velocity[i] = boundary[i]


@wp.kernel(enable_backward=False)
def _project_impedance(
    boundary: wp.array[float],
    displacement: wp.array[float],
    velocity: wp.array[float],
    mass: wp.array[float],
    contact: wp.array[float],
    n: int,
    nrhs: int,
    force_weight_dt: float,
    impedance: wp.array3d[wp.float64],
):
    env, i, j = wp.tid()
    value = wp.float64(0.0)
    for row in range(n):
        response = mass[env * n + row] * velocity[(env * nrhs + j) * n + row]
        response += force_weight_dt * contact[env * n + row] * displacement[(env * nrhs + j) * n + row]
        value += wp.float64(boundary[(env * nrhs + i) * n + row]) * wp.float64(response)
    impedance[env, i, j] = value


@wp.kernel(enable_backward=False)
def _repeat_geometry(
    positions: wp.array[wp.vec3],
    directors: wp.array[wp.vec3],
    free: wp.array[float],
    n_nodes: int,
    nrhs: int,
    x: wp.array[wp.vec3],
    d: wp.array[wp.vec3],
    mask: wp.array[float],
):
    index = wp.tid()
    node = index % n_nodes
    env = (index // n_nodes) // nrhs
    source = env * n_nodes + node
    x[index] = positions[source]
    d[index] = directors[source]
    for row in range(6):
        mask[index * 6 + row] = free[source * 6 + row]


class ShellSchurResponse:
    """Eliminate shell interior corrections for several interface directions.

    Layout of boundary/response arrays is [tire, interface direction, shell DOF].
    The contact diagonal is an iteration approximation, not a replacement force
    law. A nonlinear caller must verify the original interface residual.
    """

    def __init__(self, offsets, columns, n_envs, n_interface, device, max_iters=10):
        self.n = (len(offsets) - 1) * 6
        self.nnz = len(columns) * 36
        self.n_envs = n_envs
        self.n_interface = n_interface
        self.device = device
        self.offsets = wp.array(offsets, dtype=int, device=device)
        self.columns = wp.array(columns, dtype=int, device=device)
        count = n_envs * n_interface
        self.pcg = PcgSolverBatched(count, self.n, self.nnz, device, max_iters=max_iters)
        self.pcg.set_graph(np.asarray(offsets), np.asarray(columns))
        self.values = wp.zeros(count * self.nnz, dtype=float, device=device)
        self.rhs = wp.zeros(count * self.n, dtype=float, device=device)
        self.displacement = wp.zeros_like(self.rhs)
        self.velocity = wp.zeros_like(self.rhs)
        self.impedance = wp.zeros((n_envs, n_interface, n_interface), dtype=wp.float64, device=device)
        self._positions = wp.zeros(count * (self.n // 6), dtype=wp.vec3, device=device)
        self._directors = wp.zeros_like(self._positions)

    def update_geometry(self, positions, directors, free):
        wp.launch(
            _repeat_geometry,
            dim=self._positions.size,
            inputs=[
                positions,
                directors,
                free,
                self.n // 6,
                self.n_interface,
                self._positions,
                self._directors,
                self.pcg.coarse_free,
            ],
            device=self.device,
        )
        self.pcg.coarse_x = self._positions
        self.pcg.coarse_D = self._directors

    def solve_displacement(self, values, free, boundary, boundary_scale, extra_rhs=None):
        """Solve the condensed interior systems for the prescribed boundary columns.

        ``extra_rhs`` runs after the boundary right-hand side is built and before
        the PCG solve, so a caller can add its own column (see
        :class:`CoupledShellNewton`).
        """
        n, nnz, k = self.n, self.nnz, self.n_interface
        wp.launch(
            _prepare_systems,
            dim=self.values.size,
            inputs=[self.offsets, self.columns, values, free, n, nnz, k, self.values],
            device=self.device,
        )
        wp.launch(
            _prepare_rhs,
            dim=self.rhs.size,
            inputs=[self.offsets, self.columns, values, free, boundary, n, nnz, k, boundary_scale, self.rhs],
            device=self.device,
        )
        if extra_rhs is not None:
            extra_rhs()
        self.pcg.solve(
            self.offsets, self.columns, self.values, self.rhs, self.displacement, compute_residual_report=False
        )

    def solve(self, values, free, boundary, mass, contact, dt, velocity_factor, boundary_scale, force_weight):
        """Build the condensed impulse response without advancing shell state."""
        n, k = self.n, self.n_interface
        self.solve_displacement(values, free, boundary, boundary_scale)
        wp.launch(
            _response_velocities,
            dim=self.velocity.size,
            inputs=[self.displacement, boundary, free, n, k, dt, velocity_factor, self.velocity],
            device=self.device,
        )
        wp.launch(
            _project_impedance,
            dim=(self.n_envs, k, k),
            inputs=[boundary, self.displacement, self.velocity, mass, contact, n, k, dt * force_weight, self.impedance],
            device=self.device,
        )


@wp.kernel(enable_backward=False)
def _rigid_motion_map(
    bodies: wp.array[int],
    poses: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    directors: wp.array[wp.vec3],
    n_nodes: int,
    n_interface: int,
    boundary: wp.array[float],
):
    tire, node, column = wp.tid()
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    if column < 3:
        linear[column] = 1.0
    else:
        angular[column - 3] = 1.0
    p = positions[tire * n_nodes + node]
    d = directors[tire * n_nodes + node]
    # Shell coordinates are Y-up; Newton's public articulation maps are Z-up.
    centre = wp.transform_point(poses[bodies[tire]], com[bodies[tire]])
    v = linear + wp.cross(angular, wp.vec3(p[2], p[0], p[1]) - centre)
    vd = wp.cross(angular, wp.vec3(d[2], d[0], d[1]))
    offset = (tire * n_interface + column) * n_nodes * 6 + node * 6
    boundary[offset] = v[1]
    boundary[offset + 1] = v[2]
    boundary[offset + 2] = v[0]
    boundary[offset + 3] = vd[1]
    boundary[offset + 4] = vd[2]
    boundary[offset + 5] = vd[0]


@wp.kernel(enable_backward=False)
def _project_rigid_impedance(
    jacobian: wp.array3d[float],
    rows: wp.array[int],
    local: wp.array3d[wp.float64],
    projected: wp.array3d[wp.float64],
):
    tire, i, j = wp.tid()
    base = rows[tire] * 6
    value = wp.float64(0.0)
    for row in range(6):
        inner = wp.float64(0.0)
        for col in range(6):
            inner += local[tire, row, col] * wp.float64(jacobian[0, base + col, j])
        value += wp.float64(jacobian[0, base + row, i]) * inner
    projected[tire, i, j] = value


class RigidShellSchurResponse:
    """Approximate the velocity interface Jacobian using the shell tangent.

    Six local spindle modes suffice regardless of the number of vehicle joints;
    their impulse response is mapped into joint coordinates with J.T @ Z @ J.
    Supports one articulation and one spindle per implicit tire. This estimate
    lags geometry and uses the contact majorizer, omitting follower-force
    derivatives. It updates only iteration scratch arrays; callers must still
    verify the original nonlinear response and retain full reaction wrenches.
    """

    def __init__(self, shell, model, body_indices):
        k = model.joint_dof_count
        bodies = np.asarray(body_indices, dtype=np.int32)
        if model.articulation_count != 1 or not 0 < k <= 64:
            raise ValueError("Shell response condensation requires one articulation with 1–64 velocities.")
        if bodies.shape != (shell.n_envs,):
            raise ValueError("Provide one rigid spindle body for each tire in the shell solver.")
        if not hasattr(shell, "_update_K_eff_inplace_batched") or shell._dirichlet_idx is None:
            raise ValueError("Shell response condensation requires implicit integration and prescribed beads.")
        if shell.terrain is not None and not shell.terrain.rigid:
            raise ValueError("Shell response condensation currently supports rigid terrain only.")
        children = model.joint_child.numpy()
        start = int(model.articulation_start.numpy()[0])
        end = int(model.articulation_end.numpy()[0])
        rows = []
        for body in bodies:
            found = np.flatnonzero(children[start:end] == body)
            if len(found) != 1:
                raise ValueError(f"Spindle body {body} must belong to the articulation.")
            rows.append(found[0])
        self.shell, self.model, self.device = shell, model, model.device
        self.bodies = wp.array(bodies, dtype=int, device=self.device)
        self.rows = wp.array(rows, dtype=int, device=self.device)
        self.jacobian = wp.zeros((1, model.max_joints_per_articulation * 6, k), dtype=float, device=self.device)
        self.mass = wp.zeros((1, k, k), dtype=float, device=self.device)
        self.impedance = wp.zeros((shell.n_envs, k, k), dtype=wp.float64, device=self.device)
        self._spatial_inertia = wp.zeros(model.body_count, dtype=wp.spatial_matrix, device=self.device)
        self._motion_subspace = wp.zeros(k, dtype=wp.spatial_vector, device=self.device)
        self._contact = wp.zeros_like(shell.K_contact_diag)

    def build_tangent(self, dt):
        """Scale K_eff and add the contact majorizer for one condensed solve.

        Returns the HHT velocity factor cv = gamma / (beta dt) and the scale the
        prescribed-bead right-hand side must use with this tangent.
        """
        s, dev = self.shell, self.device
        cv = HHT_GAMMA / (HHT_BETA * dt)
        s._update_K_eff_inplace_batched((1.0 + HHT_ALPHA) * (1.0 + s._alpha_damp * cv))
        wp.launch(
            pointwise_scale,
            dim=self._contact.size,
            inputs=[s.K_contact_diag, 1.0 + HHT_ALPHA, self._contact],
            device=dev,
        )
        wp.launch(
            add_diag_to_blk_values_batched,
            dim=self._contact.size,
            inputs=[self._contact, s.blk_offsets, s.blk_columns, s.bsr_values_batched, s.pcg.n_dof, s._nnz],
            device=dev,
        )
        # Prescribed bead velocity has derivative B, whereas free-node velocity
        # has derivative cv * dx. Rayleigh damping therefore needs this ratio.
        boundary_scale = (dt + s._alpha_damp) / (1.0 + s._alpha_damp * cv)
        return cv, boundary_scale
