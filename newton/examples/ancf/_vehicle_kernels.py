# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Device kernels shared by full-vehicle coupling and tire visualization.

Rigid poses use Z-up coordinates; ANCF tires use Y-up coordinates. Keep these
conversions here so flat-ground, terrain, sand, and telemetry use identical bead
kinematics and reaction transfer.
"""

import warp as wp


@wp.kernel(enable_backward=False)
def advance_substep_pair(remaining: wp.array[int]):
    remaining[0] -= 1


@wp.kernel
def stage_interface_kinematics(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    bodies: wp.array[int],
    mj_bodies: wp.array[int],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    cvel: wp.array2d[wp.spatial_vector],
    subtree_com: wp.array2d[wp.vec3],
    root: wp.array[int],
):
    # Newton stores linear/angular velocity at each body's COM. MuJoCo stores
    # angular/linear velocity at the articulation's subtree COM. Transport the
    # twist to that reference so the existing bead kernel reads identical data.
    i = wp.tid()
    body, mj = bodies[i], mj_bodies[i]
    pose = body_q[body]
    pos = wp.transform_get_translation(pose)
    q = wp.transform_get_rotation(pose)
    omega = wp.spatial_bottom(body_qd[body])
    velocity = wp.spatial_top(body_qd[body]) - wp.cross(omega, wp.quat_rotate(q, body_com[body]))
    reference_velocity = velocity + wp.cross(omega, subtree_com[0, root[mj]] - pos)
    xpos[0, mj] = pos
    xquat[0, mj] = wp.quat(q[3], q[0], q[1], q[2])
    cvel[0, mj] = wp.spatial_vector(
        omega[0],
        omega[1],
        omega[2],
        reference_velocity[0],
        reference_velocity[1],
        reference_velocity[2],
    )


@wp.kernel
def prescribe_beads(
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    cvel: wp.array2d[wp.spatial_vector],
    subtree_com: wp.array2d[wp.vec3],  # mjw_data.subtree_com — cvel's linear reference point
    body_rootid: wp.array[int],  # mjw_model.body_rootid
    spindle_mj_arr: wp.array[wp.int32],
    bead_idx: wp.array[wp.int32],
    bead_rest: wp.array[wp.vec3],
    bead_D0: wp.array[wp.vec3],
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_xdd: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
    n_bead: int,
    n_nodes: int,
    inv_dt: float,  # >0: write bead accel as d(prescribed velocity)/dt; 0: leave accel alone
    vel_predict_dt: float,  # >0: extrapolate hub translation and rotation to t_{n+1}
):
    """Prescribe bead node position, director and velocities from the spindle pose.

    MuJoCo Z-up -> ANCF Y-up: (x,y,z) -> (y, z, x).
    """
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_idx = env * n_nodes + bead_idx[i]

    mj = spindle_mj_arr[env]
    pos_zu = xpos[0, mj]
    # mujoco_warp stores xquat in MuJoCo order (w, x, y, z) inside a wp.quat whose
    # slots are (x, y, z, w) — reorder before wp.quat_rotate.
    qm = xquat[0, mj]
    q = wp.quat(qm[1], qm[2], qm[3], qm[0])
    sv = cvel[0, mj]

    w_mj = wp.vec3(sv[0], sv[1], sv[2])
    # cvel is "com-based": its linear part is the velocity of the point
    # subtree_com[root] (the whole vehicle's COM), not of the body origin.
    # Transport it to the spindle origin (same correction Newton applies in
    # mujoco/kernels.py mj_body_acceleration).  Without it the bead velocity is
    # wrong by omega x (xpos - com): vehicle pitch rate times the 1.65 m
    # half-wheelbase, opposite sign front vs rear.
    com = subtree_com[0, body_rootid[mj]]
    v_mj = wp.vec3(sv[3], sv[4], sv[5]) + wp.cross(w_mj, pos_zu - com)

    r = bead_rest[i]
    d0 = bead_D0[i]

    # Bead offset is defined in the ANCF frame (x_lat, y_up, z_fwd); express it
    # in the spindle's local MuJoCo frame and rotate by the spindle orientation
    # (xquat already contains the axle spin).
    r_mj = wp.vec3(r[2], r[0], r[1])
    d_mj = wp.vec3(d0[2], d0[0], d0[1])
    speed = wp.length(w_mj)
    if speed > 0.0:
        q = wp.quat_from_axis_angle(w_mj / speed, speed * vel_predict_dt) * q
    r_w = wp.quat_rotate(q, r_mj)
    d_w = wp.quat_rotate(q, d_mj)

    p_w = pos_zu + v_mj * vel_predict_dt + r_w

    # Material velocity of a hub-attached point: v_hub + omega x r.
    vel_w = v_mj + wp.cross(w_mj, r_w)
    dd_w = wp.cross(w_mj, d_w)

    v_new = wp.vec3(vel_w[1], vel_w[2], vel_w[0])
    dd_new = wp.vec3(dd_w[1], dd_w[2], dd_w[0])

    # Bead acceleration consistent with the prescribed velocity (finite difference
    # of the previous prescribed velocity); only on the pre-step call.
    if inv_dt > 0.0:
        node_xdd[global_idx] = (v_new - node_xd[global_idx]) * inv_dt
        node_Ddd[global_idx] = (dd_new - node_Dd[global_idx]) * inv_dt

    node_x[global_idx] = wp.vec3(p_w[1], p_w[2], p_w[0])
    node_xd[global_idx] = v_new
    node_D[global_idx] = wp.vec3(d_w[1], d_w[2], d_w[0])
    node_Dd[global_idx] = dd_new


_CONTACT_SPIKE_STRIDE = 4
"""Draw every Nth node's spike so a wide contact patch reads as a sparse outline, not a dense forest."""


@wp.kernel
def gather_contact_spikes(
    node_x: wp.array[wp.vec3],  # ANCF Y-up, all envs flat
    n_nodes: int,  # nodes per tire, for the per-tire local index
    ground_y: float,  # ground level in ANCF Y-up
    vis_scale: float,  # penetration amplification for visibility
    stride: int,  # draw every ``stride``-th node of each tire (1 = every node)
    up_axis: int,  # viewer up axis: 2 converts to Z-up, (x, y, z) -> (z, x, y); 1 keeps Y-up
    line_starts: wp.array[wp.vec3],
    line_ends: wp.array[wp.vec3],
):
    """GPU-only contact visualization: a spike of height pen*vis_scale per sampled penetrating node.

    Every node emits a segment from its ground-plane footprint; a node that is not sampled or not
    penetrating gets a zero-length (invisible) one.
    """
    i = wp.tid()
    p = node_x[i]
    pen = ground_y - p[1]  # positive = inside ground
    if up_axis == 2:
        base = wp.vec3(p[2], p[0], ground_y)
        tip = wp.vec3(base[0], base[1], base[2] + pen * vis_scale)
    else:
        base = wp.vec3(p[0], ground_y, p[2])
        tip = wp.vec3(base[0], base[1] + pen * vis_scale, base[2])
    line_starts[i] = base
    if pen > 0.0 and (i % n_nodes) % stride == 0:
        line_ends[i] = tip
    else:
        line_ends[i] = base


@wp.kernel
def fill_spindle_positions(
    xpos: wp.array2d[wp.vec3],
    spindle_mj_arr: wp.array[wp.int32],
    out: wp.array[wp.vec3],
    n_bead: int,
):
    """Fill spoke-start positions from xpos[0, spindle_mj_arr[env]] (already Z-up)."""
    tid = wp.tid()
    env = tid // n_bead
    out[tid] = xpos[0, spindle_mj_arr[env]]
