# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""FEM-tracked capsule colliders as native Newton kinematic bodies.

Each finger of ``K`` keypoints gives ``K - 1`` capsules between
consecutive keypoints; the per-capsule radius is baked into the asset
(roughly half the finger cross-section). Keypoint world positions
follow the FEM via the barycentric binding stored in
``customData["finger_skeletons"]``, so each capsule's endpoints
deform with chamber inflation and stem motion at no extra cost.

This module is the build-phase + per-substep glue that turns those
spline-tracked capsules into *real* Newton bodies + shapes integrated
by the standard collision pipeline:

* :func:`register_kinematic_capsule_bodies` — called once during
  ``ModelBuilder`` setup. Emits one ``is_kinematic=True`` body and one
  ``add_shape_capsule`` per finger segment, returns the body / shape
  ids and per-capsule half-heights so the example can plumb them into
  collision-filter pairs (capsule↔stem, same-finger capsule↔capsule).
  No physics yet — just geometry registration.
* :class:`FingerCapsuleColliders` — constructed after
  ``builder.finalize()``. Every substep its
  :meth:`update_endpoints` + :meth:`update_kinematic_state` pair runs
  inside the captured graph to push the latest FEM-tracked capsule
  poses (and their finite-differenced velocities) into the kinematic
  bodies, *before* ``model.collide`` so the broadphase / narrowphase
  see the right shapes at the right place this substep.

Detection and force application are delegated entirely to the
standard contact pipeline — there is no custom penalty kernel here.
Friction lives on each capsule's ``ShapeConfig.mu`` (set by the
caller before registration).
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ._spline_render import _decode_skeleton_dict


# =============================================================================
# Pre-finalize: register kinematic Newton bodies + capsule shapes
# =============================================================================


def _segment_to_quat(d: np.ndarray) -> tuple[float, float, float, float]:
    """Build the (x, y, z, w) quaternion rotating +Z onto direction ``d``.

    Mirrors the Warp ``_build_capsule_transforms`` kernel's branchy form
    (collinear / anti-collinear / general) so the rest-pose capsules
    rendered by Newton match exactly what the runtime pose-driver
    kernel will produce when it later updates the kinematic body_q each
    substep.
    """
    L = float(np.linalg.norm(d))
    if L < 1.0e-9:
        return (0.0, 0.0, 0.0, 1.0)
    dn = d / L
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cos_a = float(np.dot(z, dn))
    if cos_a > 0.99999:
        return (0.0, 0.0, 0.0, 1.0)
    if cos_a < -0.99999:
        # 180° rotation around X (matches the kernel branch).
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(z, dn)
    axis /= float(np.linalg.norm(axis))
    angle = float(np.arccos(cos_a))
    sh = float(np.sin(0.5 * angle))
    ch = float(np.cos(0.5 * angle))
    return (float(axis[0]) * sh, float(axis[1]) * sh, float(axis[2]) * sh, ch)


def register_kinematic_capsule_bodies(
    builder: newton.ModelBuilder,
    *,
    skeletons_dict: dict,
    cap_radii_np: np.ndarray,
    cfg_template: newton.ModelBuilder.ShapeConfig | None = None,
    label_prefix: str = "finger_cap",
) -> tuple[list[int], list[int], np.ndarray] | None:
    """Register one kinematic Newton body + capsule shape per finger segment.

    Decodes the asset's ``finger_skeletons`` customData (same payload
    consumed by :class:`FingerSplineLines`), walks the builder's current
    ``particle_q`` and ``tet_indices`` to compute the rest-pose world
    position of each keypoint, and emits ``F·(K-1)`` kinematic bodies
    each with one capsule shape. In Phase A all shapes are non-colliding
    (``has_shape_collision=False``, ``has_particle_collision=False``,
    ``density=0``) so adding them has zero effect on physics — Phase B
    will drive their poses each substep, Phase C will flip on collision
    and the custom force kernels go away.

    The bodies are created in the order ``(finger_index, segment_index)``
    matching the existing capsule layout (``cid = fi * (K-1) + s``) so
    downstream code keyed on that index can reuse it verbatim.

    Args:
        builder: Newton ``ModelBuilder`` mid-build (before
            ``finalize``). Bodies are added to whatever world block is
            currently open.
        skeletons_dict: Asset's raw ``finger_skeletons`` customData.
        cap_radii_np: Per-capsule capsule radius [m], shape
            ``(F·(K-1),)`` float32. Caller resolves from CLI override /
            asset bake / bbox fallback.
        cfg_template: Optional ``ShapeConfig`` to copy as the per-shape
            base. The helper overwrites the collision/density fields
            for Phase A regardless. When ``None`` a fresh default
            ``ShapeConfig`` is used.
        label_prefix: Per-body label prefix; bodies are labelled
            ``"{prefix}_f{finger}_s{segment}"``.

    Returns:
        ``(body_ids, shape_ids, half_heights)`` where each is length
        ``F·(K-1)`` (the first two as Python ``int`` lists, the third
        as a ``(F·(K-1),)`` float32 ndarray of cylindrical
        half-lengths). Returns ``None`` when the skeleton dict is
        missing or malformed.
    """
    decoded = _decode_skeleton_dict(skeletons_dict) if skeletons_dict else None
    if decoded is None:
        return None
    F, K, tet_idx_np, bary_np, _ = decoded
    num_caps = F * max(0, K - 1)
    if num_caps == 0:
        return None
    cap_radii = np.asarray(cap_radii_np, dtype=np.float32).reshape(-1)
    if cap_radii.size != num_caps:
        raise ValueError(
            f"cap_radii_np size {cap_radii.size} != F·(K-1)={num_caps}"
        )
    # Build keypoint rest positions: kp = Σ bary_k · particle[tet_v_k].
    particle_q = np.asarray(builder.particle_q, dtype=np.float32).reshape(-1, 3)
    tet_v = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
    tet_idx = tet_idx_np.reshape(F * K)
    bary = bary_np.reshape(F * K, 4)
    kp_world = np.zeros((F * K, 3), dtype=np.float32)
    for k in range(F * K):
        t = int(tet_idx[k])
        v = tet_v[t]
        w = bary[k]
        kp_world[k] = (
            w[0] * particle_q[v[0]]
            + w[1] * particle_q[v[1]]
            + w[2] * particle_q[v[2]]
            + w[3] * particle_q[v[3]]
        )

    cfg = (
        cfg_template.copy()
        if cfg_template is not None
        else newton.ModelBuilder.ShapeConfig()
    )
    # MuJoCo rejects free-joint bodies whose computed mass is ≤
    # ``mjMINVAL`` at ``spec.compile()`` time even though kinematic
    # bodies get a ``1e10`` armature on their free joint so the
    # implicit solve pins them regardless of nominal mass. The mass
    # therefore needs to be *positive* but doesn't need to match the
    # real capsule. If the caller forgot to set a positive density,
    # bump it up to the default (1.0).
    if cfg.density <= 0.0:
        cfg.density = 1.0

    body_ids: list[int] = []
    shape_ids: list[int] = []
    half_heights = np.zeros(num_caps, dtype=np.float32)

    for fi in range(F):
        for s in range(K - 1):
            cid = fi * (K - 1) + s
            kp_a = fi * K + s
            kp_b = fi * K + s + 1
            a = kp_world[kp_a]
            b = kp_world[kp_b]
            center = 0.5 * (a + b)
            d = b - a
            half_h = max(1.0e-4, 0.5 * float(np.linalg.norm(d)))
            half_heights[cid] = half_h
            qx, qy, qz, qw = _segment_to_quat(d)
            body_id = builder.add_body(
                xform=wp.transform(
                    wp.vec3(float(center[0]), float(center[1]), float(center[2])),
                    wp.quat(qx, qy, qz, qw),
                ),
                label=f"{label_prefix}_f{fi}_s{s}",
                is_kinematic=True,
            )
            shape_id = builder.add_shape_capsule(
                body=body_id,
                radius=float(cap_radii[cid]),
                half_height=float(half_h),
                cfg=cfg,
            )
            body_ids.append(int(body_id))
            shape_ids.append(int(shape_id))

    # Filter same-finger capsule pairs: consecutive segments within
    # one finger share a keypoint at the segment join, so their
    # capsule shapes always overlap there by construction. Without
    # filtering, the contact pipeline would emit a permanent contact
    # at every internal keypoint of every finger, effectively turning
    # the soft finger into a stack of rigidly-fused segments. Other
    # capsule-capsule pairs (across different fingers) can still
    # collide and are physically meaningful when fingers touch
    # mid-grasp. ``add_shape_collision_filter_pair`` is symmetric;
    # passing same-finger non-consecutive pairs too (e.g. f0-s0 vs
    # f0-s2) is cheap insurance against curl bringing distant
    # segments of the *same* finger into self-contact, which is also
    # geometrically guaranteed when the finger curls past ~180° — and
    # never physically meaningful for the modelled soft finger.
    if cfg.has_shape_collision:
        for fi in range(F):
            base = fi * (K - 1)
            for s_i in range(K - 1):
                for s_j in range(s_i + 1, K - 1):
                    builder.add_shape_collision_filter_pair(
                        int(shape_ids[base + s_i]),
                        int(shape_ids[base + s_j]),
                    )

    return body_ids, shape_ids, half_heights


# =============================================================================
# Endpoint update kernel
# =============================================================================


@wp.kernel
def _build_capsule_endpoints_from_keypoints(
    keypoint_world: wp.array(dtype=wp.vec3),  # (F·K,)
    cap_kp_a_idx: wp.array(dtype=wp.int32),  # (num_capsules,)
    cap_kp_b_idx: wp.array(dtype=wp.int32),
    # outputs
    cap_a: wp.array(dtype=wp.vec3),
    cap_b: wp.array(dtype=wp.vec3),
):
    cid = wp.tid()
    cap_a[cid] = keypoint_world[cap_kp_a_idx[cid]]
    cap_b[cid] = keypoint_world[cap_kp_b_idx[cid]]


@wp.kernel
def _build_capsule_transforms(
    cap_a: wp.array(dtype=wp.vec3),
    cap_b: wp.array(dtype=wp.vec3),
    out_xforms: wp.array(dtype=wp.transform),
):
    """Per-capsule ``wp.transform`` from segment endpoints.

    Newton's capsule primitive extends along its local +Z (``up_axis=Z``
    in :func:`newton.Mesh.create_capsule`), so we place the capsule's
    centre at the segment midpoint and rotate +Z onto the segment
    direction. Degenerate (zero-length) segments fall back to the
    identity quaternion — the caller never feeds those because they'd
    collapse to a sphere anyway.
    """
    cid = wp.tid()
    a = cap_a[cid]
    b = cap_b[cid]
    center = 0.5 * (a + b)
    d = b - a
    L = wp.length(d)
    if L < wp.float32(1.0e-9):
        out_xforms[cid] = wp.transform(center, wp.quat_identity())
        return
    dn = d / L
    z = wp.vec3(0.0, 0.0, 1.0)
    cos_a = wp.dot(z, dn)
    if cos_a > wp.float32(0.99999):
        q = wp.quat_identity()
    elif cos_a < wp.float32(-0.99999):
        # 180° rotation: pick any axis perpendicular to Z.
        q = wp.quat(1.0, 0.0, 0.0, 0.0)
    else:
        axis = wp.normalize(wp.cross(z, dn))
        angle = wp.acos(cos_a)
        q = wp.quat_from_axis_angle(axis, angle)
    out_xforms[cid] = wp.transform(center, q)


@wp.kernel
def _write_capsule_kinematic_state(
    cap_xforms: wp.array(dtype=wp.transform),
    cap_xforms_prev: wp.array(dtype=wp.transform),
    capsule_body_ids: wp.array(dtype=wp.int32),
    capsule_joint_q_starts: wp.array(dtype=wp.int32),
    capsule_joint_qd_starts: wp.array(dtype=wp.int32),
    dt: float,
    # outputs (indexed by capsule_body_ids[cid] for body buffers,
    # and by the per-cid joint_q_start / joint_qd_start for the
    # free-joint coordinate buffers)
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    """Push per-capsule (cap_xforms, prev_xforms) into the kinematic bodies.

    Phase B of Option 2: the kinematic Newton bodies created at build
    time by :func:`register_kinematic_capsule_bodies` need their
    free-joint coordinates (``joint_q`` / ``joint_qd``) **and** the
    derived body buffers (``body_q`` / ``body_qd``) driven every
    substep from the FEM-tracked capsule transforms.

    Why both:

    * ``body_q`` is read by ``model.collide`` (broadphase + narrowphase
      transform shapes via ``body_q[body_idx] · shape_transform[shape_id]``).
      The collide pass runs *before* the rigid solver, so writing
      ``body_q`` is what makes the contact pipeline see the capsules
      at this substep's poses.
    * ``joint_q`` is what ``SolverMuJoCo`` actually integrates against.
      The solver converts ``joint_q`` → MuJoCo ``qpos``, applies a
      ``1e10`` armature on every kinematic body's free joint to pin
      them, then writes ``joint_q`` back. If we didn't update
      ``joint_q``, the pin would hold each capsule at its rest pose
      forever (the joint_q the solver pinned to from the previous
      step) and the next-substep ``state_0.body_q`` would snap back to
      rest. Updating both keeps them in lockstep.
    * ``body_qd`` and ``joint_qd`` carry the linear/angular velocity
      MuJoCo uses for the contact-point relative-velocity term in
      its friction solve. Computed via finite-difference: linear from
      ``(center_new − center_prev) / dt``, angular from
      ``ω ≈ 2 · imag(q_new · q_prev⁻¹) / dt`` (small-angle quaternion
      derivative; the shortest-path branch flip avoids the 2π wrap
      artifact). Both buffers are written in identity convention for
      free joints (``[v.xyz, ω.xyz]``, linear in the top slot,
      matching :func:`wp.spatial_vector` / the rest of this example).
    """
    cid = wp.tid()
    bid = capsule_body_ids[cid]
    new_xf = cap_xforms[cid]
    old_xf = cap_xforms_prev[cid]

    new_pos = wp.transform_get_translation(new_xf)
    new_q = wp.transform_get_rotation(new_xf)
    old_pos = wp.transform_get_translation(old_xf)
    old_q = wp.transform_get_rotation(old_xf)

    body_q[bid] = new_xf

    # Free joint joint_q layout: (px, py, pz, qx, qy, qz, qw) per
    # ``_apply_kin_target``'s stem write pattern and Newton's
    # ``convert_warp_coords_to_mj_kernel`` mapping.
    qs = capsule_joint_q_starts[cid]
    joint_q[qs + 0] = new_pos[0]
    joint_q[qs + 1] = new_pos[1]
    joint_q[qs + 2] = new_pos[2]
    joint_q[qs + 3] = new_q[0]
    joint_q[qs + 4] = new_q[1]
    joint_q[qs + 5] = new_q[2]
    joint_q[qs + 6] = new_q[3]

    inv_dt = wp.float32(1.0) / dt
    v_lin = (new_pos - old_pos) * inv_dt

    # Angular velocity from quat delta: q_delta = q_new · q_prev⁻¹.
    # Take shortest-path branch (negate when q_delta.w < 0) so the 2π
    # wrap doesn't flip ω sign at small angles.
    q_delta = new_q * wp.quat_inverse(old_q)
    sign = wp.float32(1.0)
    if q_delta[3] < wp.float32(0.0):
        sign = wp.float32(-1.0)
    omega = wp.vec3(q_delta[0], q_delta[1], q_delta[2]) * (
        wp.float32(2.0) * sign * inv_dt
    )

    body_qd[bid] = wp.spatial_vector(v_lin, omega)

    # Free joint joint_qd layout: (vx, vy, vz, ωx, ωy, ωz).
    qds = capsule_joint_qd_starts[cid]
    joint_qd[qds + 0] = v_lin[0]
    joint_qd[qds + 1] = v_lin[1]
    joint_qd[qds + 2] = v_lin[2]
    joint_qd[qds + 3] = omega[0]
    joint_qd[qds + 4] = omega[1]
    joint_qd[qds + 5] = omega[2]


# =============================================================================
# Class wrapper
# =============================================================================


class FingerCapsuleColliders:
    """FEM-tracked capsule colliders for a multi-finger gripper.

    Owns the per-substep machinery that turns the finger spline
    keypoints into Newton-native kinematic capsule bodies' poses. The
    bodies + capsule shapes themselves are registered by
    :func:`register_kinematic_capsule_bodies` during the build phase
    (before ``builder.finalize()``); this class is constructed after
    finalize and bound to those bodies via the ``kinematic_body_ids``
    constructor arg + :meth:`bind_kinematic_joint_offsets` call.

    Per substep the caller invokes:

    1. :meth:`update_endpoints` with the spline's keypoint-world
       buffer to refresh the device-side capsule endpoints and the
       per-capsule ``wp.transform`` buffer.
    2. :meth:`update_kinematic_state` with the simulation ``State`` to
       push those transforms (and finite-differenced ``body_qd`` /
       free-joint ``joint_qd``) into the kinematic bodies — so
       ``model.collide`` + the MuJoCo solve see the capsules at this
       substep's poses, with the right relative velocity for friction.

    Contact detection + resolution are entirely delegated to Newton's
    standard collision pipeline (the capsule shapes carry
    ``has_shape_collision=True`` and their friction is set on the
    ``ShapeConfig`` at registration time). There is no per-capsule
    force kernel in this class — that path was deleted in Phase D of
    the Option-2 refactor.

    Attributes:
        num_capsules: ``F · (K - 1)``.
        cap_radii_np: Per-capsule radii [m], shape ``(num_capsules,)``,
            float32. Host-side, set at construction time from the
            asset bake or CLI override.
        kinematic_body_ids: Newton body ids of the kinematic capsule
            bodies, one per capsule in ``cid = fi · (K-1) + s`` order.
            Empty when the build phase didn't register them (Phase 0
            legacy paths).
    """

    def __init__(
        self,
        *,
        num_fingers: int,
        num_keypoints_per_finger: int,
        cap_radius: float,
        device: wp.Device | str,
        cap_radii_np: np.ndarray | None = None,
        kinematic_body_ids: list[int] | None = None,
    ) -> None:
        if num_fingers <= 0 or num_keypoints_per_finger < 2:
            self.num_capsules = 0
            self.cap_radius = 0.0
            self._device = device
            self.kinematic_body_ids: list[int] = []
            return
        F = int(num_fingers)
        K = int(num_keypoints_per_finger)
        per_finger = K - 1
        self.num_fingers = F
        self.num_keypoints_per_finger = K
        self.num_capsules = F * per_finger
        self.cap_radius = float(cap_radius)
        self._device = device
        # Body ids of the kinematic Newton capsules registered by
        # :func:`register_kinematic_capsule_bodies`. Stored in the
        # same ``cid = fi · (K-1) + s`` order as every other
        # per-capsule buffer in this class.
        self.kinematic_body_ids: list[int] = (
            [int(b) for b in kinematic_body_ids]
            if kinematic_body_ids is not None
            else []
        )
        if self.kinematic_body_ids and len(self.kinematic_body_ids) != self.num_capsules:
            raise ValueError(
                f"kinematic_body_ids length {len(self.kinematic_body_ids)} "
                f"!= num_capsules {self.num_capsules}"
            )
        # Per-capsule keypoint endpoints (global keypoint indices into
        # the spline renderer's keypoint_world buffer).
        kp_a = np.empty(self.num_capsules, dtype=np.int32)
        kp_b = np.empty(self.num_capsules, dtype=np.int32)
        for fi in range(F):
            for s in range(per_finger):
                cid = fi * per_finger + s
                kp_a[cid] = fi * K + s
                kp_b[cid] = fi * K + s + 1
        self._cap_kp_a = wp.array(kp_a, dtype=wp.int32, device=device)
        self._cap_kp_b = wp.array(kp_b, dtype=wp.int32, device=device)
        # Per-capsule radii (public read-only view). The Newton shapes
        # have already been built with these radii at register time;
        # this is kept for diagnostics / future re-binding paths.
        if cap_radii_np is not None:
            cap_radii_arr = np.asarray(cap_radii_np, dtype=np.float32).reshape(-1)
            if cap_radii_arr.size != self.num_capsules:
                raise ValueError(
                    f"cap_radii_np size {cap_radii_arr.size} != num_capsules "
                    f"{self.num_capsules}"
                )
        else:
            cap_radii_arr = np.full(self.num_capsules, float(cap_radius), dtype=np.float32)
        self.cap_radii_np = cap_radii_arr
        # Capsule endpoints in world frame; rebuilt every substep from
        # the spline's keypoint buffer.
        self._cap_a = wp.zeros(self.num_capsules, dtype=wp.vec3, device=device)
        self._cap_b = wp.zeros(self.num_capsules, dtype=wp.vec3, device=device)
        # Per-substep capsule transforms (current + previous). The
        # finite-difference of these gives the kinematic body
        # ``v_lin`` / ``ω`` written into ``body_qd`` / ``joint_qd``.
        self._cap_xforms = wp.zeros(self.num_capsules, dtype=wp.transform, device=device)
        self._cap_xforms_prev = wp.zeros(self.num_capsules, dtype=wp.transform, device=device)
        self._has_prev = False
        # Device-side ``int32`` body-id array (``None`` when no
        # kinematic body wiring was supplied).
        if self.kinematic_body_ids:
            self._cap_body_ids_arr = wp.array(
                np.asarray(self.kinematic_body_ids, dtype=np.int32),
                dtype=wp.int32,
                device=device,
            )
        else:
            self._cap_body_ids_arr = None
        # Free-joint q/qd offsets per capsule body. Filled by
        # :meth:`bind_kinematic_joint_offsets` once the model has been
        # finalized (the offsets live in ``model.joint_q_start`` /
        # ``model.joint_qd_start``).
        self._cap_joint_q_start_arr: wp.array | None = None
        self._cap_joint_qd_start_arr: wp.array | None = None

    # ------------------------------------------------------------------
    # Per-substep updates

    def update_endpoints(self, keypoint_world: wp.array) -> None:
        """Refresh capsule endpoints + transforms from the spline keypoints.

        Snapshots the current transforms into the ``prev`` slot first
        so :meth:`update_kinematic_state` can finite-difference them
        for ``body_qd`` / ``joint_qd``, then rebuilds the current
        endpoints + transforms from ``keypoint_world``.
        """
        if self.num_capsules == 0:
            return
        wp.copy(self._cap_xforms_prev, self._cap_xforms)
        wp.launch(
            _build_capsule_endpoints_from_keypoints,
            dim=self.num_capsules,
            inputs=[keypoint_world, self._cap_kp_a, self._cap_kp_b],
            outputs=[self._cap_a, self._cap_b],
            device=self._device,
        )
        wp.launch(
            _build_capsule_transforms,
            dim=self.num_capsules,
            inputs=[self._cap_a, self._cap_b],
            outputs=[self._cap_xforms],
            device=self._device,
        )
        if not self._has_prev:
            # First call: snap ``prev`` to ``current`` so the first
            # substep's finite-difference velocity is zero (instead of
            # a spurious jump from the zero-initialised ``prev``
            # buffer).
            wp.copy(self._cap_xforms_prev, self._cap_xforms)
            self._has_prev = True

    def bind_kinematic_joint_offsets(self, model) -> None:
        """Cache each capsule body's free-joint q/qd offsets from ``model``.

        Must be called once after ``builder.finalize()``, before the
        first :meth:`update_kinematic_state`. Pulls
        ``model.joint_q_start`` / ``model.joint_qd_start`` to host,
        gathers the entries indexed by each capsule body's id (one
        free joint per body), and uploads them as the per-cid offset
        arrays the kinematic-state kernel uses.
        """
        if (
            self.num_capsules == 0
            or self._cap_body_ids_arr is None
            or not self.kinematic_body_ids
        ):
            return
        q_starts = model.joint_q_start.numpy()
        qd_starts = model.joint_qd_start.numpy()
        cap_q = np.asarray(
            [int(q_starts[bid]) for bid in self.kinematic_body_ids],
            dtype=np.int32,
        )
        cap_qd = np.asarray(
            [int(qd_starts[bid]) for bid in self.kinematic_body_ids],
            dtype=np.int32,
        )
        self._cap_joint_q_start_arr = wp.array(
            cap_q, dtype=wp.int32, device=self._device
        )
        self._cap_joint_qd_start_arr = wp.array(
            cap_qd, dtype=wp.int32, device=self._device
        )

    def update_kinematic_state(self, state, dt: float) -> None:
        """Drive the kinematic Newton bodies that own each capsule.

        Writes the free-joint coordinates (``state.joint_q`` /
        ``state.joint_qd``) **and** the derived body buffers
        (``state.body_q`` / ``state.body_qd``) for each capsule's
        kinematic body id. Call once per substep, *after*
        :meth:`update_endpoints` and *before* ``model.collide`` +
        ``rigid_solver.step`` so the contact pipeline and the MuJoCo
        solve both see this substep's FEM-tracked capsule poses with
        consistent velocities.

        No-ops when :meth:`bind_kinematic_joint_offsets` hasn't been
        called yet (Phase 0 legacy back-compat).
        """
        if (
            self.num_capsules == 0
            or self._cap_body_ids_arr is None
            or self._cap_joint_q_start_arr is None
            or dt <= 0.0
        ):
            return
        wp.launch(
            _write_capsule_kinematic_state,
            dim=self.num_capsules,
            inputs=[
                self._cap_xforms,
                self._cap_xforms_prev,
                self._cap_body_ids_arr,
                self._cap_joint_q_start_arr,
                self._cap_joint_qd_start_arr,
                float(dt),
            ],
            outputs=[
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
            ],
            device=self._device,
        )

