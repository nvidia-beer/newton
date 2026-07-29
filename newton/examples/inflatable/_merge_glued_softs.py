# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Merge a multi-soft :class:`GluedAssembly` into a single-soft asset dict.

The multi-finger ``sdf_gripper.usda`` is authored as a
:class:`GluedAssembly` — one TetMesh per finger under ``/<root>/softs/``,
one hub Mesh under ``/<root>/rigids/``, N glue sets at root level. This
module concatenates the N soft TetMeshes into one merged vertex/tet
array (with index offsets, per-finger chamber labels, and the hub's
glue pairs aggregated into a single soft → hub pair list) so the rest
of the inflatable example pipeline can consume it via the same
single-soft contract as ``gripper_4f.usda``.
"""

from __future__ import annotations

import numpy as np


def merge_glued_softs(glued: dict) -> dict:
    """Flatten a :func:`._usd_asset.load_glued_assembly` result.

    Args:
        glued: Dict returned by
            :func:`._usd_asset.load_glued_assembly`. Must contain at
            least one soft body and one rigid surface, plus matching
            glue sets.

    Returns:
        Dict shaped exactly like :func:`._usd_asset.load_asset`'s output:
        ``vertices`` / ``tets`` / ``surface_triangles`` / ``chamber_map``
        / ``finger_skeletons`` / ``extra_surfaces`` / ``contact_params``.
        ``extra_surfaces`` contains one entry per rigid; the rigid named
        ``hub`` (when present) carries the aggregated soft → hub glue
        pairs as ``glue_pairs.soft_indices`` / ``rigid_indices``.

    Raises:
        ValueError: If the assembly has no softs / rigids, or if a soft
            body is missing its chamber map (chambers are required for
            the example's pressure / spring / torque pipeline).
    """
    soft_names: list[str] = list(glued.get("soft_names") or [])
    softs: dict[str, dict] = dict(glued.get("softs") or {})
    rigids: dict[str, dict] = dict(glued.get("rigids") or {})
    glue_sets: list[dict] = list(glued.get("glue_sets") or [])
    if not soft_names or not softs:
        raise ValueError("merge_glued_softs: assembly has no softs")
    if not rigids:
        raise ValueError("merge_glued_softs: assembly has no rigids")

    # Soft block: concatenate verts / tets / surface_triangles. Verts
    # are already in world coords (the writer bakes the per-instance
    # transform into the vertex positions, with identity placement), so
    # no rotation is applied here. Tets / surface_triangles get a
    # per-soft vertex-index offset.
    verts_blocks: list[np.ndarray] = []
    tets_blocks: list[np.ndarray] = []
    surf_blocks: list[np.ndarray] = []
    # Per-soft surface triangles **with soft-local vertex indices** (i.e.
    # NOT offset into the merged ``vertices`` array). Callers that build a
    # per-finger :class:`wp.Mesh` need 0-based local indices so they can be
    # reused unchanged across finger instances.
    surf_per_soft: list[np.ndarray] = []
    v_offsets: dict[str, int] = {}
    t_offsets: dict[str, int] = {}
    chamber_offsets: dict[str, int] = {}
    chambers_out: list[dict] = []
    regions_out: list[dict] = []
    tet_chamber_mask_blocks: list[np.ndarray] = []
    spring_idx_blocks: list[np.ndarray] = []
    spring_cham_blocks: list[np.ndarray] = []
    spring_rest_blocks: list[np.ndarray] = []
    torque_dir_blocks: list[np.ndarray] = []
    finger_id_out: list[int] = []
    kp_index_out: list[int] = []
    skel_tet_idx_out: list[np.ndarray] = []
    skel_bary_out: list[np.ndarray] = []
    skel_rest_out: list[np.ndarray] = []
    skel_arc_out: list[np.ndarray] = []
    skel_radii_out: list[np.ndarray] = []
    skel_ax_start_out: list[np.ndarray] = []
    skel_ax_end_out: list[np.ndarray] = []
    contact_params_first: dict | None = None
    sdf_bake_first: dict | None = None
    total_v = 0
    total_t = 0
    total_chambers = 0
    for soft_idx, name in enumerate(soft_names):
        s = softs[name]
        cmap = s.get("chamber_map")
        if cmap is None:
            raise ValueError(
                f"merge_glued_softs: soft {name!r} has no chamber data; the "
                "example's pressure / spring pipeline requires it. "
                "Re-bake the asset via build_sdf_gripper.py."
            )
        V = np.asarray(s["vertices"], dtype=np.float32)
        T = np.asarray(s["tets"], dtype=np.int32)
        S = s.get("surface_triangles")
        if S is not None:
            S = np.asarray(S, dtype=np.int32)
        v_offsets[name] = total_v
        t_offsets[name] = total_t
        chamber_offsets[name] = total_chambers
        verts_blocks.append(V)
        tets_blocks.append((T + total_v).astype(np.int32))
        if S is not None:
            surf_blocks.append((S + total_v).astype(np.int32))
            # Same triangles, but with the local 0-based vertex indices
            # the live :class:`wp.Mesh` consumer needs (see
            # ``surf_per_soft`` declaration above).
            surf_per_soft.append(S.astype(np.int32).copy())
        else:
            surf_per_soft.append(np.zeros((0, 3), dtype=np.int32))

        # Chambers — shift indices + tet ids, label group["finger"] with
        # this soft's position in soft_names.
        for c in cmap["chambers"]:
            grp = dict(c.get("group", {}) or {})
            # OVERRIDE any per-soft finger label. The bake script
            # writes ``group["finger"]=0`` inside each soft so
            # ``compute_spring_bend_data`` indexes its single-row
            # ``torque_directions`` correctly; at merge time we
            # re-label with the global ring index ``soft_idx`` so
            # downstream code (per-finger pressure sliders, etc.)
            # sees N distinct fingers.
            grp["finger"] = soft_idx
            chambers_out.append(
                {
                    "name": f"{name}_{c['name']}",
                    "index": int(c["index"]) + total_chambers,
                    "tet_indices": (np.asarray(c["tet_indices"], dtype=np.int32) + total_t).astype(np.int32),
                    "group": grp,
                }
            )
        for r in cmap.get("regions", []) or []:
            grp = dict(r.get("group", {}) or {})
            grp["finger"] = soft_idx
            regions_out.append(
                {
                    "name": f"{name}_{r['name']}",
                    "tet_indices": (np.asarray(r["tet_indices"], dtype=np.int32) + total_t).astype(np.int32),
                    "group": grp,
                }
            )
        mask = np.asarray(cmap["tet_chamber_mask"], dtype=np.int32).copy()
        valid = mask >= 0
        mask[valid] = mask[valid] + total_chambers
        tet_chamber_mask_blocks.append(mask)

        # Spring data (optional). Offset spring vertex indices + chamber
        # mask just like the chamber bins.
        s_idx = cmap.get("spring_indices")
        s_cham = cmap.get("spring_chamber_mask")
        s_rest = cmap.get("spring_rest_direction")
        if s_idx is not None and s_cham is not None and s_rest is not None:
            spring_idx_blocks.append((np.asarray(s_idx, dtype=np.int32).reshape(-1, 2) + total_v).astype(np.int32))
            sc = np.asarray(s_cham, dtype=np.int32).copy()
            valid = sc >= 0
            sc[valid] = sc[valid] + total_chambers
            spring_cham_blocks.append(sc)
            spring_rest_blocks.append(np.asarray(s_rest, dtype=np.float32).reshape(-1, 3))

        # Torque directions per soft (one row per group): stacked as-is.
        td = cmap.get("torque_directions")
        if td is not None:
            torque_dir_blocks.append(np.asarray(td, dtype=np.float32).reshape(-1, 3))

        # Skeleton — flat per-soft single-finger set. Shift tet_indices
        # by total_t; rest_positions / axis_start / axis_end are already
        # in world coords (baked by build_sdf_gripper.py per instance).
        skel = s.get("finger_skeletons")
        if skel is not None:
            K = int(skel["num_keypoints_per_finger"])
            tet_idx_arr = np.asarray(list(skel["tet_indices"]), dtype=np.int32).reshape(-1)
            bary_arr = np.asarray(list(skel["bary_weights"]), dtype=np.float32).reshape(-1, 4)
            rest_arr = np.asarray(list(skel["rest_positions"]), dtype=np.float32).reshape(-1, 3)
            arc_arr = np.asarray(list(skel["arc_lengths"]), dtype=np.float32).reshape(-1)
            radii_arr = np.asarray(list(skel["radii"]), dtype=np.float32).reshape(-1)
            ax_s_arr = np.asarray(list(skel["axis_start"]), dtype=np.float32).reshape(-1, 3)
            ax_e_arr = np.asarray(list(skel["axis_end"]), dtype=np.float32).reshape(-1, 3)
            n_local_fingers = int(skel.get("num_fingers", 1))
            for local_finger in range(n_local_fingers):
                finger_id_out.extend([soft_idx] * K)
                kp_index_out.extend(range(K))
                k_lo = local_finger * K
                k_hi = (local_finger + 1) * K
                skel_tet_idx_out.append((tet_idx_arr[k_lo:k_hi] + total_t).astype(np.int32))
                skel_bary_out.append(bary_arr[k_lo:k_hi])
                skel_rest_out.append(rest_arr[k_lo:k_hi])
                skel_arc_out.append(arc_arr[k_lo:k_hi])
                skel_radii_out.append(radii_arr[k_lo:k_hi])
                skel_ax_start_out.append(ax_s_arr[local_finger : local_finger + 1])
                skel_ax_end_out.append(ax_e_arr[local_finger : local_finger + 1])

        if contact_params_first is None and s.get("contact_params"):
            contact_params_first = dict(s["contact_params"])

        # Baked per-cage-vertex SDF (``vertex_distance`` + ``vertex_uvw``)
        # — same canonical data on every soft (rotation-invariant uvw),
        # so we capture the first soft's bake as the asset-wide
        # canonical ``sdf_bake``. The FFD-inverse contact kernel maps
        # finger i's global particles [i·V, (i+1)·V) back onto this
        # canonical array via local-vertex index ``k`` so a single
        # vertex_distance(V,) buffer drives all four fingers.
        if sdf_bake_first is None and s.get("sdf_bake"):
            sdf_bake_first = dict(s["sdf_bake"])

        total_v += int(V.shape[0])
        total_t += int(T.shape[0])
        total_chambers += int(cmap["num_chambers"])

    vertices = np.concatenate(verts_blocks, axis=0)
    tets = np.concatenate(tets_blocks, axis=0)
    surface_triangles = np.concatenate(surf_blocks, axis=0) if surf_blocks else None

    cmap_out: dict = {
        "num_chambers": int(total_chambers),
        "tet_chamber_mask": np.concatenate(tet_chamber_mask_blocks, axis=0),
        "chambers": chambers_out,
        "regions": regions_out,
    }
    if spring_idx_blocks:
        cmap_out["spring_indices"] = np.concatenate(spring_idx_blocks, axis=0)
        cmap_out["spring_chamber_mask"] = np.concatenate(spring_cham_blocks, axis=0)
        cmap_out["spring_rest_direction"] = np.concatenate(spring_rest_blocks, axis=0)
    if torque_dir_blocks:
        cmap_out["torque_directions"] = np.concatenate(torque_dir_blocks, axis=0)

    finger_skeletons_out: dict | None = None
    if skel_tet_idx_out:
        finger_skeletons_out = {
            "num_fingers": len(soft_names),
            "num_keypoints_per_finger": int(skel_tet_idx_out[0].shape[0]),
            "finger_id": np.asarray(finger_id_out, dtype=np.int32),
            "keypoint_index": np.asarray(kp_index_out, dtype=np.int32),
            "tet_indices": np.concatenate(skel_tet_idx_out, axis=0),
            "bary_weights": np.concatenate(skel_bary_out, axis=0).reshape(-1),
            "rest_positions": np.concatenate(skel_rest_out, axis=0).reshape(-1),
            "arc_lengths": np.concatenate(skel_arc_out, axis=0),
            "radii": np.concatenate(skel_radii_out, axis=0),
            "axis_start": np.concatenate(skel_ax_start_out, axis=0).reshape(-1),
            "axis_end": np.concatenate(skel_ax_end_out, axis=0).reshape(-1),
        }

    # Rigids → extras. Each rigid becomes one entry; glue sets targeting
    # it are aggregated into a single (soft_indices, rigid_indices) pair
    # list with soft indices remapped through ``v_offsets``.
    extras_out: list[dict] = []
    rigid_order = list(glued.get("rigid_names") or list(rigids.keys()))
    for r_name in rigid_order:
        r = rigids[r_name]
        soft_idx_concat: list[np.ndarray] = []
        rigid_idx_concat: list[np.ndarray] = []
        for gs in glue_sets:
            if gs["rigid_name"] != r_name:
                continue
            s_name = gs["soft_name"]
            if s_name not in v_offsets:
                continue
            offset = v_offsets[s_name]
            soft_idx_concat.append((np.asarray(gs["soft_indices"], dtype=np.int32) + offset).astype(np.int32))
            rigid_idx_concat.append(np.asarray(gs["rigid_indices"], dtype=np.int32))
        glue_pairs: dict | None = None
        if soft_idx_concat:
            glue_pairs = {
                "soft_indices": np.concatenate(soft_idx_concat, axis=0),
                "rigid_indices": np.concatenate(rigid_idx_concat, axis=0),
            }
        placement = dict(r["placement"])
        r_verts = np.asarray(r["vertices"], dtype=np.float32)

        # SDF gripper variant: surface ``skin_binding`` (per-skin-vert
        # tet idx + bary4 + soft_name) so the example can deform
        # ``finger_<i>_skin`` meshes from the corresponding FEM cage
        # particle positions via FFD. ``None`` for rigids that aren't
        # finger skins.
        extras_out.append(
            {
                "name": r_name,
                "vertices": r_verts,
                "triangles": r["triangles"],
                "placement": placement,
                "glue_pairs": glue_pairs,
                "skin_binding": r.get("skin_binding"),
                "display_color": r.get("display_color"),
                "collision_only": bool(r.get("collision_only", False)),
            }
        )

    # pin_indices: derived from the glue sets — the soft_indices of every
    # GlueSet are the Dirichlet-pin vertices by construction.  Both "bottom"
    # and "top" keys point to the same set so the asset works regardless of
    # soft_position.
    pin_indices_out: dict | None = None
    if glue_sets:
        merged: list[np.ndarray] = []
        for gs in glue_sets:
            sname = str(gs.get("soft_name") or soft_names[0])
            off = v_offsets.get(sname, 0)
            si = np.asarray(gs["soft_indices"], dtype=np.int32)
            merged.append(si + off if off else si)
        all_pins = np.unique(np.concatenate(merged)).astype(np.int32)
        pin_indices_out = {"bottom": all_pins, "top": all_pins}

    return {
        "vertices": vertices,
        "tets": tets,
        "surface_triangles": surface_triangles,
        "surface_triangles_per_soft": surf_per_soft,
        "chamber_map": cmap_out,
        "extra_surfaces": extras_out,
        "contact_params": contact_params_first,
        "finger_skeletons": finger_skeletons_out,
        "sdf_bake": sdf_bake_first,
        "pin_indices": pin_indices_out,
    }
