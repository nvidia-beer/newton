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

"""Single-USD asset loader used by the inflatable examples.

Wraps the schema emitted by
``third_party/newton-mesh-tools/newton_mesh_tools/usd_io.py``:
``UsdGeom.TetMesh`` at the default prim, with ``points``,
``tetVertexIndices``, ``surfaceFaceVertexIndices``, and an optional
``chambers`` ``customData`` dict that round-trips the
:class:`newton_mesh_tools.ChamberMap` schema.

Returned as a plain dict so the examples don't pull ``newton_mesh_tools``
as a runtime dependency.
"""

from __future__ import annotations

import numpy as np

try:
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pxr (OpenUSD) required to load bundled inflatable assets. "
        "Install 'usd-core' on x86_64 or 'usd-exchange' on aarch64."
    ) from exc


def load_asset(path: str) -> dict:
    """Load a bundled ``.usda`` inflatable asset.

    Returns:
        Dict with keys:

            * ``vertices``: ``(num_vertices, 3)`` float32.
            * ``tets``: ``(num_tets, 4)`` int32, zero-based.
            * ``surface_triangles``: ``(num_surface_tris, 3)`` int32
              (or None if the stage didn't author the surface).
            * ``chamber_map``: dict with ``num_chambers``,
              ``tet_chamber_mask`` (np.int32 array), ``chambers``
              (list of chamber entries), ``regions``, ``grid``,
              ``stiffness_scale``, ``inflation_disabled``,
              ``directions``. All optional except the first two; each
              chamber entry has at minimum ``name``, ``index``,
              ``tet_indices``. ``None`` if the asset has no chambers.
            * ``extra_surfaces``: list of child ``UsdGeom.Mesh`` prims
              attached to the TetMesh (e.g. the gripper's rigid stem).
              Each entry is ``{name, vertices, triangles, placement:
              {pos, quat}, glue_pairs: {soft_indices, rigid_indices} |
              None}``. Empty list when the asset has no child meshes.

        For multi-soft :class:`GluedAssembly` USDAs (e.g. the SDF
        gripper baked by ``build_sdf_gripper.py``), the loader detects
        the ``/<root>/softs/`` scope and dispatches through
        :func:`load_glued_assembly` +
        :func:`._merge_glued_softs.merge_glued_softs`, so callers
        receive the same flat dict regardless of source schema.
    """
    stage = Usd.Stage.Open(path)

    # Multi-soft GluedAssembly (root Xform + ``/softs/`` scope of N
    # TetMeshes + ``/rigids/`` scope): load via the assembly path and
    # flatten into the single-soft contract this loader exposes.
    root = stage.GetDefaultPrim()
    if root.IsValid() and not root.IsA(UsdGeom.TetMesh):
        softs_scope = root.GetChild("softs")
        if softs_scope.IsValid():
            tet_children = [c for c in softs_scope.GetChildren() if c.IsA(UsdGeom.TetMesh)]
            if tet_children:
                from ._merge_glued_softs import merge_glued_softs  # noqa: PLC0415

                glued = load_glued_assembly(path)
                return merge_glued_softs(glued)

    # Composite single-soft ``GluedAssembly`` stage: root Xform + /soft
    # TetMesh + /rigid Mesh + ``glue`` customData. Resolve the TetMesh
    # under the root first; fall back to the legacy single-TetMesh
    # layout below.
    prim = None
    if root.IsValid() and not root.IsA(UsdGeom.TetMesh):
        soft_child = root.GetChild("soft")
        if soft_child.IsValid() and soft_child.IsA(UsdGeom.TetMesh):
            prim = soft_child
    if prim is None:
        prim = root
        if not prim.IsValid() or not prim.IsA(UsdGeom.TetMesh):
            for p in stage.Traverse():
                if p.IsA(UsdGeom.TetMesh):
                    prim = p
                    break
    if not prim.IsValid() or not prim.IsA(UsdGeom.TetMesh):
        raise ValueError(f"no UsdGeom.TetMesh prim found in {path}")

    tm = UsdGeom.TetMesh(prim)
    verts = np.asarray(tm.GetPointsAttr().Get(), dtype=np.float32).reshape(-1, 3)
    tets = np.asarray(tm.GetTetVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 4)

    surface: np.ndarray | None = None
    surf_attr = tm.GetSurfaceFaceVertexIndicesAttr()
    if surf_attr.HasAuthoredValue():
        raw = surf_attr.Get()
        if raw is not None and len(raw) > 0:
            surface = np.asarray(raw, dtype=np.int32).reshape(-1, 3)

    # Mesh-tools-derived contact tunings: see ``write_asset_usd`` in
    # ``newton_mesh_tools/usd_io.py``. Stored under ``customData["contact_params"]``
    # with keys ``particle_radius``, ``self_contact_radius``,
    # ``min_edge_length``, ``max_edge_length``. The first two follow
    # ``particle_radius = 0.5·min_edge`` and ``self_contact_radius = 2·max_edge``;
    # consumers should prefer these over bbox-derived heuristics. Returns
    # ``None`` for legacy assets that don't carry the customData.
    contact_params: dict | None = None
    raw_contact = prim.GetCustomDataByKey("contact_params")
    if raw_contact:
        contact_params = {
            k: float(raw_contact[k])
            for k in (
                "particle_radius",
                "self_contact_radius",
                "min_edge_length",
                "max_edge_length",
            )
            if k in raw_contact
        }

    # Per-finger spline keypoint set baked by ``newton_mesh_tools.write_asset_usd``.
    # Stored as a flat ``customData["finger_skeletons"]`` dict round-trippable
    # through :func:`newton_mesh_tools.skeleton.finger_skeletons_from_custom_data`.
    # Returned as the raw dict so consumers can either decode (if they have
    # ``newton_mesh_tools`` available) or skip silently. ``None`` when the
    # asset has none.
    raw_finger_skeletons = prim.GetCustomDataByKey("finger_skeletons")
    finger_skeletons_dict: dict | None = dict(raw_finger_skeletons) if raw_finger_skeletons else None

    chamber_map: dict | None = None
    raw_chambers = prim.GetCustomDataByKey("chambers")
    if raw_chambers:
        chamber_map = _decode_chamber_map(dict(raw_chambers))

    # Child ``UsdGeom.Mesh`` prims attached under the TetMesh —
    # annotation surfaces the FEM solver ignores but consumers can
    # promote to rigid colliders (the gripper's stem cylinder is the
    # canonical case). Placement and optional glue pairs are read
    # straight off the prim's customData so no extra schema is needed.
    extras: list[dict] = []
    for child in prim.GetAllChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        child_mesh = UsdGeom.Mesh(child)
        c_pts = child_mesh.GetPointsAttr().Get()
        c_fvi = child_mesh.GetFaceVertexIndicesAttr().Get()
        c_fvc = child_mesh.GetFaceVertexCountsAttr().Get()
        if not c_pts or not c_fvi or not c_fvc:
            continue
        c_verts = np.asarray(c_pts, dtype=np.float32).reshape(-1, 3)
        c_fvc_np = np.asarray(c_fvc, dtype=np.int32)
        if c_fvc_np.size == 0 or np.any(c_fvc_np != 3):
            continue
        c_tris = np.asarray(c_fvi, dtype=np.int32).reshape(-1, 3)
        # Child prim xform ops: extract translate + orient if authored.
        pos = (0.0, 0.0, 0.0)
        quat = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)
        xform = UsdGeom.Xformable(child)
        for op in xform.GetOrderedXformOps():
            name = op.GetOpName()
            if "translate" in name:
                v = op.Get()
                if v is not None:
                    pos = (float(v[0]), float(v[1]), float(v[2]))
            elif "orient" in name:
                q = op.Get()
                if q is not None:
                    # USD Gf.Quatf: w = real, Imaginary = (x, y, z).
                    imag = q.GetImaginary()
                    quat = (
                        float(imag[0]),
                        float(imag[1]),
                        float(imag[2]),
                        float(q.GetReal()),
                    )
        glue_dict = child.GetCustomDataByKey("glue")
        glue_pairs: dict | None = None
        if glue_dict and "soft_indices" in glue_dict and "rigid_indices" in glue_dict:
            glue_pairs = {
                "soft_indices": np.asarray(list(glue_dict["soft_indices"]), dtype=np.int32),
                "rigid_indices": np.asarray(list(glue_dict["rigid_indices"]), dtype=np.int32),
            }
        extras.append(
            {
                "name": child.GetName(),
                "vertices": c_verts,
                "triangles": c_tris,
                "placement": {"pos": pos, "quat": quat},
                "glue_pairs": glue_pairs,
            }
        )

    return {
        "vertices": verts,
        "tets": tets,
        "surface_triangles": surface,
        "chamber_map": chamber_map,
        "extra_surfaces": extras,
        "contact_params": contact_params,
        "finger_skeletons": finger_skeletons_dict,
    }


def load_glued_assembly(path: str) -> dict:
    """Load a composite ``soft + rigids + glue`` assembly USDA.

    The writer lives in
    ``third_party/newton-mesh-tools/newton_mesh_tools/glued_assembly.py``.
    Stage layout::

        /<name>                       Xform   (customData: glue,
                                               placement=soft placement)
          /<name>/soft                TetMesh (+ chambers customData)
          /<name>/rigids              Scope
            /<name>/rigids/<rigid>    Mesh    (+ placement customData)

    Returns:
        Dict with keys:

            * ``soft``: ``{vertices, tets, surface_triangles, chamber_map}``.
            * ``rigids``: dict keyed by ``rigid.name``. Each value is
              ``{vertices, triangles, placement={pos, quat}}``. Order is
              preserved under the ``rigid_names`` key.
            * ``rigid_names``: ordered list of rigid names (stable USD
              child order).
            * ``glue_sets``: list of
              ``{name, rigid_name, soft_indices, rigid_indices}`` dicts.
              Index arrays are ``np.int32``.
            * ``soft_placement``: ``{pos, quat}`` of the soft body at
              the canonical placement.
    """
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise ValueError(f"{path}: no defaultPrim")

    # Legacy schema dispatch: the gripper-tools writer puts the soft body's
    # ``UsdGeom.TetMesh`` *at the root* with rigid ``UsdGeom.Mesh`` prims
    # attached as children (each carrying ``glue`` customData). That layout
    # predates the ``/softs`` + ``/rigids`` scopes used by the newer
    # :class:`GluedAssembly` writer. Detect it by the root prim type and
    # adapt :func:`load_asset`'s output to this function's unified shape so
    # callers don't need to care which writer produced the file.
    if root.IsA(UsdGeom.TetMesh):
        return _legacy_to_unified(load_asset(path), root)

    softs_scope = root.GetChild("softs")
    rigids_scope = root.GetChild("rigids")
    if not rigids_scope.IsValid():
        raise ValueError(f"{path}: expected Scope at {root.GetPath()}/rigids")

    softs: dict[str, dict] = {}
    soft_names: list[str] = []

    def _read_soft_tet_prim(prim) -> dict:
        tm = UsdGeom.TetMesh(prim)
        sv = np.asarray(tm.GetPointsAttr().Get(), dtype=np.float32).reshape(-1, 3)
        st = np.asarray(tm.GetTetVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 4)
        sa = tm.GetSurfaceFaceVertexIndicesAttr()
        ss: np.ndarray | None = None
        if sa.HasAuthoredValue():
            raw = sa.Get()
            if raw is not None and len(raw) > 0:
                ss = np.asarray(raw, dtype=np.int32).reshape(-1, 3)
        raw_c = prim.GetCustomDataByKey("chambers")
        cmap = _decode_chamber_map(dict(raw_c)) if raw_c else None
        placement = prim.GetCustomDataByKey("placement") or {}
        # Per-soft customData baked by build_sdf_gripper.py:
        #   ``finger_skeletons`` (flat skeleton dict per soft)
        #   ``contact_params``  (particle_radius / self_contact_radius / ...)
        # Both surface as raw dicts so downstream consumers can decode
        # them (e.g. :func:`._merge_glued_softs.merge_glued_softs`).
        raw_fs = prim.GetCustomDataByKey("finger_skeletons")
        finger_skeletons = dict(raw_fs) if raw_fs else None
        raw_cp = prim.GetCustomDataByKey("contact_params")
        contact_params: dict | None = None
        if raw_cp:
            contact_params = {
                k: float(raw_cp[k])
                for k in (
                    "particle_radius",
                    "self_contact_radius",
                    "min_edge_length",
                    "max_edge_length",
                )
                if k in raw_cp
            }
        # Per-cage-vertex SDF data baked by build_sdf_gripper.py
        # (``customData["sdf"]``). Each cage vertex carries:
        #   * ``vertex_distance`` — signed distance to the rigid
        #     finger surface in **canonical** (rest) frame [m],
        #     mujoco-warp sign convention (negative inside).
        #   * ``vertex_uvw`` — vertex position in the canonical AABB
        #     normalised to ``[0, 1]^3`` (rotation-invariant
        #     across the 4 instances).
        # The FFD-inverse contact kernel consumes ``vertex_distance``
        # (interpolated bary-style over the deformed cage tet) to
        # sample the morphed SDF directly — no static texture lookup
        # is needed. ``rigid_aabb_min`` / ``rigid_aabb_max`` give the
        # AABB of the canonical rigid surface (used to denormalise
        # uvw → canonical coords if the kernel needs them).
        raw_sdf = prim.GetCustomDataByKey("sdf")
        sdf_bake: dict | None = None
        if raw_sdf:
            vd = list(raw_sdf.get("vertex_distance") or [])
            vu = list(raw_sdf.get("vertex_uvw") or [])
            if vd:
                sdf_bake = {
                    "vertex_distance": np.asarray(vd, dtype=np.float32),
                    "vertex_uvw": np.asarray(vu, dtype=np.float32).reshape(-1, 3) if vu else None,
                    "rigid_aabb_min": np.asarray(
                        list(raw_sdf.get("rigid_aabb_min") or (0.0, 0.0, 0.0)),
                        dtype=np.float32,
                    ),
                    "rigid_aabb_max": np.asarray(
                        list(raw_sdf.get("rigid_aabb_max") or (0.0, 0.0, 0.0)),
                        dtype=np.float32,
                    ),
                    "rigid_mesh": str(raw_sdf.get("rigid_mesh") or ""),
                }
        return {
            "vertices": sv,
            "tets": st,
            "surface_triangles": ss,
            "chamber_map": cmap,
            "placement": {
                "pos": [float(v) for v in (placement.get("pos") or (0.0, 0.0, 0.0))],
                "quat": [float(v) for v in (placement.get("quat") or (0.0, 0.0, 0.0, 1.0))],
            },
            "finger_skeletons": finger_skeletons,
            "contact_params": contact_params,
            "sdf_bake": sdf_bake,
        }

    if softs_scope.IsValid():
        for child in softs_scope.GetChildren():
            if not child.IsA(UsdGeom.TetMesh):
                continue
            info = _read_soft_tet_prim(child)
            name = str((child.GetCustomDataByKey("placement") or {}).get("name") or child.GetName())
            soft_names.append(name)
            softs[name] = info
    else:
        # Back-compat: legacy single-soft layout at ``/<root>/soft``.
        legacy_soft = root.GetChild("soft")
        if legacy_soft.IsValid() and legacy_soft.IsA(UsdGeom.TetMesh):
            info = _read_soft_tet_prim(legacy_soft)
            # Legacy placement lived on the root, not the prim.
            legacy_pl = root.GetCustomDataByKey("placement") or {}
            info["placement"] = {
                "pos": [float(v) for v in (legacy_pl.get("soft_pos") or (0.0, 0.0, 0.0))],
                "quat": [float(v) for v in (legacy_pl.get("soft_quat") or (0.0, 0.0, 0.0, 1.0))],
            }
            soft_names.append("soft")
            softs["soft"] = info
    if not softs:
        raise ValueError(f"{path}: no SoftBody under {root.GetPath()}/softs")

    # Rigid surfaces.
    rigids: dict[str, dict] = {}
    rigid_names: list[str] = []
    for child in rigids_scope.GetChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        rm = UsdGeom.Mesh(child)
        # ``collision_only`` (mesh-tools authoring convention): the
        # rigid is a sibling collision proxy (``<name>_collision_hull``)
        # authored with ``visibility=invisible``. We still load it — the
        # example branches on the flag to wire it as the collider while
        # the matching visible mesh stays render-only.
        collision_only_flag = bool(child.GetCustomDataByKey("collision_only"))
        imageable = UsdGeom.Imageable(child)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            collision_only_flag = True
        r_verts = np.asarray(rm.GetPointsAttr().Get(), dtype=np.float32).reshape(-1, 3)
        fvc = np.asarray(rm.GetFaceVertexCountsAttr().Get() or [], dtype=np.int32)
        fvi = np.asarray(rm.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32)
        if fvc.size == 0 or np.any(fvc != 3):
            raise ValueError(f"{path}: rigid mesh at {child.GetPath()} must be pure-triangle")
        r_tris = fvi.reshape(-1, 3)
        placement = child.GetCustomDataByKey("placement") or {}
        name = str(placement.get("name") or child.GetName())
        rigid_names.append(name)
        # SDF gripper variant: per-finger ``finger_<i>_skin`` rigids
        # carry a ``skin_binding`` customData (tet index + bary4 per
        # skin vert + the name of the soft body to deform against),
        # baked by ``build_sdf_gripper.py``. Surface it here so the
        # example can wire FFD-style deformation at runtime.
        raw_sb = child.GetCustomDataByKey("skin_binding")
        skin_binding: dict | None = None
        if raw_sb:
            skin_binding = {
                "soft_name": str(raw_sb["soft_name"]),
                "tet_indices": np.asarray(list(raw_sb["tet_indices"]), dtype=np.int32),
                "bary_weights": np.asarray(list(raw_sb["bary_weights"]), dtype=np.float32).reshape(-1, 4),
            }
        # ``displayColor`` (UsdGeomMesh primvar) — same convention the
        # SO-101 link meshes use. The example forwards this to its
        # per-frame ``viewer.log_mesh`` re-log so the asset stays the
        # single source of truth for visual color.
        display_color: tuple[float, float, float] | None = None
        dc_attr = rm.GetDisplayColorAttr()
        if dc_attr.HasAuthoredValue():
            dc_val = dc_attr.Get()
            if dc_val is not None and len(dc_val) > 0:
                c0 = dc_val[0]
                display_color = (float(c0[0]), float(c0[1]), float(c0[2]))
        rigids[name] = {
            "vertices": r_verts,
            "triangles": r_tris,
            "placement": {
                "pos": [float(v) for v in (placement.get("pos") or (0.0, 0.0, 0.0))],
                "quat": [float(v) for v in (placement.get("quat") or (0.0, 0.0, 0.0, 1.0))],
            },
            "skin_binding": skin_binding,
            "display_color": display_color,
            "collision_only": collision_only_flag,
        }

    # Glue sets — each references a specific (soft_name, rigid_name) pair.
    glue_sets: list[dict] = []
    raw_glue = root.GetCustomDataByKey("glue")
    if raw_glue:
        entries = dict(raw_glue.get("entries") or {})
        default_soft = soft_names[0] if soft_names else "soft"
        default_rigid = rigid_names[0] if rigid_names else "rigid"
        for key in sorted(entries.keys()):
            e = dict(entries[key])
            soft_name = str(e.get("soft_name") or default_soft)
            # Back-compat: older sets used integer ``soft_instance``.
            if "soft_name" not in e and "soft_instance" in e and soft_names:
                idx = int(e.get("soft_instance") or 0)
                if 0 <= idx < len(soft_names):
                    soft_name = soft_names[idx]
            glue_sets.append(
                {
                    "name": str(e.get("name", key)),
                    "soft_name": soft_name,
                    "rigid_name": str(e.get("rigid_name") or default_rigid),
                    "soft_indices": np.asarray(list(e.get("soft_indices") or []), dtype=np.int32),
                    "rigid_indices": np.asarray(list(e.get("rigid_indices") or []), dtype=np.int32),
                }
            )

    return {
        "softs": softs,
        "soft_names": soft_names,
        "rigids": rigids,
        "rigid_names": rigid_names,
        "glue_sets": glue_sets,
    }


def _legacy_to_unified(legacy: dict, root) -> dict:
    """Adapt :func:`load_asset`'s legacy output to the unified assembly shape.

    The legacy schema stores the soft body at the root TetMesh and rigids
    as child Mesh prims (``extras``). Each glued extra carries the
    ``soft_indices`` / ``rigid_indices`` index arrays in its own customData.
    This adapter packs that into the same dict shape that the new
    :class:`GluedAssembly` schema produces, so the example doesn't need a
    schema branch.

    Soft placement is taken from the root prim's xformOps when authored
    (legacy assets typically left it at identity, with the rigid carrying
    the offset). Extras without ``glue_pairs`` are ignored — they're
    visual-only and would have nothing for the kinematic glue to bind.
    """
    soft_pos = (0.0, 0.0, 0.0)
    soft_quat = (0.0, 0.0, 0.0, 1.0)
    xform = UsdGeom.Xformable(root)
    for op in xform.GetOrderedXformOps():
        name = op.GetOpName()
        if "translate" in name:
            v = op.Get()
            if v is not None:
                soft_pos = (float(v[0]), float(v[1]), float(v[2]))
        elif "orient" in name:
            q = op.Get()
            if q is not None:
                imag = q.GetImaginary()
                soft_quat = (
                    float(imag[0]),
                    float(imag[1]),
                    float(imag[2]),
                    float(q.GetReal()),
                )

    softs = {
        "soft": {
            "vertices": legacy["vertices"],
            "tets": legacy["tets"],
            "surface_triangles": legacy["surface_triangles"],
            "chamber_map": legacy.get("chamber_map"),
            "placement": {
                "pos": [soft_pos[0], soft_pos[1], soft_pos[2]],
                "quat": [soft_quat[0], soft_quat[1], soft_quat[2], soft_quat[3]],
            },
        }
    }

    rigids: dict[str, dict] = {}
    rigid_names: list[str] = []
    glue_sets: list[dict] = []
    for extra in legacy.get("extra_surfaces", []):
        glue_pairs = extra.get("glue_pairs")
        if glue_pairs is None:
            continue
        name = str(extra["name"])
        rigids[name] = {
            "vertices": extra["vertices"],
            "triangles": extra["triangles"],
            "placement": extra["placement"],
        }
        rigid_names.append(name)
        glue_sets.append(
            {
                "name": f"glue_{name}",
                "soft_name": "soft",
                "rigid_name": name,
                "soft_indices": np.asarray(glue_pairs["soft_indices"], dtype=np.int32),
                "rigid_indices": np.asarray(glue_pairs["rigid_indices"], dtype=np.int32),
            }
        )

    return {
        "softs": softs,
        "soft_names": ["soft"],
        "rigids": rigids,
        "rigid_names": rigid_names,
        "glue_sets": glue_sets,
    }


def load_surface_mesh(path: str) -> dict:
    """Load a plain triangulated surface Mesh USDA.

    Companion to :func:`load_asset` / :func:`load_glued_assembly` for
    standalone rigid colliders baked via
    ``newton_mesh_tools.write_surface_mesh_usd`` (e.g. the table's
    plate). Returns ``{"vertices", "triangles"}`` with ``np.float32`` /
    ``np.int32`` arrays.
    """
    stage = Usd.Stage.Open(path)
    prim = stage.GetDefaultPrim()
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Mesh):
                prim = p
                break
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"no UsdGeom.Mesh prim found in {path}")
    mesh = UsdGeom.Mesh(prim)
    verts = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float32).reshape(-1, 3)
    fvc = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int32)
    fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32)
    if fvc.size == 0 or np.any(fvc != 3):
        raise ValueError(f"{path}: surface mesh must be pure-triangle")
    tris = fvi.reshape(-1, 3)
    return {"vertices": verts, "triangles": tris}


# ---------------------------------------------------------------------------
# Hex-mesh assembly loader
# ---------------------------------------------------------------------------


def load_hex_assembly(path: str) -> list[dict]:
    """Load a Q1 hex FEM assembly USDA.

    Args:
        path: Path to the ``.usda`` file.

    Returns:
        List of dicts (in document order), each with:

            * ``name`` (str)
            * ``vertices`` — ``(V, 3)`` float32 world-space positions.
            * ``hex_indices`` — ``(H, 8)`` int32 element connectivity
              (local, 0-based).
            * ``surface_triangles`` — ``(T, 3)`` int32 boundary
              triangulation (local).
            * ``chamber_id`` (int). ``-1`` when no chamber is assigned
              to the object.
    """
    stage = Usd.Stage.Open(path)

    objects: list[dict] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        hex_data = prim.GetCustomDataByKey("hex_mesh")
        if not hex_data:
            continue

        usd_mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(usd_mesh.GetPointsAttr().Get() or [], dtype=np.float32).reshape(-1, 3)
        fvi = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32)
        fvc = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
        if fvc.size == 0 or np.any(fvc != 3):
            raise ValueError(f"Surface mesh on '{prim.GetPath()}' is not triangulated")
        S = fvi.reshape(-1, 3)

        flat = np.asarray(list(hex_data["hex_vertex_indices"]), dtype=np.int32)
        hex_conn = flat.reshape(-1, 8)

        cid = prim.GetCustomDataByKey("chamber_id")
        objects.append(
            {
                "name": prim.GetName(),
                "vertices": verts,
                "hex_indices": hex_conn,
                "surface_triangles": S,
                "chamber_id": int(cid) if cid is not None else -1,
            }
        )
    return objects


def detect_asset_type(path: str) -> str:
    """Detect the FEM element type of a USDA asset without fully loading it.

    Scans the file text for schema keywords and returns one of
    ``"tet"``, ``"hex"``, or ``"glue"``.

    Args:
        path: Path to the ``.usda`` file.

    Returns:
        ``"tet"`` — file contains ``tetVertexIndices``.
        ``"hex"`` — file contains ``hex_vertex_indices`` inside ``hex_mesh``.
        ``"glue"`` — file contains ``glue_sets`` (multi-body glued assembly).

    Raises:
        ValueError: if the type cannot be determined.
    """
    with open(path) as f:
        text = f.read()

    if "hex_vertex_indices" in text:
        return "hex"
    # Glue assembly: has both a rigid scope and a soft/tet scope.
    # Detected by the "glue" customData dict key AND rigid geometry.
    if "dictionary glue" in text and "rigid" in text:
        return "glue"
    if "tetVertexIndices" in text:
        return "tet"

    raise ValueError(
        f"Cannot determine asset type from {path}. Expected 'tetVertexIndices', 'hex_vertex_indices', or 'glue_sets'."
    )


def _decode_chamber_map(data: dict) -> dict:
    """Decode the ``chambers`` customData dict into a friendly shape."""
    entries_src = data.get("entries", {}) or {}
    chambers: list[dict] = []
    for key in sorted(entries_src.keys()):
        e = dict(entries_src[key])
        entry: dict = {
            "name": str(e["name"]),
            "index": int(e["index"]),
            "tet_indices": np.asarray(e["tet_indices"], dtype=np.int32),
        }
        if "grid" in e:
            entry["grid"] = tuple(int(v) for v in e["grid"])
        if "group" in e:
            entry["group"] = {str(k): int(v) for k, v in dict(e["group"]).items()}
        chambers.append(entry)

    regions_src = data.get("regions", {}) or {}
    regions: list[dict] = []
    for key in sorted(regions_src.keys()):
        e = dict(regions_src[key])
        entry = {
            "name": str(e["name"]),
            "tet_indices": np.asarray(e["tet_indices"], dtype=np.int32),
        }
        if "group" in e:
            entry["group"] = {str(k): int(v) for k, v in dict(e["group"]).items()}
        regions.append(entry)

    out: dict = {
        "num_chambers": int(data["num_chambers"]),
        "tet_chamber_mask": np.asarray(data["tet_chamber_mask"], dtype=np.int32),
        "chambers": chambers,
        "regions": regions,
    }
    if "grid" in data:
        out["grid"] = tuple(int(v) for v in data["grid"])
    if "stiffness_scale" in data:
        out["stiffness_scale"] = np.asarray(data["stiffness_scale"], dtype=np.float32)
    if "inflation_disabled" in data:
        out["inflation_disabled"] = sorted(int(v) for v in data["inflation_disabled"])
    if "directions" in data:
        flat = np.asarray(data["directions"], dtype=np.float32)
        cols = int(data.get("directions_cols", 3))
        out["directions"] = flat.reshape(-1, cols)
    if "torque_directions" in data:
        flat = np.asarray(data["torque_directions"], dtype=np.float32)
        cols = int(data.get("torque_directions_cols", 3))
        out["torque_directions"] = flat.reshape(-1, cols)
    if "spring_indices" in data:
        flat = np.asarray(data["spring_indices"], dtype=np.int32)
        out["spring_indices"] = flat.reshape(-1, 2)
    if "spring_chamber_mask" in data:
        out["spring_chamber_mask"] = np.asarray(data["spring_chamber_mask"], dtype=np.int32)
    if "spring_rest_direction" in data:
        flat = np.asarray(data["spring_rest_direction"], dtype=np.float32)
        out["spring_rest_direction"] = flat.reshape(-1, 3)
    if "stiff_axis" in data:
        out["stiff_axis"] = np.asarray(data["stiff_axis"], dtype=np.float32).reshape(3)
    return out
