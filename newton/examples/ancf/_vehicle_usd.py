# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Vehicle from a USD asset for the ANCF tire examples (``--vehicle-asset``).

Every vehicle is a USD file baked by newton-tire-tool (``newton_tire_tool.vehicle``; schema
in its module docstring). Its default prim carries, as ``customData``, everything that
distinguishes one vehicle from another: which bodies are the spindles, which joints are
the axles and the steering, the steering kind, speed and steer limits, hub positions,
camera and the CTIS panel envelope. Two kinds exist:

* ``articulated`` — a full rigid multibody (double wishbone, Pitman arm, tendons, loop
  constraints ...) authored with the MuJoCo USD schema; loaded with
  :meth:`newton.ModelBuilder.add_usd`.
* ``rigid_hull`` — one hull body plus fixed axles: ``/Vehicle/Hull`` (``PhysicsMassAPI`` and
  ``Cube`` / ``Cylinder`` colliders carry mass and collision, ``/Vehicle/Hull/Visual`` is the
  display mesh) and the ``/Vehicle/Wheels`` hub frames; the bodies and hinges are built here.

Nothing in this module knows which vehicle it is loading.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

# ANCF tire frame (Y-up, axle X) -> Z-up body frame (axle Y): (x, y, z) -> (z, x, y).
_P_YUP_TO_ZU = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# Box collider tiles: footprint side [m] that keeps one box under MuJoCo's 50-prism hfield cap
# on the examples' 0.5 m collider cell at any yaw (see VehicleUSD._add_usd_shape).
_HULL_BOX_MAX_SIDE = 1.6


@dataclass(frozen=True)
class VehicleSpec:
    """Dimensions and limits the examples read (``self.spec``), all from the two assets."""

    tire_asset: str
    tire_R_outer: float
    tire_R_inner: float
    tire_width: float
    half_wheelbase: float
    half_track: float
    max_steer: float
    max_wheel_speed: float
    rigid_corner_mass: float
    camera_pos: tuple[float, float, float]
    camera_pitch: float
    camera_yaw: float


def find_joint(builder: newton.ModelBuilder, joint_name: str) -> int:
    """Index of the joint whose label ends in ``joint_name``."""
    for j, label in enumerate(builder.joint_label):
        if label.split("/")[-1] == joint_name:
            return j
    raise KeyError(f"joint '{joint_name}' not found")


def find_dof(builder: newton.ModelBuilder, joint_name: str) -> int:
    """First velocity DOF of the joint whose label ends in ``joint_name``."""
    return int(builder.joint_qd_start[find_joint(builder, joint_name)])


def joint_world_axis(builder: newton.ModelBuilder, joint: int) -> np.ndarray:
    """World-frame axis of a 1-DOF hinge at the builder's rest pose.

    ``joint_axis`` lives in the joint parent-anchor frame: the USD importer folds a hinge's
    ``localRot0`` into ``joint_X_p`` and keeps the axis at +X, so the raw ``joint_axis``
    says nothing about the world direction (unlike MJCF, whose axis is authored in the body).
    """
    dof = int(builder.joint_qd_start[joint])
    q_p = wp.transform_get_rotation(builder.joint_X_p[joint])
    parent = int(builder.joint_parent[joint])
    if parent >= 0:
        q_p = wp.mul(wp.transform_get_rotation(builder.body_q[parent]), q_p)
    return np.array(wp.quat_rotate(q_p, wp.vec3(*builder.joint_axis[dof])), dtype=np.float64)


def find_body(model_or_builder, body_name: str) -> int:
    """Index of the body whose label ends in ``body_name``."""
    for j, label in enumerate(model_or_builder.body_label):
        if label.split("/")[-1] == body_name:
            return j
    raise KeyError(f"body '{body_name}' not found")


@wp.kernel
def _drive_steer_axle(
    steer_dofs: wp.array[wp.int32],  # kingpin / steering-motor hinge DOFs — position targets
    cmd: wp.array[wp.float32],  # [0] steer_angle [rad], [1] wheel_speed [rad/s]
    axle_dofs: wp.array[wp.int32],
    axle_sign: wp.array[wp.float32],  # per axle DOF: +1 if positive joint speed rolls the vehicle forward
    joint_target_pos: wp.array[wp.float32],
    joint_target_vel: wp.array[wp.float32],
):
    """dim = n_axles.  Steered vehicle: steer position on the steer DOFs, wheel speed on the axles."""
    tid = wp.tid()
    if tid < steer_dofs.shape[0]:
        joint_target_pos[steer_dofs[tid]] = cmd[0]
    joint_target_vel[axle_dofs[tid]] = axle_sign[tid] * cmd[1]


@wp.kernel
def _drive_skid(
    left_dofs: wp.array[wp.int32],
    right_dofs: wp.array[wp.int32],
    left_sign: wp.array[wp.float32],
    right_sign: wp.array[wp.float32],
    cmd: wp.array[wp.float32],  # [0] turn in [-1, 1] (+ = left), [1] mean wheel speed [rad/s]
    joint_target_vel: wp.array[wp.float32],
):
    """dim = n_axles / 2.  Skid steer: left pair slows, right pair speeds up for a left turn."""
    tid = wp.tid()
    # Brake steering (SHERP "side-turn": a friction mechanism brakes one side): the braked
    # side slows by the lever travel, the other side keeps the throttle speed. turn > 0 =
    # left lever = left turn. Yaw rate = w r turn / (2 half_track), speed = w r (1 - turn/2).
    turn = wp.clamp(cmd[0], -1.0, 1.0)
    w = cmd[1]
    joint_target_vel[left_dofs[tid]] = left_sign[tid] * w * (1.0 - wp.max(turn, 0.0))
    joint_target_vel[right_dofs[tid]] = right_sign[tid] * w * (1.0 - wp.max(-turn, 0.0))


def _quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Rotation matrix of a unit quaternion given in USD (w, x, y, z) order."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_usd_body(stage, body_name: str) -> dict:
    """Mass properties and primitive shapes of the rigid-body prim named ``body_name`` (pxr only).

    Reads ``PhysicsMassAPI`` (mass, centre of mass, diagonal inertia + principal axes) and the
    child ``Cube`` / ``Cylinder`` / ``Capsule`` gprims (extents, local translate / orient /
    scale, display colour) the way ``mujoco-usd-converter`` writes an MJCF body (a box is a
    ``Cube`` of size 2 scaled by its half extents). Shapes carrying ``PhysicsCollisionAPI`` are
    colliders (drawn only if their display opacity is > 0); the others are visual-only.

    Returns:
        ``{"mass", "com", "inertia", "shapes": [{"name", "type", "pos", "quat_xyzw", "color",
        "visible", "collide"} + {"half"} for a box, {"radius", "half_height"} otherwise]}`` in
        the body frame.
    """
    from pxr import Usd  # noqa: PLC0415

    prims = [p for p in Usd.PrimRange(stage.GetDefaultPrim()) if p.GetName() == body_name]
    if len(prims) != 1:
        raise ValueError(f"expected one body prim named '{body_name}', found {len(prims)}")
    body = prims[0]
    if not body.HasAPI("PhysicsMassAPI") and not body.GetAttribute("physics:mass").HasAuthoredValue():
        raise ValueError(f"{body.GetPath()} has no PhysicsMassAPI")
    mass = float(body.GetAttribute("physics:mass").Get())
    com_attr = body.GetAttribute("physics:centerOfMass")
    com = np.array(com_attr.Get() if com_attr.HasAuthoredValue() else (0.0, 0.0, 0.0), dtype=np.float64)
    diag = np.array(body.GetAttribute("physics:diagonalInertia").Get(), dtype=np.float64)
    axes_attr = body.GetAttribute("physics:principalAxes")
    if axes_attr.HasAuthoredValue():
        q = axes_attr.Get()
        im = q.GetImaginary()
        rot = _quat_wxyz_to_matrix(float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))
    else:
        rot = np.eye(3)
    inertia = rot @ np.diag(diag) @ rot.T

    shapes = []
    for child in body.GetChildren():
        kind = child.GetTypeName()
        if kind not in ("Cube", "Cylinder", "Capsule"):
            continue
        t_attr = child.GetAttribute("xformOp:translate")
        o_attr = child.GetAttribute("xformOp:orient")
        pos = np.array(t_attr.Get() if t_attr.HasAuthoredValue() else (0.0, 0.0, 0.0), dtype=np.float64)
        if o_attr.HasAuthoredValue():
            q = o_attr.Get()
            im = q.GetImaginary()
            quat_xyzw = (float(im[0]), float(im[1]), float(im[2]), float(q.GetReal()))
        else:
            quat_xyzw = (0.0, 0.0, 0.0, 1.0)
        color_attr = child.GetAttribute("primvars:displayColor")
        color = tuple(float(c) for c in color_attr.Get()[0]) if color_attr.HasAuthoredValue() else None
        op_attr = child.GetAttribute("primvars:displayOpacity")
        opacity = float(op_attr.Get()[0]) if op_attr.HasAuthoredValue() else 1.0
        schemas = str(child.GetMetadata("apiSchemas") or "")
        s = {
            "name": child.GetName(),
            "type": kind.lower(),
            "pos": pos,
            "quat_xyzw": quat_xyzw,
            "color": color,
            "visible": opacity > 0.0,
            "collide": "PhysicsCollisionAPI" in schemas,
        }
        if kind == "Cube":
            size = float(child.GetAttribute("size").Get() or 2.0)
            sc_attr = child.GetAttribute("xformOp:scale")
            scale = np.array(sc_attr.Get() if sc_attr.HasAuthoredValue() else (1.0, 1.0, 1.0), dtype=np.float64)
            s["type"] = "box"
            s["half"] = 0.5 * size * scale
        else:
            axis = str(child.GetAttribute("axis").Get() or "Z")
            if axis != "Z":
                raise ValueError(f"{child.GetPath()}: only Z-axis primitives are supported, got axis {axis}")
            s["radius"] = float(child.GetAttribute("radius").Get())
            s["half_height"] = 0.5 * float(child.GetAttribute("height").Get())
        shapes.append(s)
    return {"mass": mass, "com": com, "inertia": inertia, "shapes": shapes}


def read_usd_chassis_footprint(stage, body_name: str = "chassis") -> tuple[tuple[float, float, float, float], float]:
    """Ground footprint and belly clearance of a vehicle body from its USD (pxr only).

    Union AABB (x0, x1, y0, y1) of the body's ``Cube`` colliders in the vehicle frame and the
    ground clearance of their lowest face at the authored rest pose [m]. ``mujoco-usd-converter``
    writes an MJCF box as a ``Cube`` of size 2 scaled by the half extents.
    """
    from pxr import Usd  # noqa: PLC0415

    prims = [p for p in Usd.PrimRange(stage.GetDefaultPrim()) if p.GetName() == body_name]
    if len(prims) != 1:
        raise ValueError(f"expected one body prim named '{body_name}', found {len(prims)}")
    body = prims[0]
    t_attr = body.GetAttribute("xformOp:translate")
    body_z = float(t_attr.Get()[2]) if t_attr.HasAuthoredValue() else 0.0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for child in body.GetChildren():
        if child.GetTypeName() != "Cube" or "PhysicsCollisionAPI" not in str(child.GetMetadata("apiSchemas") or ""):
            continue
        size = float(child.GetAttribute("size").Get() or 2.0)
        sc_attr = child.GetAttribute("xformOp:scale")
        half = 0.5 * size * np.array(sc_attr.Get() if sc_attr.HasAuthoredValue() else (1.0, 1.0, 1.0), dtype=np.float64)
        c_attr = child.GetAttribute("xformOp:translate")
        c = np.array(c_attr.Get() if c_attr.HasAuthoredValue() else (0.0, 0.0, 0.0), dtype=np.float64)
        lo = np.minimum(lo, c - half)
        hi = np.maximum(hi, c + half)
    if not np.all(np.isfinite(lo)):
        raise ValueError(f"{body.GetPath()} has no Cube colliders to take a footprint from")
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])), float(body_z + lo[2])


class VehicleUSD:
    """A ``--vehicle-asset`` vehicle behind the examples' vehicle hooks."""

    def __init__(self, path: str):
        from pxr import Usd, UsdGeom  # noqa: PLC0415 — optional dependency, like load_ancf_tire_usd

        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        stage = Usd.Stage.Open(path)
        root = stage.GetDefaultPrim()
        d = root.GetCustomData()
        self.kind = str(d["vehicleKind"])
        self.default_tire_asset = str(d.get("defaultTireAsset", ""))  # the tire this vehicle was built for
        self.spindle_bodies = [str(s) for s in d["spindleBodies"]]
        self.axle_joints = [str(s) for s in d["axleJoints"]]
        self.steer_joints = [str(s) for s in d.get("steerJoints", [])]
        self.steering = str(d["steering"])  # "kingpin" | "skid"
        self.max_steer = float(d["maxSteer"])
        self.max_wheel_speed = float(d["maxWheelSpeed"])
        self.rigid_corner_mass = float(d["rigidCornerMass"])
        self.hubs = np.array(d["hubs"], dtype=np.float64).reshape(-1, 3)  # world, at rest, FL FR RL RR
        self.camera_pos = tuple(float(c) for c in d["cameraPos"])
        self.camera_pitch = float(d["cameraPitch"])
        self.camera_yaw = float(d["cameraYaw"])
        self.ctis_unit = str(d["ctisUnit"])
        self.ctis_pa_per_unit = float(d["ctisPaPerUnit"])
        self.ctis_fill_time = float(d.get("ctisFillTime", 60.0))
        self.ctis_max_hoop_strain = float(d["ctisMaxHoopStrain"]) if "ctisMaxHoopStrain" in d else None
        self.ctis_circuit = bool(d.get("ctisCircuit", False))  # one air line joins all tires
        self.ctis_explicit = None
        if "ctisMax" in d:
            names = [str(s) for s in d.get("ctisPresetNames", [])]
            vals = [float(v) for v in d.get("ctisPresets", [])]
            self.ctis_explicit = (
                float(d["ctisMin"]),
                float(d["ctisMax"]),
                float(d["ctisRate"]),
                tuple(zip(vals, names, strict=True)),
            )
        self._ctis_cache = None

        if self.kind == "rigid_hull":
            hull = read_usd_body(stage, "Hull")
            self.hull_mass = hull["mass"]
            self.hull_com = hull["com"]
            self.hull_inertia = hull["inertia"]
            self.hull_shapes = hull["shapes"]
            vis = UsdGeom.Mesh(stage.GetPrimAtPath("/Vehicle/Hull/Visual"))
            self.hull_points = np.array(vis.GetPointsAttr().Get(), dtype=np.float32)
            self.hull_triangles = np.array(vis.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
            wc = stage.GetPrimAtPath("/Vehicle/Wheels").GetCustomData()
            self.hubs_local = np.array(wc["hubs"], dtype=np.float64).reshape(-1, 3)
            self.wheel_mass = float(wc["wheelMass"])
            self.axle_kv = float(wc["axleKv"])
            self.axle_damping = float(wc["axleDamping"])
            self.axle_effort_limit = float(wc.get("axleEffortLimit", 0.0)) or None  # [N m] per wheel; None = uncapped
            # Centre the wheelbase on the world origin.
            mid = 0.5 * (self.hubs_local[0] + self.hubs_local[2])
            self.hull_origin = np.array([-mid[0], -mid[1], 0.0])
            self.hubs = self.hubs_local + self.hull_origin
            # Ground footprint of the hull colliders and belly clearance (their lowest face over
            # the ground with the tires at their rolling radius), vehicle frame.
            (x0, x1, y0, y1), self.belly = read_usd_chassis_footprint(stage, "Hull")
            ox, oy = float(self.hull_origin[0]), float(self.hull_origin[1])
            self.footprint = (x0 + ox, x1 + ox, y0 + oy, y1 + oy)
        elif self.kind == "articulated":
            # The wheel body as the vehicle carries it (mass, inertia, hub / axle / bead-stop
            # shapes), so a single-wheel rig can build the very same spindle.
            self.spindle_body = read_usd_body(stage, self.spindle_bodies[0])
            self.wheel_mass = float(self.spindle_body["mass"])
            self.footprint, self.belly = read_usd_chassis_footprint(stage)
        else:
            raise ValueError(f"{self.name}: unknown vehicleKind '{self.kind}'")
        self.r_roll = float(self.hubs[0, 2])  # hub height above ground = rolling radius at rest
        self._axle_joint_idx: list[int] = []

    # ── Spec / geometry ───────────────────────────────────────────────────────

    def spec(self, tire_asset: str, tire_meta) -> VehicleSpec:
        hubs = self.hubs
        return VehicleSpec(
            tire_asset=tire_asset,
            tire_R_outer=tire_meta.R_outer,
            tire_R_inner=tire_meta.R_inner,
            tire_width=tire_meta.width,
            half_wheelbase=0.5 * float(hubs[0, 0] - hubs[2, 0]),
            half_track=0.5 * float(hubs[0, 1] - hubs[1, 1]),
            max_steer=self.max_steer,
            max_wheel_speed=self.max_wheel_speed,
            rigid_corner_mass=self.rigid_corner_mass,
            camera_pos=self.camera_pos,
            camera_pitch=self.camera_pitch,
            camera_yaw=self.camera_yaw,
        )

    def spindle_positions_zu(self) -> np.ndarray:
        """Spindle world positions (x_fwd, y_lat, z_up), FL/FR/RL/RR, before any world lift."""
        return self.hubs.astype(np.float32)

    # ── Rigid body construction ───────────────────────────────────────────────

    def build(
        self, car: newton.ModelBuilder, z_off: float, tire_spindle=None, r_bead: float = 0.0, half_bead: float = 0.0
    ) -> None:
        """Add the vehicle to ``car`` lifted by ``z_off``; Newton's default ground plane at z = 0.

        The vehicle assets carry no ground (the bake strips the MJCF world plane), so both
        vehicle kinds get the same ground here.
        """
        if self.kind == "articulated":
            car.add_ground_plane()
            car.add_usd(self.path, xform=wp.transform(wp.vec3(0.0, 0.0, z_off), wp.quat_identity()))
            return
        self._build_rigid_hull(car, z_off, tire_spindle, r_bead, half_bead)

    def _build_rigid_hull(self, car, z_off, tire_spindle, r_bead, half_bead) -> None:
        if tire_spindle is None:
            raise ValueError(f"{self.name}: a rigid-hull vehicle needs the tire asset's /Tire/Spindle for its wheels")
        vis_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)

        car.add_ground_plane()

        hull_pos = wp.vec3(*(self.hull_origin + np.array([0.0, 0.0, z_off])).tolist())
        hull = car.add_link(
            xform=wp.transform(hull_pos, wp.quat_identity()),
            mass=self.hull_mass,
            com=wp.vec3(*self.hull_com.tolist()),
            inertia=wp.mat33(*self.hull_inertia.reshape(-1).tolist()),
            label="chassis",
        )
        j_free = car.add_joint_free(hull, label="chassis_free")
        hull_mesh = newton.Mesh(self.hull_points, self.hull_triangles.reshape(-1), compute_inertia=False)
        car.add_shape_mesh(hull, mesh=hull_mesh, cfg=vis_cfg, color=(0.55, 0.58, 0.45), label="hull_vis")
        for s in self.hull_shapes:
            self._add_usd_shape(car, hull, s, label=f"hull_{s['name']}")

        # Spindles on hinges about the axle (Y), velocity actuated.
        self._axle_joint_idx = []
        for i, name in enumerate(self.spindle_bodies):
            hub_local = self.hubs_local[i]
            hub_world = self.hubs[i] + np.array([0.0, 0.0, z_off])
            spindle = self.add_spindle_link(
                car,
                tire_spindle,
                label=name,
                xform=wp.transform(wp.vec3(*hub_world.tolist()), wp.quat_identity()),
                r_bead=r_bead,
                half_bead=half_bead,
                mirror=bool(hub_local[1] < 0.0),  # right-hand wheels
            )
            j = car.add_joint_revolute(
                hull,
                spindle,
                parent_xform=wp.transform(wp.vec3(*hub_local.tolist()), wp.quat_identity()),
                axis=newton.Axis.Y,
                target_vel=0.0,
                target_kd=self.axle_kv,
                damping=self.axle_damping,
                effort_limit=self.axle_effort_limit,
                actuator_mode=newton.JointTargetMode.VELOCITY,
                label=self.axle_joints[i],
            )
            self._axle_joint_idx.append(j)
        car.add_articulation([j_free, *self._axle_joint_idx], label=self.name)

    def _add_usd_shape(self, car: newton.ModelBuilder, body: int, s: dict, label: str) -> None:
        """Add one primitive of a vehicle body as :func:`read_usd_body` returns it.

        Colliders with display opacity 0 get no VISIBLE flag, so the viewer draws them only in
        its collision view (the display meshes carry the visuals); the rest are visual-only.
        """
        if s["collide"] and s["visible"]:
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_particle_collision=False)
        elif s["collide"]:
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_particle_collision=False, is_visible=False)
        else:
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        color = s["color"] if s["visible"] else None
        xform = wp.transform(wp.vec3(*s["pos"].tolist()), wp.quat(*s["quat_xyzw"]))
        if s["type"] == "box":
            # Tiled along the footprint so each tile stays under MuJoCo's hfield prism cap on the
            # examples' 0.5 m collider cell: the narrow phase collects at most 50 prisms (25 cells)
            # under one geom's AABB, and a 1.6 m tile yawed 45 deg covers <= 5 x 5 cells; one
            # 3.8 x 2.9 m box covered ~48 -> "height field collision overflow".
            hx, hy, hz = (float(v) for v in s["half"])
            nx = max(1, math.ceil(2.0 * hx / _HULL_BOX_MAX_SIDE))
            ny = max(1, math.ceil(2.0 * hy / _HULL_BOX_MAX_SIDE))
            dx, dy = 2.0 * hx / nx, 2.0 * hy / ny
            for ix in range(nx):
                for iy in range(ny):
                    tile = wp.transform(wp.vec3(-hx + (ix + 0.5) * dx, -hy + (iy + 0.5) * dy, 0.0), wp.quat_identity())
                    car.add_shape_box(
                        body,
                        xform=wp.transform_multiply(xform, tile),
                        hx=0.5 * dx,
                        hy=0.5 * dy,
                        hz=hz,
                        cfg=cfg,
                        color=color,
                        label=f"{label}_{ix}{iy}" if nx * ny > 1 else label,
                    )
            return
        add = car.add_shape_cylinder if s["type"] == "cylinder" else car.add_shape_capsule
        add(body, xform=xform, radius=s["radius"], half_height=s["half_height"], cfg=cfg, color=color, label=label)

    def add_spindle_link(
        self,
        car: newton.ModelBuilder,
        tire_spindle,
        label: str,
        xform: wp.transform | None = None,
        r_bead: float = 0.0,
        half_bead: float = 0.0,
        mirror: bool = False,
    ) -> int:
        """Add one wheel body (Z-up, axle = +Y) exactly as this vehicle carries it; returns the body index.

        ``mirror`` reflects a rigid-hull wheel across its mid-plane for the right-hand side.

        Articulated vehicles: the spindle prim of the vehicle USD (mass, inertia, hub / axle
        visuals, invisible bead-stop collider). Rigid-hull vehicles: a cylinder at the bead
        radius carries the vehicle's wheel mass and bottoms out; the tire asset's
        ``/Tire/Spindle`` mesh is display only. Used by the vehicle build and by the
        single-wheel rig, so the rig shows and weighs the same wheel as the car.
        """
        vis_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        col_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_particle_collision=False, is_visible=False)
        xform = wp.transform_identity() if xform is None else xform

        if self.kind == "articulated":
            sb = self.spindle_body
            body = car.add_link(
                xform=xform,
                mass=float(sb["mass"]),
                com=wp.vec3(*sb["com"].tolist()),
                inertia=wp.mat33(*sb["inertia"].reshape(-1).tolist()),
                label=label,
            )
            for k, s in enumerate(sb["shapes"]):
                self._add_usd_shape(car, body, s, label=f"{label}_{s['type']}{k}")
            return body

        if tire_spindle is None:
            raise ValueError(f"{self.name}: a rigid-hull vehicle needs the tire asset's /Tire/Spindle for its wheels")
        if r_bead <= 0.0 or half_bead <= 0.0:
            raise ValueError(f"{self.name}: a rigid-hull wheel needs the tire's bead radius and half width")
        P = _P_YUP_TO_ZU
        if mirror:
            # Right-side wheel: the tire asset's spindle has its hub cap on the tire-frame +X
            # (= vehicle +Y, outboard on the left). Mirror across the wheel's mid-plane so the
            # cap faces outboard on the right too (the URDF does this with a yaw of pi); the
            # body frame keeps the same axle direction so every hinge has the same sign.
            P = np.diag([1.0, -1.0, 1.0]) @ P
        sp_pts = (tire_spindle.points.astype(np.float64) @ P.T).astype(np.float32)
        sp_tris = tire_spindle.triangle_indices.copy()
        if mirror:
            sp_tris = sp_tris[:, [0, 2, 1]]  # a reflection flips the winding; keep normals outward
        sp_mesh = newton.Mesh(sp_pts, sp_tris.reshape(-1), compute_inertia=False)
        # The wheel's mass is the bead-stop cylinder (rim, wheel-disk tank, hub) at the hub
        # centre, axle along Y, like the hull's primitives; the spindle mesh is display only.
        m = self.wheel_mass
        i_ax = 0.5 * m * r_bead * r_bead
        i_tr = m * (3.0 * r_bead * r_bead + 4.0 * half_bead * half_bead) / 12.0
        body = car.add_link(
            xform=xform,
            mass=m,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(i_tr, 0.0, 0.0, 0.0, i_ax, 0.0, 0.0, 0.0, i_tr),
            label=label,
        )
        car.add_shape_mesh(body, mesh=sp_mesh, cfg=vis_cfg, color=(0.35, 0.35, 0.75), label=f"{label}_vis")
        q_cyl = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)  # cylinder Z -> axle Y
        car.add_shape_cylinder(
            body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), q_cyl),
            radius=r_bead,
            half_height=half_bead,
            cfg=col_cfg,
            label=f"{label}_bead_stop",
        )
        return body

    # ── Drive ─────────────────────────────────────────────────────────────────

    def axle_forward_signs(self, car: newton.ModelBuilder) -> list[float]:
        """Per axle joint: +1 if a positive hinge speed rolls the vehicle forward, else -1.

        Rolling with the centre moving along +f (forward) on a Z-up ground has the angular
        velocity along +(z x f) = the vehicle's left axis; forward is the hub-derived side
        the front spindles sit on. Derived from the hinge's world axis, so an MJCF/USD axle
        authored as ``0 -1 0`` (positive speed = reverse) drives the right way.
        """
        fwd_x = 1.0 if float(self.hubs[0, 0]) >= 0.0 else -1.0  # hubs[0] = FL spindle
        left = np.array([0.0, fwd_x, 0.0])  # z x (fwd_x, 0, 0)
        signs = []
        for n in self.axle_joints:
            ax = joint_world_axis(car, find_joint(car, n))
            signs.append(1.0 if float(ax @ left) > 0.0 else -1.0)
        return signs

    def setup_drive(self, car: newton.ModelBuilder, device: str) -> None:
        axle = [find_dof(car, n) for n in self.axle_joints]  # FL, FR, RL, RR
        sign = self.axle_forward_signs(car)
        self._axle_dofs = wp.array(axle, dtype=wp.int32, device=device)
        self._axle_sign = wp.array(sign, dtype=wp.float32, device=device)
        if self.steering == "skid":
            self._left_dofs = wp.array(axle[0::2], dtype=wp.int32, device=device)
            self._right_dofs = wp.array(axle[1::2], dtype=wp.int32, device=device)
            self._left_sign = wp.array(sign[0::2], dtype=wp.float32, device=device)
            self._right_sign = wp.array(sign[1::2], dtype=wp.float32, device=device)
        else:
            self._steer_dofs = wp.array([find_dof(car, n) for n in self.steer_joints], dtype=wp.int32, device=device)

    def launch_drive(
        self,
        cmd: wp.array[wp.float32],
        joint_target_q: wp.array[wp.float32],
        joint_target_qd: wp.array[wp.float32],
    ) -> None:
        """Write ``cmd`` ([0] steer / turn, [1] wheel speed, + = forward) into the joint targets (graph-safe)."""
        if self.steering == "skid":
            wp.launch(
                _drive_skid,
                dim=self._left_dofs.shape[0],
                inputs=[self._left_dofs, self._right_dofs, self._left_sign, self._right_sign, cmd, joint_target_qd],
                device=cmd.device,
            )
        else:
            wp.launch(
                _drive_steer_axle,
                dim=self._axle_dofs.shape[0],
                inputs=[self._steer_dofs, cmd, self._axle_dofs, self._axle_sign, joint_target_q, joint_target_qd],
                device=cmd.device,
            )

    def drive_axes(self, car: newton.ModelBuilder) -> tuple[float, float]:
        """``(kingpin_axis_z, axle_sign)`` for the track / field drivers.

        ``kingpin_axis_z``: world z component of the first steer hinge axis (sign of a positive
        steer command; +1 for skid steer). ``axle_sign``: always +1 — kept in the tuple for the
        callers that unpack it, since :meth:`launch_drive` already applies the per-axle forward
        sign, so callers pass wheel speed with + = forward.
        """
        if self.steering == "skid":
            return 1.0, 1.0
        return float(joint_world_axis(car, find_joint(car, self.steer_joints[0]))[2]), 1.0

    # ── UI / CTIS ─────────────────────────────────────────────────────────────

    def gui_drive(self, ui, example) -> None:
        spec = example.spec
        vmax = spec.max_wheel_speed
        if self.steering == "skid":
            # Two brake levers like the real cabin: pulling one brakes that side's wheels.
            # example.steer_angle holds the net turn command (left lever - right lever).
            ui.text(f"{self.name} + ANCF tires — skid steer (brake levers)")
            ui.separator()
            turn = float(example.steer_angle)
            lever_l, lever_r = max(turn, 0.0), max(-turn, 0.0)
            changed_l, lever_l = ui.slider_float("left lever  (brake left)", lever_l, 0.0, 1.0)
            changed_r, lever_r = ui.slider_float("right lever (brake right)", lever_r, 0.0, 1.0)
            if changed_l or changed_r:
                # One lever at a time: the one just moved wins, like releasing the other.
                if changed_l and lever_l > 0.0:
                    lever_r = 0.0
                elif changed_r and lever_r > 0.0:
                    lever_l = 0.0
                example.steer_angle = float(lever_l - lever_r)
            changed, val = ui.slider_float("throttle [rad/s]", example._target_wheel_speed, -vmax, vmax)
            if changed:
                example._target_wheel_speed = float(val)
            ui.separator()
            w = example.wheel_speed
            v_l = w * (1.0 - lever_l) * self.r_roll
            v_r = w * (1.0 - lever_r) * self.r_roll
            ui.text(f"left wheels   {v_l:+.2f} m/s   right wheels {v_r:+.2f} m/s")
            ui.text(f"forward speed ~ {0.5 * (v_l + v_r):+.2f} m/s")
            ui.text(f"yaw rate      ~ {(v_r - v_l) / (2.0 * spec.half_track):+.2f} rad/s (+ = left)")
            return
        ui.text(f"{self.name} + ANCF tires — {self.steering} steer")
        ui.separator()
        changed, val = ui.slider_float("steer [rad]", example.steer_angle, -spec.max_steer, spec.max_steer)
        if changed:
            example.steer_angle = float(val)
        changed, val = ui.slider_float("throttle [rad/s]", example._target_wheel_speed, -vmax, vmax)
        if changed:
            example._target_wheel_speed = float(val)
        ui.separator()
        ui.text(f"forward speed ~ {example.wheel_speed * self.r_roll:+.2f} m/s")
        if abs(example.steer_angle) > 1e-3:
            ui.text(f"turn radius   ~ {(2.0 * spec.half_wheelbase) / math.tan(abs(example.steer_angle)):.2f} m")
        else:
            ui.text("turn radius   ~ straight")

    def make_ctis(self, solver, n_tires: int, pressure: float, build_pressure: float, envelope):
        """The vehicle's inflation system on ``solver`` (:mod:`_ctis`): one air line joining the
        tires when the asset says ``ctisCircuit``, else a valve per tire."""
        from newton.examples.ancf._ctis import CtisCircuit, CtisPerTire  # noqa: PLC0415

        cls = CtisCircuit if self.ctis_circuit else CtisPerTire
        return cls(solver, n_tires, pressure, build_pressure, envelope)

    def ctis_envelope(self, e_tire: float, elem_h: np.ndarray, tire_meta):
        """(unit, Pa/unit, min, max, rate [unit/s], presets, title).

        Explicit envelope from the asset when it has one; otherwise the setpoint cap comes
        from the fitted tire's crown hoop strain p R / (h E) <= ctisMaxHoopStrain.
        """
        if self._ctis_cache is None:
            unit, per = self.ctis_unit, self.ctis_pa_per_unit
            if self.ctis_explicit is not None:
                lo, hi, rate, presets = self.ctis_explicit
                self._ctis_cache = (unit, per, lo, hi, rate, presets, self.name)
            else:
                n_rows = len(elem_h) // tire_meta.n_circ
                h_crown = float(elem_h[(n_rows // 2) * tire_meta.n_circ])  # tread-section thickness, centre row
                p_max = e_tire * h_crown * self.ctis_max_hoop_strain / tire_meta.R_outer  # [Pa]
                hi = float(round(p_max / per))
                levels = ((0.25, "soft"), (0.5, "trail"), (0.75, "firm"), (1.0, "max"))
                presets = tuple((float(round(hi * f)), name) for f, name in levels)
                self._ctis_cache = (unit, per, 0.0, hi, hi / self.ctis_fill_time, presets, "hoop-strain limit")
        return self._ctis_cache
