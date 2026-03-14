# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from time import perf_counter
from typing import Any

# Spinner characters for "Loading RTX viewer..." progress
_RTX_SPINNER = ("/", "-", "\\", "|")


def _should_filter_omni_stderr(line: str) -> bool:
    """Return True if line is a known noisy omni.* warning we want to hide."""
    if "omni.fabric.internal" in line and (
        "Unsupported type encountered during VtValue extraction" in line
        or "Attempting to set an invalid data source to array attribute" in line
    ):
        return True
    if "omni.hydra" in line and "was already registered" in line:
        return True
    return False


class _FilteredStderr:
    """Wrapper for sys.stderr that drops known omni.fabric / omni.hydra warning lines."""

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    def write(self, s: str):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line += "\n"
            if not _should_filter_omni_stderr(line):
                self._stream.write(line)

    def flush(self):
        if self._buf and not _should_filter_omni_stderr(self._buf + "\n"):
            self._stream.write(self._buf)
        self._buf = ""
        self._stream.flush()

    def fileno(self):
        return self._stream.fileno()

import numpy as np
import warp as wp

from ..core.types import override

try:
    from pxr import Gf, UsdGeom
except ImportError:
    Gf = UsdGeom = None

from .camera import Camera
from .picking import Picking
from .viewer_gui import ViewerGui
from .viewer_usd import ViewerUSD
from .wind import Wind

PROFILE_ENABLED = os.environ.get("NEWTON_PROFILE", "0") != "0"


@wp.kernel(enable_backward=False)
def write_transforms(
    xform: wp.array(dtype=wp.transform), scale: wp.array(dtype=wp.vec3), offset: int, m_out: wp.array(dtype=wp.mat44d)
):
    tid = wp.tid()
    xf32 = xform[tid]
    sc32 = scale[tid]
    # convert to float64
    p64 = wp.vec3d(wp.float64(xf32[0]), wp.float64(xf32[1]), wp.float64(xf32[2]))
    q64 = wp.quatd(wp.float64(xf32[3]), wp.float64(xf32[4]), wp.float64(xf32[5]), wp.float64(xf32[6]))
    s64 = wp.vec3d(wp.float64(sc32[0]), wp.float64(sc32[1]), wp.float64(sc32[2]))
    # NOTE: transpose needed
    m_out[offset + tid] = wp.transpose(wp.transform_compose(p64, q64, s64))


class ViewerRTX(ViewerUSD):
    """Real-time ray-traced viewer using NVIDIA OVRTX.

    Builds a USD scene during the first simulation frame using the ViewerUSD
    base class, serializes it to disk, then creates an OVRTX renderer for
    real-time path-traced rendering.  Subsequent frames update rigid-body
    transforms (and deforming-mesh vertices) via the OVRTX attribute API
    and present the rendered image in a pyglet / OpenGL window.
    """

    _PHASE_BUILD = 0
    _PHASE_RENDER = 1
    _PICKING_LINE_NAME = "picking_line"
    _PICKING_LINE_RADIUS = 0.01
    _PICKING_LINE_COLOR = (0.0, 1.0, 1.0)
    _POINT_SPHERE_PROTOTYPE_PATH = "/World/Prototypes/PointSphere"
    # Scale down point-cloud sphere radius for display so MPM sand looks like sand grains
    _POINT_SPHERE_DISPLAY_SCALE = 0.2

    # Available lighting environment presets.
    ENVIRONMENTS = ("default", "studio", "none")

    def __init__(
        self,
        width=1280,
        height=720,
        fps=60,
        up_axis="Z",
        num_frames=None,
        scaling=1.0,
        headless=False,
        paused=False,
        environment="default",
        vsync=False,
    ):
        # Filter noisy omni.* warnings before any ovrtx/omni code runs (e.g. hydra "already registered").
        if not isinstance(sys.stderr, _FilteredStderr):
            sys.stderr = _FilteredStderr(sys.stderr)

        os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

        try:
            import ovrtx  # noqa: F401
        except ImportError as e:
            raise ImportError("ovrtx package is required for ViewerRTX. Install with: pip install ovrtx") from e

        if UsdGeom is None:
            raise ImportError("usd-core package is required for ViewerRTX. Install with: pip install usd-core")

        self._environment = environment.lower()
        if self._environment not in self.ENVIRONMENTS:
            raise ValueError(
                f"Unknown RTX environment {self._environment!r}. Choose from: {', '.join(self.ENVIRONMENTS)}"
            )

        self._tmp_usd_path = os.path.abspath("_tmp_rtx_scene.usd")

        # Pre-initialize fields that clear_model() (called from super().__init__) touches
        self._ui_callbacks: dict[str, list] = {"side": [], "stats": [], "free": [], "panel": []}
        self._paused = False

        super().__init__(
            output_path=self._tmp_usd_path,
            fps=fps,
            up_axis=up_axis,
            num_frames=num_frames,
            scaling=scaling,
        )

        self._width = width
        self._height = height
        self._headless = headless

        # OVRTX renderer (created at end of first frame)
        self._rtx = None
        self._phase = self._PHASE_BUILD

        # Pyglet window state
        self._window = None
        self._pyglet_app = None
        self._should_close = False
        self._camera_dirty = True

        # Instance prim paths collected during build phase, keyed by instancer name.
        # Iteration order (insertion order) defines the layout inside the binding.
        self._instance_prim_paths: dict[str, list[str]] = {}
        self._all_instance_paths: list[str] = []
        self._transform_binding = None

        # Timing for first RTX frame (from start of _init_ovrtx to first displayed image).
        self._rtx_first_frame_start: float | None = None
        self._rtx_first_frame_done = threading.Event()
        self._rtx_spinner_thread: threading.Thread | None = None

        # Mesh prim paths for deforming-mesh point updates
        self._mesh_prim_paths: dict[str, str] = {}

        # Per-frame pending data (cleared each begin_frame)
        self._pending_xforms: dict[str, tuple[Any, Any]] = {}
        self._pending_mesh_points: dict[str, np.ndarray] = {}
        self._pending_mesh_widths: dict[str, np.ndarray] = {}

        # Last rendered frame pixels (set in _render_and_display for save_screenshot)
        self._last_frame_pixels: np.ndarray | None = None

        # Camera (shared Camera class with ViewerGL)
        self.camera = Camera(width=self._width, height=self._height, up_axis=up_axis)
        self._camera_prim_path = "/World/Camera"
        self._render_product_path = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"

        # Input / timing state
        self._keys_down: set[int] = set()
        self._last_perf_time: float | None = None
        self.gui = None

        # FPS tracking (used by ViewerGui.render_frame)
        self._fps_history: list[float] = []
        self._last_fps_time: float = perf_counter()
        self._fps_frame_count: int = 0
        self._current_fps: float = 0.0

        # async rendering
        self._render_result = None

        # Window is deferred until _init_ovrtx to avoid pyglet/Warp
        # kernel compilation deadlock on Windows.

        # initial value for vsync (applied once window is created)
        self._vsync_init = vsync

    # ------------------------------------------------------------------ window

    def _init_window(self):
        """Create a pyglet window with GL texture + shader for fast framebuffer blitting."""
        import ctypes  # noqa: PLC0415

        import pyglet

        pyglet.options["debug_gl"] = False
        from pyglet import gl

        self._window = pyglet.window.Window(
            width=self._width,
            height=self._height,
            caption="Newton RTX Viewer",
            resizable=True,
            visible=not self._headless,
            vsync=self._vsync_init,
        )
        self._pyglet_app = pyglet.app
        # Request keyboard focus so key events (I/K, Space, etc.) are received (important for X11/Docker).
        self._focus_activate_frames_left = 15
        if not self._headless and hasattr(self._window, "activate"):
            try:
                self._window.activate()
            except Exception:
                pass

        # ---- GL texture + shader for zero-copy blit --------------------------
        self._window.switch_to()

        tex_id = (gl.GLuint * 1)()
        gl.glGenTextures(1, tex_id)
        self._gl_texture = tex_id[0]
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            self._width,
            self._height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Compile fullscreen-triangle shader (linear→sRGB gamma + Y-flip in fragment)
        _VS = b"""#version 330
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
\x00"""
        _FS = b"""#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(tex, vec2(uv.x, 1.0 - uv.y));
    fragColor = c;
}
\x00"""

        def _compile_shader(src, stype):
            s = gl.glCreateShader(stype)
            src_p = ctypes.c_char_p(src)
            src_pp = (ctypes.c_char_p * 1)(src_p)
            gl.glShaderSource(s, 1, ctypes.cast(src_pp, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))), None)
            gl.glCompileShader(s)
            return s

        vs = _compile_shader(_VS, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_FS, gl.GL_FRAGMENT_SHADER)
        self._gl_program = gl.glCreateProgram()
        gl.glAttachShader(self._gl_program, vs)
        gl.glAttachShader(self._gl_program, fs)
        gl.glLinkProgram(self._gl_program)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)

        # Empty VAO required by core profile for the fullscreen triangle
        vao = (gl.GLuint * 1)()
        gl.glGenVertexArrays(1, vao)
        self._gl_vao = vao[0]

        # ---- input callbacks ------------------------------------------------
        @self._window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            allow_active_pick_drag = (
                bool(buttons & pyglet.window.mouse.RIGHT)
                and self.picking_enabled
                and self.gui is not None
                and self.gui.is_pick_active()
            )
            if self.gui and self.gui.should_ignore_mouse_input(allow_active_pick_drag=allow_active_pick_drag):
                return
            if buttons & pyglet.window.mouse.LEFT and self.gui:
                self.gui.rotate_camera_from_drag(dx, dy, sensitivity=0.1)
            if buttons & pyglet.window.mouse.RIGHT and self.picking_enabled and self.picking is not None:
                if self.gui:
                    self.gui.update_picking_from_screen(x, y, self._to_framebuffer_coords)

        @self._window.event
        def on_mouse_press(x, y, button, modifiers):
            if self.gui and self.gui.should_ignore_mouse_input():
                return
            if button == pyglet.window.mouse.RIGHT and self.picking_enabled and self.picking is not None:
                if self.gui:
                    self.gui.start_picking_from_screen(x, y, self._to_framebuffer_coords)

        @self._window.event
        def on_mouse_release(x, y, button, modifiers):
            if button == pyglet.window.mouse.RIGHT and self.picking is not None:
                if self.gui:
                    self.gui.release_picking()

        @self._window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            if self.gui and self.gui.should_ignore_mouse_input():
                return
            if self.gui:
                self.gui.adjust_camera_fov_from_scroll(scroll_y, scale=2.0)

        @self._window.event
        def on_key_press(symbol, modifiers):
            if self.gui and self.gui.should_ignore_keyboard_input():
                return
            self._keys_down.add(symbol)
            if symbol == pyglet.window.key.ESCAPE:
                self._window.close()
            elif symbol == pyglet.window.key.SPACE:
                self._paused = not self._paused
            elif symbol == pyglet.window.key.H:
                self.show_ui = not self.show_ui
            elif symbol == pyglet.window.key.F:
                self._frame_camera_on_model()

        @self._window.event
        def on_key_release(symbol, modifiers):
            self._keys_down.discard(symbol)

        @self._window.event
        def on_close():
            self._should_close = True

        self.gui = ViewerGui(self, self._window)

    @property
    def ui(self):
        if self.gui is None:
            return None
        return self.gui.ui

    @property
    def vsync(self) -> bool:
        """
        Get the current vsync state.

        Returns:
            bool: True if vsync is enabled, False otherwise.
        """
        if self._window is not None:
            return self._window.vsync
        else:
            return self._vsync_init

    @vsync.setter
    def vsync(self, enabled: bool):
        """
        Set the vsync state.

        Args:
            enabled: Enable or disable vsync.
        """
        if self._window is not None:
            self._window.set_vsync(enabled)
        else:
            self._vsync_init = enabled

    # ------------------------------------------------------------------ camera

    def _compute_camera_matrix(self):
        """Return a 4x4 row-major world-transform for the camera prim (USD convention)."""
        fwd = np.array(self.camera.get_front(), dtype=np.float64)
        right = np.array(self.camera.get_right(), dtype=np.float64)
        up = np.array(self.camera.get_up(), dtype=np.float64)

        mat = np.eye(4, dtype=np.float64)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -fwd  # USD cameras look along local -Z
        mat[3, :3] = np.array(self.camera.pos, dtype=np.float64)
        return mat

    def _to_framebuffer_coords(self, x: float, y: float) -> tuple[float, float]:
        if self.gui:
            return self.gui.map_window_to_target_coords(x, y, self._window, target_size=(self._width, self._height))
        return float(x), float(y)

    def _frame_camera_on_model(self):
        """Frame the camera to show all visible objects in the scene."""
        if self.model is None:
            return
        from pyglet.math import Vec3 as PyVec3

        min_bounds = np.array([float("inf")] * 3)
        max_bounds = np.array([float("-inf")] * 3)
        found_objects = False

        state = getattr(self, "_last_state", None)
        if state is not None:
            if getattr(state, "body_q", None) is not None:
                body_q = state.body_q.numpy()
                positions = body_q[:, :3]
                min_bounds = np.minimum(min_bounds, positions.min(axis=0))
                max_bounds = np.maximum(max_bounds, positions.max(axis=0))
                found_objects = True
            if getattr(state, "particle_q", None) is not None:
                pq = state.particle_q.numpy()
                if len(pq) > 0:
                    min_bounds = np.minimum(min_bounds, pq.min(axis=0))
                    max_bounds = np.maximum(max_bounds, pq.max(axis=0))
                    found_objects = True

        if not found_objects:
            min_bounds = np.array([-5.0, -5.0, -5.0])
            max_bounds = np.array([5.0, 5.0, 5.0])

        center = (min_bounds + max_bounds) * 0.5
        size = max_bounds - min_bounds
        max_extent = float(np.max(size))
        if max_extent < 1.0:
            max_extent = 1.0

        fov_rad = np.radians(self.camera.fov)
        padding = 1.5
        distance = max_extent / (2.0 * np.tan(fov_rad / 2.0)) * padding
        front = self.camera.get_front()
        self.camera.pos = PyVec3(
            center[0] - front.x * distance,
            center[1] - front.y * distance,
            center[2] - front.z * distance,
        )
        self._camera_dirty = True

    # -------------------------------------------------------- USD scene helpers

    def _add_camera_lights_and_render_product(self):
        """Insert camera, lights, and RenderProduct into the stage before serialisation."""
        from pxr import Sdf

        # ---- Camera ----------------------------------------------------------
        cam = UsdGeom.Camera.Define(self.stage, self._camera_prim_path)

        aspect = self._width / max(self._height, 1)
        # camera.fov is vertical FOV, so derive focal length from the vertical aperture.
        v_aperture = 20.955
        h_aperture = v_aperture * aspect
        focal_length = v_aperture / (2.0 * math.tan(math.radians(self.camera.fov) / 2.0))

        cam.GetFocalLengthAttr().Set(focal_length)
        cam.GetHorizontalApertureAttr().Set(h_aperture)
        cam.GetVerticalApertureAttr().Set(v_aperture)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(self.camera.near, self.camera.far))

        xform = UsdGeom.Xform(cam.GetPrim())
        xform.ClearXformOpOrder()
        mat_op = xform.AddTransformOp()
        cam_mat = self._compute_camera_matrix()
        gf_mat = Gf.Matrix4d(*cam_mat.flatten().tolist())
        mat_op.Set(gf_mat)

        # ---- Lights ----------------------------------------------------------
        if self._environment == "studio":
            self._add_studio_lights()
        elif self._environment == "default":
            self._add_default_lights()

        # ---- Render hierarchy (must match Kit convention for OVRTX) ------------
        # Structure: /Render/OmniverseKit/HydraTextures/<product>
        #            /Render/Vars/LdrColor
        #            /Render/OmniverseGlobalRenderSettings
        self.stage.DefinePrim("/Render")
        self.stage.DefinePrim("/Render/OmniverseKit")
        self.stage.DefinePrim("/Render/OmniverseKit/HydraTextures")

        rp = self.stage.DefinePrim(self._render_product_path, "RenderProduct")
        rp.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=[
                    "OmniRtxSettingsCommonAdvancedAPI_1",
                    "OmniRtxSettingsRtAdvancedAPI_1",
                    "OmniRtxSettingsPtAdvancedAPI_1",
                    "OmniRtxPostColorGradingAPI_1",
                    "OmniRtxPostChromaticAberrationAPI_1",
                    "OmniRtxPostBloomPhysicalAPI_1",
                    "OmniRtxPostMatteObjectAPI_1",
                    "OmniRtxPostCompositingAPI_1",
                    "OmniRtxPostDofAPI_1",
                    "OmniRtxPostMotionBlurAPI_1",
                    "OmniRtxPostTvNoiseAPI_1",
                    "OmniRtxPostTonemapIrayReinhardAPI_1",
                    "OmniRtxPostDebugSettingsAPI_1",
                    "OmniRtxDebugSettingsAPI_1",
                ]
            ),
        )
        rp.CreateRelationship("camera").SetTargets([Sdf.Path(self._camera_prim_path)])
        rp.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2, custom=False).Set(Gf.Vec2i(self._width, self._height))

        # RenderVar lives at /Render/Vars/LdrColor (NOT nested under the product)
        rv_path = "/Render/Vars/LdrColor"
        rv = self.stage.DefinePrim(rv_path, "RenderVar")
        rv.CreateAttribute("sourceName", Sdf.ValueTypeNames.String, custom=False).Set("LdrColor")
        rp.CreateRelationship("orderedVars").SetTargets([Sdf.Path(rv_path)])

        # ---- RTX render settings on the RenderProduct -------------------------
        rp.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set("RealTimePathTracing")
        rp.CreateAttribute("omni:rtx:ambientOcclusion:denoiserMode", Sdf.ValueTypeNames.Token).Set("none")
        rp.CreateAttribute("omni:rtx:background:source:texture:textureMode", Sdf.ValueTypeNames.Token).Set(
            "repeatMirrored"
        )
        rp.CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("domeLight")
        rp.CreateAttribute("omni:rtx:debug:view:pixelDebug:enableFixedTextPos", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:directLighting:sampledLighting:denoisingTechnique", Sdf.ValueTypeNames.Token).Set(
            "None"
        )
        rp.CreateAttribute("omni:rtx:dlss:frameGeneration", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:indirectDiffuse:denoiser:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:post:aa:limitedOps", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:post:registeredCompositing:invertColorCorrection", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        rp.CreateAttribute("omni:rtx:post:registeredCompositing:invertToneMap", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:pt:maxSamplesPerLaunch", Sdf.ValueTypeNames.Int).Set(2073600)
        rp.CreateAttribute("omni:rtx:pt:mgpu:maxPixelsPerRegionExponent", Sdf.ValueTypeNames.Int).Set(12)
        rp.CreateAttribute("omni:rtx:pt:denoising:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:pt:samplesPerPixel", Sdf.ValueTypeNames.UInt).Set(1)
        rp.CreateAttribute("omni:rtx:reflections:denoiser:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:rt:ambientLight:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
        rp.CreateAttribute("omni:rtx:rt:demoire", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:rt:lightcache:spatialCache:dontResolveConflicts", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        rp.CreateAttribute("omni:rtx:rt:sss:samples", Sdf.ValueTypeNames.Int).Set(1)
        rp.CreateAttribute("omni:rtx:rtpt:maxVolumeBounces", Sdf.ValueTypeNames.Int).Set(15)
        rp.CreateAttribute("omni:rtx:rtpt:modulatingRoughnessThreshold", Sdf.ValueTypeNames.Float).Set(0.08)
        rp.CreateAttribute("omni:rtx:scene:hydra:mdlMaterialWarmup", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:viewTile:limit", Sdf.ValueTypeNames.UInt).Set(4294967295)

        # Disable the quality convergence loop to minimize step() latency
        rp.CreateAttribute("omni:rtx:quality", Sdf.ValueTypeNames.Int, custom=False).Set(0)
        rp.CreateAttribute("omni:rtx:waitForEvents", Sdf.ValueTypeNames.TokenArray).Set([])

        # ---- RenderSettings --------------------------------------------------
        rs = self.stage.DefinePrim("/Render/OmniverseGlobalRenderSettings", "RenderSettings")
        rs.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=[
                    "OmniRtxSettingsGlobalRtAdvancedAPI_1",
                    "OmniRtxSettingsGlobalPtAdvancedAPI_1",
                ]
            ),
        )
        rs.CreateRelationship("products").SetTargets([Sdf.Path(self._render_product_path)])

    def _add_default_lights(self):
        """Default lighting: dome light + distant directional light."""
        from pxr import UsdLux

        dome = UsdLux.DomeLight.Define(self.stage, "/root/_RTXDomeLight")
        dome.GetIntensityAttr().Set(150.0)

        distant = UsdLux.DistantLight.Define(self.stage, "/root/_RTXDistantLight")
        distant.GetIntensityAttr().Set(900.0)
        distant.GetAngleAttr().Set(0.53)
        dx = UsdGeom.Xform(distant.GetPrim())
        dx.ClearXformOpOrder()
        rot = dx.AddRotateXYZOp()
        if self.camera.up_axis == 2:
            rot.Set(Gf.Vec3f(-45.0, 30.0, 0.0))
        else:
            rot.Set(Gf.Vec3f(-45.0, 0.0, 30.0))

    def _add_studio_lights(self):
        """Studio lighting rig from dome + warm distant + cool fill sphere."""
        from pxr import Sdf, UsdLux

        # Dome light — cool-tinted low ambient
        dome_xf = UsdGeom.Xform.Define(self.stage, "/root/_RTXDomeLight")
        dome_xf.ClearXformOpOrder()
        dome = UsdLux.DomeLight.Define(self.stage, "/root/_RTXDomeLight/_RTXDomeLight")
        dome.GetColorAttr().Set(Gf.Vec3f(0.250, 0.319, 0.409))
        dome.GetIntensityAttr().Set(200.0)

        # Distant light — warm key, angled from above-behind
        dist_xf = UsdGeom.Xform.Define(self.stage, "/root/_RTXDistantLight")
        dist_xf.ClearXformOpOrder()
        dist_xf.AddRotateXYZOp().Set(Gf.Vec3f(41.4, 0.0, -175.7))
        distant = UsdLux.DistantLight.Define(self.stage, "/root/_RTXDistantLight/_RTXDistantLight")
        distant.GetColorAttr().Set(Gf.Vec3f(1.0, 0.906, 0.722))
        distant.GetIntensityAttr().Set(3000.0)

        # Cool fill sphere light (blue-white)
        fill_xf = UsdGeom.Xform.Define(self.stage, "/root/_RTXFillLight")
        fill_xf.ClearXformOpOrder()
        fill_xf.AddTranslateOp().Set(Gf.Vec3d(5.0, 0.0, 5.5))
        fill = UsdLux.SphereLight.Define(self.stage, "/root/_RTXFillLight/_RTXFillLight")
        fill.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["ShapingAPI"]))
        fill.GetColorAttr().Set(Gf.Vec3f(0.468, 0.684, 1.0))
        fill.GetIntensityAttr().Set(300000.0)
        fill.GetRadiusAttr().Set(0.5)

    def _apply_ground_material(self):
        """Bind a dark, shiny UsdPreviewSurface material to ground-plane meshes."""
        from pxr import Sdf, UsdShade

        plane_prims = [prim for name, prim in self._meshes.items() if "plane" in name.lower()]
        if not plane_prims:
            return

        mat_path = "/root/Materials/mat_ground"
        self._ensure_scopes_for_path(self.stage, mat_path)

        material = UsdShade.Material.Define(self.stage, mat_path)
        surface = UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.05, 0.06))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.15)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

        for prim in plane_prims:
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
            UsdShade.MaterialBindingAPI(prim).Bind(material)

    def add_background_usd(self, path: str):
        """Add a reference to a background USD (e.g. Gaussian splat scan).

        Must be called before the first frame (during the build phase).

        Args:
            path: Absolute or relative path to a USD file.
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Background USD not found: {path}")
        bg_prim = self.stage.DefinePrim("/root/background")
        bg_prim.GetReferences().AddReference(path)

    # ------------------------------------------------------------- OVRTX init

    def _init_ovrtx(self):
        """Serialise the USD stage, create the OVRTX renderer and load the scene."""
        import ovrtx

        self._rtx_first_frame_start = perf_counter()
        self._rtx_first_frame_done.clear()

        def _spinner():
            i = 0
            while not self._rtx_first_frame_done.wait(timeout=0.12):
                c = _RTX_SPINNER[i % len(_RTX_SPINNER)]
                print(f"\rLoading RTX viewer... {c} ", end="", flush=True)
                i += 1
            print("\r" + " " * 28 + "\r", end="", flush=True)

        self._rtx_spinner_thread = threading.Thread(target=_spinner, daemon=True)
        self._rtx_spinner_thread.start()

        # Try to reduce Omniverse/Carb log level so omni.fabric.internal warnings
        # (Unsupported type / invalid data source for points) are less noisy.
        try:
            import carb  # noqa: PLC0415
            settings = carb.settings.get_settings()
            for key in ("/log/level", "/log/fileLogLevel", "/log/outputStreamLevel"):
                try:
                    settings.set_string(key, "Error")
                except Exception:
                    pass
        except Exception:  # carb may not be available or settings may not take effect
            pass

        self._add_camera_lights_and_render_product()
        self._apply_ground_material()

        self.stage.GetRootLayer().Save()

        config = ovrtx.RendererConfig()
        config.log_level = "error"
        try:
            self._rtx = ovrtx.Renderer(config=config)
        except RuntimeError as e:
            msg = str(e)
            if "Vulkan" in msg or "ERROR_INCOMPATIBLE_DRIVER" in msg or "createDevices failed" in msg or "Failed to setup graphics" in msg:
                raise RuntimeError(
                    "OVRTX failed (Vulkan 1.1 / graphics driver not available, e.g. in some containers). "
                    "Use --viewer gl to run with the OpenGL viewer instead."
                ) from e
            raise
        self._rtx.add_usd(self._tmp_usd_path)

        # Flat prim-path list for a single transform binding
        self._all_instance_paths = []
        for paths in self._instance_prim_paths.values():
            self._all_instance_paths.extend(paths)

        if self._all_instance_paths:
            from ovrtx import PrimMode, Semantic

            self._transform_binding = self._rtx.bind_attribute(
                prim_paths=self._all_instance_paths,
                attribute_name="omni:xform",
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.MUST_EXIST,
            )

        # Create the presentation window now that all Warp kernels have been
        # compiled.  Doing this earlier causes a deadlock on Windows because
        # the Win32 message pump and Warp's JIT compilation fight for the
        # main thread.
        self._init_window()

        self._phase = self._PHASE_RENDER

    # ------------------------------------------------ ViewerUSD overrides

    @override
    def set_model(self, model, max_worlds=None):
        super().set_model(model, max_worlds=max_worlds)
        if model is not None:
            from pyglet.math import Vec3 as PyVec3

            axis_idx = (
                model.up_axis
                if isinstance(model.up_axis, int)
                else {"X": 0, "Y": 1, "Z": 2}.get(str(model.up_axis).upper(), 2)
            )
            self.camera.up_axis = axis_idx
            if axis_idx == 0:
                self.camera.pos = PyVec3(2.0, 0.0, 10.0)
            elif axis_idx == 2:
                self.camera.pos = PyVec3(10.0, 0.0, 2.0)
            else:
                self.camera.pos = PyVec3(0.0, 2.0, 10.0)

        self.picking = Picking(model, world_offsets=self.world_offsets)
        self.wind = Wind(model)

        if model is not None:
            try:
                from ..geometry import raycast as _raycast_module  # noqa: PLC0415

                wp.load_module(module=_raycast_module, device=model.device)
                wp.load_module(module="newton._src.viewer.kernels", device=model.device)
            except Exception:
                pass

    @override
    def set_world_offsets(self, spacing):
        super().set_world_offsets(spacing)
        if self.picking is not None:
            self.picking.world_offsets = self.world_offsets

    @override
    def set_camera(self, pos, pitch: float, yaw: float):
        from pyglet.math import Vec3 as PyVec3

        if hasattr(pos, "__iter__"):
            self.camera.pos = PyVec3(float(pos[0]), float(pos[1]), float(pos[2]))
        self.camera.pitch = pitch
        self.camera.yaw = yaw
        self._camera_dirty = True

    def _ensure_picking_line_primitive(self):
        if self._phase != self._PHASE_BUILD or self._PICKING_LINE_NAME in self._instance_prim_paths:
            return

        if Gf is None or UsdGeom is None:
            return

        path = self._get_path(self._PICKING_LINE_NAME)
        self._ensure_scopes_for_path(self.stage, path)

        xform = UsdGeom.Xform.Define(self.stage, path)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        capsule = UsdGeom.Capsule.Define(self.stage, xform.GetPath().AppendChild("capsule"))
        capsule.GetAxisAttr().Set(UsdGeom.Tokens.z)
        capsule.GetRadiusAttr().Set(self._PICKING_LINE_RADIUS)
        capsule.GetHeightAttr().Set(1.0)
        capsule.GetDisplayColorAttr().Set([Gf.Vec3f(*self._PICKING_LINE_COLOR)])

        # Use an emissive material so the picking line stays visibly cyan even under scene shadows.
        from pxr import Sdf, UsdShade

        mat_path = "/root/Materials/mat_picking_line"
        self._ensure_scopes_for_path(self.stage, mat_path)
        material = UsdShade.Material.Define(self.stage, mat_path)
        surface = UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self._PICKING_LINE_COLOR))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim())
        UsdShade.MaterialBindingAPI(capsule).Bind(material)

        self._instance_prim_paths[self._PICKING_LINE_NAME] = [path]

    def _ensure_point_sphere_prototype(self):
        """Create a unit sphere prim for instancing point clouds (sand, particles) in RTX."""
        if self._phase != self._PHASE_BUILD or Gf is None or UsdGeom is None:
            return
        if self.stage.GetPrimAtPath(self._POINT_SPHERE_PROTOTYPE_PATH).IsValid():
            return
        self._ensure_scopes_for_path(self.stage, self._POINT_SPHERE_PROTOTYPE_PATH)
        sphere = UsdGeom.Sphere.Define(self.stage, self._POINT_SPHERE_PROTOTYPE_PATH)
        sphere.GetRadiusAttr().Set(1.0)
        # Hide prototype so only instanced grains are visible (otherwise a big sphere appears at origin)
        UsdGeom.Imageable(sphere.GetPrim()).GetVisibilityAttr().Set("invisible")

    @staticmethod
    def _build_line_instance_buffers(
        starts_np: np.ndarray, ends_np: np.ndarray, capacity: int
    ) -> tuple[np.ndarray, np.ndarray]:
        xforms_np = np.zeros((capacity, 7), dtype=np.float32)
        xforms_np[:, 6] = 1.0
        scales_np = np.zeros((capacity, 3), dtype=np.float32)

        if capacity <= 0 or Gf is None:
            return xforms_np, scales_np

        count = min(capacity, len(starts_np), len(ends_np))
        for i in range(count):
            pos0 = np.asarray(starts_np[i], dtype=np.float32)
            pos1 = np.asarray(ends_np[i], dtype=np.float32)
            delta = pos1 - pos0
            height = float(np.linalg.norm(delta))

            xforms_np[i, :3] = 0.5 * (pos0 + pos1)
            if height <= 1.0e-8:
                continue

            direction = delta / height
            rot = Gf.Rotation()
            rot.SetRotateInto(
                Gf.Vec3d(0.0, 0.0, 1.0),
                Gf.Vec3d(float(direction[0]), float(direction[1]), float(direction[2])),
            )
            quat = rot.GetQuat()
            imag = quat.GetImaginary()
            xforms_np[i, 3] = float(imag[0])
            xforms_np[i, 4] = float(imag[1])
            xforms_np[i, 5] = float(imag[2])
            xforms_np[i, 6] = float(quat.GetReal())
            scales_np[i] = (1.0, 1.0, height)

        return xforms_np, scales_np

    def _queue_picking_line_transform(self, starts_np: np.ndarray, ends_np: np.ndarray):
        xforms_np, scales_np = self._build_line_instance_buffers(starts_np, ends_np, capacity=1)
        self._pending_xforms[self._PICKING_LINE_NAME] = (
            wp.array(xforms_np, dtype=wp.transform, device=self.device),
            wp.array(scales_np, dtype=wp.vec3, device=self.device),
        )

    def _hide_picking_line(self):
        self._queue_picking_line_transform(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )

    @override
    def is_key_down(self, key: str | int) -> bool:
        try:
            import pyglet
        except Exception:
            return False

        if isinstance(key, str):
            key = key.lower()
            if len(key) == 1 and key.isalpha():
                key_code = getattr(pyglet.window.key, key.upper(), None)
            elif len(key) == 1 and key.isdigit():
                key_code = getattr(pyglet.window.key, f"_{key}", None)
            else:
                special_keys = {
                    "space": pyglet.window.key.SPACE,
                    "escape": pyglet.window.key.ESCAPE,
                    "esc": pyglet.window.key.ESCAPE,
                    "enter": pyglet.window.key.ENTER,
                    "return": pyglet.window.key.ENTER,
                    "tab": pyglet.window.key.TAB,
                    "shift": pyglet.window.key.LSHIFT,
                    "ctrl": pyglet.window.key.LCTRL,
                    "alt": pyglet.window.key.LALT,
                    "up": pyglet.window.key.UP,
                    "down": pyglet.window.key.DOWN,
                    "left": pyglet.window.key.LEFT,
                    "right": pyglet.window.key.RIGHT,
                    "backspace": pyglet.window.key.BACKSPACE,
                    "delete": pyglet.window.key.DELETE,
                }
                key_code = special_keys.get(key, None)
            if key_code is None:
                return False
        else:
            key_code = key

        if key_code in self._keys_down:
            return True
        # Pyglet on some platforms (e.g. X11) sends uppercase key symbols (73 for I) while examples use lowercase (105). Treat A-Z and a-z as the same key.
        if isinstance(key_code, int) and 65 <= key_code <= 90:
            return (key_code + 32) in self._keys_down  # check lowercase
        if isinstance(key_code, int) and 97 <= key_code <= 122:
            return (key_code - 32) in self._keys_down  # check uppercase
        return False

    @override
    def log_gizmo(self, name, transform):
        self._gizmo_log[name] = transform

    @override
    def log_state(self, state):
        self._last_state = state
        super().log_state(state)
        self._ensure_picking_line_primitive()
        self._render_picking_line(state)

    def _render_picking_line(self, state):
        if not self.picking_enabled or self.picking is None or not self.picking.is_picking():
            if self._phase == self._PHASE_RENDER:
                self._hide_picking_line()
            return

        pick_body_idx = self.picking.pick_body.numpy()[0]
        if pick_body_idx < 0:
            if self._phase == self._PHASE_RENDER:
                self._hide_picking_line()
            return

        pick_state = self.picking.pick_state.numpy()
        pick_target = pick_state[0]["picking_target_world"]
        picked_point = pick_state[0]["picked_point_world"]

        body_world = self.model.body_world if self.model is not None else None
        if self.world_offsets is not None and self.world_offsets.shape[0] > 0:
            if body_world is not None:
                body_world_idx = body_world.numpy()[pick_body_idx]
                if body_world_idx >= 0 and body_world_idx < self.world_offsets.shape[0]:
                    world_offset = self.world_offsets.numpy()[body_world_idx]
                    pick_target = pick_target + world_offset
                    picked_point = picked_point + world_offset

        self._queue_picking_line_transform(
            np.asarray([[picked_point[0], picked_point[1], picked_point[2]]], dtype=np.float32),
            np.asarray([[pick_target[0], pick_target[1], pick_target[2]]], dtype=np.float32),
        )

    @override
    def apply_forces(self, state):
        if self.picking_enabled and self.picking is not None:
            self.picking._apply_picking_force(state)

        if self.wind is not None:
            self.wind._apply_wind_force(state)

    @override
    def begin_frame(self, time):
        with wp.ScopedTimer("ViewerRTX::begin_frame", active=PROFILE_ENABLED, use_nvtx=True):
            super().begin_frame(time)
            self._pending_xforms.clear()
            # Do not clear _pending_mesh_points here: sand/particles are logged before
            # log_state() in some examples (e.g. inchworm_crawling_sand), so we clear
            # after _update_ovrtx_mesh_points() in end_frame() to include them.
            self._gizmo_log = {}

            if self._window and not self._headless:
                try:
                    self._window.switch_to()
                    # Request focus for the first few frames so keyboard works (X11/Docker often doesn't focus the window).
                    if getattr(self, "_focus_activate_frames_left", 0) > 0 and hasattr(self._window, "activate"):
                        try:
                            self._window.activate()
                        except Exception:
                            pass
                        self._focus_activate_frames_left -= 1
                    self._window.dispatch_events()
                except Exception:
                    pass

            now = perf_counter()
            if self._last_perf_time is not None:
                dt = min(now - self._last_perf_time, 0.1)
                if self.gui:
                    self.gui.update_camera_from_keys(dt, lambda k: k in self._keys_down)
                if self.wind is not None:
                    self.wind.update(dt)
            self._last_perf_time = now

    @override
    def end_frame(self):
        if self._phase == self._PHASE_BUILD:
            self._init_ovrtx()
        elif self._phase == self._PHASE_RENDER:
            with wp.ScopedTimer("ViewerRTX::end_frame", active=PROFILE_ENABLED, use_nvtx=True):
                self._update_ovrtx_camera()
                self._update_ovrtx_transforms()
                self._update_ovrtx_mesh_points()
                self._pending_mesh_points.clear()
                self._pending_mesh_widths.clear()
                self._render_and_display()

    @override
    def log_mesh(
        self, name, points, indices, normals=None, uvs=None, texture=None, hidden=False, backface_culling=True
    ):
        if self._phase == self._PHASE_BUILD:
            super().log_mesh(name, points, indices, normals, uvs, texture, hidden, backface_culling)
            self._mesh_prim_paths[name] = self._get_path(name)
        elif name in self._mesh_prim_paths:
            pts = (
                points.numpy().astype(np.float32)
                if isinstance(points, wp.array)
                else np.asarray(points, dtype=np.float32)
            )
            self._pending_mesh_points[name] = pts

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        if self._phase == self._PHASE_BUILD:
            super().log_instances(name, mesh, xforms, scales, colors, materials, hidden)
            if xforms is not None:
                count = len(xforms)
                paths = [self._get_path(name) + f"/instance_{i}" for i in range(count)]
                self._instance_prim_paths[name] = paths
        else:
            if xforms is not None:
                if scales is None:
                    scales = wp.ones(len(xforms), dtype=wp.vec3)
                self._pending_xforms[name] = (xforms, scales)

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        # RTX: use per-line capsule prims + transform binding (like OpenGL) so springs/lines
        # update every frame. We do not use the USD PointInstancer here because the fabric
        # does not get PointInstancer array updates after the scene is loaded.
        if self._phase == self._PHASE_BUILD:
            if (
                starts is not None
                and ends is not None
                and len(starts) > 0
                and name not in self._instance_prim_paths
                and Gf is not None
                and UsdGeom is not None
            ):
                n_lines = len(starts)
                base_path = self._get_path(name)
                self._ensure_scopes_for_path(self.stage, base_path)
                paths = []
                for i in range(n_lines):
                    inst_path = f"{base_path}/instance_{i}"
                    xform = UsdGeom.Xform.Define(self.stage, inst_path)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
                    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
                    cap = UsdGeom.Capsule.Define(self.stage, f"{inst_path}/capsule")
                    cap.GetAxisAttr().Set(UsdGeom.Tokens.z)
                    cap.GetRadiusAttr().Set(float(width))
                    cap.GetHeightAttr().Set(1.0)
                    paths.append(inst_path)
                self._instance_prim_paths[name] = paths
            return
        # RENDER: update line instance transforms from current starts/ends
        if name not in self._instance_prim_paths:
            return
        paths = self._instance_prim_paths[name]
        n_slots = len(paths)
        if hidden or starts is None or ends is None or len(starts) == 0:
            xforms_np = np.zeros((n_slots, 7), dtype=np.float32)
            xforms_np[:, 6] = 1.0
            scales_np = np.zeros((n_slots, 3), dtype=np.float32)
        else:
            starts_np = starts.numpy() if hasattr(starts, "numpy") else np.asarray(starts, dtype=np.float32)
            ends_np = ends.numpy() if hasattr(ends, "numpy") else np.asarray(ends, dtype=np.float32)
            n_actual = min(n_slots, len(starts_np), len(ends_np))
            xforms_np, scales_np = self._build_line_instance_buffers(starts_np, ends_np, capacity=n_slots)
            if n_actual < n_slots:
                xforms_np[n_actual:] = 0.0
                xforms_np[n_actual:, 6] = 1.0
                scales_np[n_actual:] = 0.0
        self._pending_xforms[name] = (
            wp.array(xforms_np, dtype=wp.transform, device=self.device),
            wp.array(scales_np, dtype=wp.vec3, device=self.device),
        )

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        # RTX: render point clouds as instanced spheres (like OpenGL MeshInstancerGL), since
        # UsdGeom.Points are not reliably rendered as spheres by OVRTX.
        if self._phase == self._PHASE_BUILD:
            if (
                points is not None
                and len(points) > 0
                and name not in self._instance_prim_paths
                and Gf is not None
                and UsdGeom is not None
            ):
                self._ensure_point_sphere_prototype()
                n_pts = len(points)
                base_path = self._get_path(name)
                self._ensure_scopes_for_path(self.stage, base_path)
                paths = []
                for i in range(n_pts):
                    inst_path = f"{base_path}/instance_{i}"
                    prim = self.stage.DefinePrim(inst_path)
                    prim.GetReferences().AddInternalReference(self._POINT_SPHERE_PROTOTYPE_PATH)
                    xform = UsdGeom.Xform(prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
                    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
                    paths.append(inst_path)
                self._instance_prim_paths[name] = paths
            return
        # RENDER: update sphere instance transforms from points and radii
        if name not in self._instance_prim_paths:
            return
        paths = self._instance_prim_paths[name]
        n_slots = len(paths)
        if hidden or points is None or len(points) == 0:
            xforms_np = np.zeros((n_slots, 7), dtype=np.float32)
            xforms_np[:, 6] = 1.0
            scales_np = np.zeros((n_slots, 3), dtype=np.float32)
        else:
            pts = (
                points.numpy().astype(np.float32)
                if hasattr(points, "numpy")
                else np.asarray(points, dtype=np.float32)
            )
            n_actual = min(n_slots, len(pts))
            if np.isscalar(radii):
                r_val = float(radii)
                radii_np = np.full(n_actual, r_val, dtype=np.float32)
            else:
                r = radii.numpy() if hasattr(radii, "numpy") else np.asarray(radii, dtype=np.float32)
                radii_np = np.asarray(r, dtype=np.float32).ravel()[:n_actual]
                if radii_np.size < n_actual:
                    radii_np = np.broadcast_to(radii_np, (n_actual,)).copy()
            xforms_np = np.zeros((n_slots, 7), dtype=np.float32)
            xforms_np[:, 6] = 1.0
            xforms_np[:n_actual, :3] = pts[:n_actual]
            scales_np = np.zeros((n_slots, 3), dtype=np.float32)
            r_display = radii_np * self._POINT_SPHERE_DISPLAY_SCALE
            scales_np[:n_actual, 0] = r_display
            scales_np[:n_actual, 1] = r_display
            scales_np[:n_actual, 2] = r_display
            if n_actual < n_slots:
                scales_np[n_actual:] = 0.0
        self._pending_xforms[name] = (
            wp.array(xforms_np, dtype=wp.transform, device=self.device),
            wp.array(scales_np, dtype=wp.vec3, device=self.device),
        )

    # --------------------------------------------------------- OVRTX updates

    def _update_ovrtx_camera(self):
        if self._rtx is None or not self._camera_dirty:
            return
        with wp.ScopedTimer("ViewerRTX::update_camera", active=PROFILE_ENABLED, use_nvtx=True):
            import ovrtx.math

            mat = self._compute_camera_matrix()
            cam_mat = ovrtx.math.Matrix4d()
            for i in range(4):
                for j in range(4):
                    cam_mat[i][j] = mat[i, j]

            from ovrtx import Semantic

            self._rtx.write_attribute(
                prim_paths=[self._camera_prim_path],
                attribute_name="omni:xform",
                tensor=cam_mat.to_dltensor(),
                semantic=Semantic.XFORM_MAT4x4,
            )
            self._camera_dirty = False

    def _update_ovrtx_transforms(self):
        if not self._transform_binding or not self._pending_xforms:
            return
        with wp.ScopedTimer("ViewerRTX::update_transforms", active=PROFILE_ENABLED, use_nvtx=True):
            from ovrtx import Device

            with self._transform_binding.map(device=Device.CUDA) as mapping:
                matrices = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)  # (N, 4, 4) float64

                offset = 0
                for name, paths in self._instance_prim_paths.items():
                    count = len(paths)
                    if name in self._pending_xforms:
                        xf, sc = self._pending_xforms[name]
                        n = min(count, len(xf))
                        wp.launch(write_transforms, dim=n, inputs=[xf, sc, offset, matrices], device=matrices.device)
                    offset += count

                mapping.unmap(stream=matrices.device.stream.cuda_stream)

    @staticmethod
    def _make_point3f_dltensor(points_np):
        """Create a DLTensor with float3 (lanes=3) dtype from an (N,3) float32 array.

        OVRTX Fabric stores 'points' as point3f[] where each element is 12 bytes
        (float32 x 3 lanes). A plain DLTensor.from_dlpack on a (N,3) float32 array
        produces scalar float32 elements (4 bytes), causing an element-size mismatch.
        """
        from ovrtx._src.dlpack import DLTensor

        flat = np.ascontiguousarray(points_np, dtype=np.float32).reshape(-1)
        dl = DLTensor.from_dlpack(flat)
        n = len(flat) // 3
        dl.dtype.lanes = 3
        dl.ndim = 1
        shape_arr = (ctypes.c_int64 * 1)(n)
        dl.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
        dl._point3f_shape = shape_arr  # prevent GC
        return dl

    def _update_ovrtx_mesh_points(self):
        if self._rtx is None or not self._pending_mesh_points:
            return
        with wp.ScopedTimer("ViewerRTX::update_mesh_points", active=PROFILE_ENABLED, use_nvtx=True):
            from ovrtx._src.dlpack import DLTensor

            for mesh_name, points_np in self._pending_mesh_points.items():
                prim_path = self._mesh_prim_paths.get(mesh_name)
                if prim_path is None:
                    continue
                dl = self._make_point3f_dltensor(points_np)
                self._rtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="points",
                    tensors=[dl],
                )
                # Write widths so UsdGeom.Points render as spheres (sand, particles)
                widths_np = self._pending_mesh_widths.get(mesh_name)
                if widths_np is not None and widths_np.size >= len(points_np):
                    flat = np.ascontiguousarray(widths_np[: len(points_np)], dtype=np.float32).reshape(-1)
                    dl_w = DLTensor.from_dlpack(flat)
                    self._rtx.write_array_attribute(
                        prim_paths=[prim_path],
                        attribute_name="widths",
                        tensors=[dl_w],
                    )

    @staticmethod
    def _xform_to_mat44(pos, quat, scale):
        """Convert (pos, quaternion_xyzw, scale) to 4x4 row-major matrix (float64).

        Matches the USD GfMatrix4d / OVRTX fabric convention where the
        translation lives in the last row and basis vectors in the first three.
        """
        x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        mat = np.empty((4, 4), dtype=np.float64)
        mat[0, 0] = (1.0 - 2.0 * (yy + zz)) * sx
        mat[0, 1] = (2.0 * (xy + wz)) * sx
        mat[0, 2] = (2.0 * (xz - wy)) * sx
        mat[0, 3] = 0.0
        mat[1, 0] = (2.0 * (xy - wz)) * sy
        mat[1, 1] = (1.0 - 2.0 * (xx + zz)) * sy
        mat[1, 2] = (2.0 * (yz + wx)) * sy
        mat[1, 3] = 0.0
        mat[2, 0] = (2.0 * (xz + wy)) * sz
        mat[2, 1] = (2.0 * (yz - wx)) * sz
        mat[2, 2] = (1.0 - 2.0 * (xx + yy)) * sz
        mat[2, 3] = 0.0
        mat[3, 0] = float(pos[0])
        mat[3, 1] = float(pos[1])
        mat[3, 2] = float(pos[2])
        mat[3, 3] = 1.0
        return mat

    # ------------------------------------------------------- render + display

    def _update_fps(self):
        current_time = perf_counter()
        self._fps_frame_count += 1
        if current_time - self._last_fps_time >= 1.0:
            time_delta = current_time - self._last_fps_time
            self._current_fps = self._fps_frame_count / time_delta
            self._fps_history.append(self._current_fps)
            if len(self._fps_history) > 60:
                self._fps_history.pop(0)
            self._last_fps_time = current_time
            self._fps_frame_count = 0

    def _render_and_display(self):
        if self._rtx is None or self._should_close:
            return
        with wp.ScopedTimer("ViewerRTX::render_and_display", active=PROFILE_ENABLED, use_nvtx=True):
            from ovrtx import Device

            # wait for async rendering to complete
            with wp.ScopedTimer("ViewerRTX::rtx_wait", active=PROFILE_ENABLED, use_nvtx=True):
                products = self._render_result.wait() if self._render_result is not None else None

            if products is not None:
                for _pname, product in products.items():
                    for frame in product.frames:
                        if "LdrColor" in frame.render_vars:
                            with wp.ScopedTimer("ViewerRTX::fb_map", active=PROFILE_ENABLED, use_nvtx=True):
                                with frame.render_vars["LdrColor"].map(device=Device.CPU) as mapping:
                                    pixels = np.from_dlpack(mapping.tensor)
                                    with wp.ScopedTimer(
                                        "ViewerRTX::blit_to_window", active=PROFILE_ENABLED, use_nvtx=True
                                    ):
                                        self._blit_to_window(pixels)
                            if self._rtx_first_frame_start is not None:
                                elapsed = perf_counter() - self._rtx_first_frame_start
                                self._rtx_first_frame_start = None
                                self._rtx_first_frame_done.set()
                                if self._rtx_spinner_thread is not None:
                                    self._rtx_spinner_thread.join(timeout=0.5)
                                print(f"\rFirst RTX frame ready in {elapsed:.1f} s")
                                if not self._headless:
                                    print("Tip: If keys (I/K, Space, etc.) don't work, click the viewer window to give it focus.")

            # kick off next async rendering frame
            with wp.ScopedTimer("ViewerRTX::rtx_step", active=PROFILE_ENABLED, use_nvtx=True):
                self._render_result = self._rtx.step_async(
                    render_products={self._render_product_path},
                    delta_time=1.0 / self.fps,
                )

    def _blit_to_window(self, pixels: np.ndarray):
        """Upload *pixels* to a GL texture and draw a fullscreen triangle (GPU sRGB + flip)."""
        import ctypes  # noqa: PLC0415

        from pyglet import gl

        if self._window is None or self._window.context is None:
            return

        h, w = pixels.shape[:2]
        if not pixels.flags["C_CONTIGUOUS"]:
            pixels = np.ascontiguousarray(pixels)

        self._window.switch_to()
        fb_w, fb_h = self._window.get_framebuffer_size()
        gl.glViewport(0, 0, fb_w, fb_h)

        with wp.ScopedTimer("ViewerRTX::gl_tex_upload", active=PROFILE_ENABLED, use_nvtx=True):
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            )

        with wp.ScopedTimer("ViewerRTX::gl_draw", active=PROFILE_ENABLED, use_nvtx=True):
            gl.glUseProgram(self._gl_program)
            gl.glBindVertexArray(self._gl_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if self.gui:
            with wp.ScopedTimer("ViewerRTX::gui_render", active=PROFILE_ENABLED, use_nvtx=True):
                self.gui.render_frame(update_fps=True)

        with wp.ScopedTimer("ViewerRTX::swap_buffers", active=PROFILE_ENABLED, use_nvtx=True):
            self._window.flip()

    def save_screenshot(self, path: str) -> None:
        """Save the last rendered frame to a JPEG file.

        Call this after at least one frame has been rendered (e.g. after the
        simulation loop). Works in headless mode.
        """
        if self._last_frame_pixels is None:
            return
        from PIL import Image

        img = self._last_frame_pixels
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # drop alpha for JPEG
        Image.fromarray(img).save(path, "JPEG", quality=92)

    # ----------------------------------------------------------- viewer API

    def register_ui_callback(self, callback, position="side"):
        if position not in self._ui_callbacks:
            valid = list(self._ui_callbacks.keys())
            raise ValueError(f"Invalid position {position!r}. Valid: {valid}")
        self._ui_callbacks[position].append(callback)

    def clear_model(self):
        self._ui_callbacks["side"] = []
        self._ui_callbacks["free"] = []
        self.picking = None
        self.wind = None
        self._last_state = None
        self._last_control = None
        super().clear_model()

    @override
    def is_paused(self) -> bool:
        return self._paused

    @override
    def is_running(self) -> bool:
        if self._should_close:
            return False
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    @override
    def close(self):
        if self._transform_binding is not None:
            self._transform_binding.unbind()
            self._transform_binding = None

        self._rtx = None
        if self.ui:
            self.ui.shutdown()

        if self._window is not None:
            if not self._headless:
                try:
                    self._pyglet_app.event_loop.dispatch_event("on_exit")
                    self._pyglet_app.platform_event_loop.stop()
                except Exception:
                    pass
            self._window.close()
            self._window = None
