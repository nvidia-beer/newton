# Vulkan in Docker (ViewerRTX / OVRTX)

This document describes how Vulkan was enabled inside the Newton Docker container so that `--viewer rtx` (OVRTX / ViewerRTX) works when running examples via `run-examples.sh`.

## Problem

When running Vulkan-based viewers (e.g. `run-examples.sh 1 --viewer rtx`) or `vulkaninfo` inside the container, the Vulkan loader reported:

- `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`
- `vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER`

Vulkan worked on the host but not in the container.

## Root cause

1. **Driver library not in the container**  
   The Vulkan loader needs the NVIDIA driver library (`libGLX_nvidia.so.0`). With only `--gpus all`, Docker exposes the GPU and some driver bits, but the **Vulkan/GLX driver libs are only injected when the container runs with the NVIDIA Container Runtime** (`--runtime=nvidia`), not with the default `runc` runtime.

2. **ICD config**  
   The loader finds drivers via ICD (Installable Client Driver) JSON files under `/usr/share/vulkan/icd.d/`. The container had the Vulkan loader (`vulkan-tools` / `libvulkan1`) but not the host’s ICD config, so it couldn’t discover the NVIDIA driver even when the runtime would inject it.

3. **EGL/GL deps**  
   The NVIDIA Vulkan ICD can depend on EGL/GL libs (e.g. `libegl1`). Those need to be present in the image.

## Fix (summary)

- **Host:** Install and configure the NVIDIA Container Toolkit so Docker has the `nvidia` runtime; use that runtime when running the container.
- **Container:** Install Vulkan loader + EGL; mount the host’s Vulkan ICD directory; run with `--runtime=nvidia` and `--gpus all`.

---

## 1. Host setup (one-time)

### 1.1 Install NVIDIA Container Toolkit

Example on Ubuntu/Debian:

```bash
# Add repo if needed, then:
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Version **1.14.4 or newer** is recommended for reliable Vulkan ICD injection.

### 1.2 Register the nvidia runtime with Docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

### 1.3 Restart Docker

```bash
sudo systemctl restart docker
```

### 1.4 Verify the nvidia runtime is available

```bash
docker info | grep -i runtime
```

You should see `nvidia` in the list of runtimes (e.g. `Runtimes: io.containerd.runc.v2 nvidia runc`).

---

## 2. Changes in this repo

### 2.1 `run-examples.sh`

- When an NVIDIA GPU is detected, the script now passes **`--runtime=nvidia`** in addition to `--gpus all`, so the toolkit injects Vulkan/GLX driver libs.
- The host’s **Vulkan ICD directory** is mounted read-only into the container when present:
  - `-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro`
- Environment variables already set for GPU runs: `NVIDIA_DRIVER_CAPABILITIES=all`, `NVIDIA_VISIBLE_DEVICES=all`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`.

### 2.2 Dockerfiles

- **Dockerfile.x86**
  - Install `vulkan-tools` (provides Vulkan loader and `vulkaninfo`) and `libegl1` in the same `apt-get` step as other viewer deps.
- **Dockerfile.arm64**
  - Install `vulkan-tools` and `libegl1` (alongside existing `libegl-dev` / GLVND) so the NVIDIA Vulkan ICD has the libs it needs.

(If `vulkan-tools` is commented out in a Dockerfile for a minimal image, the Vulkan loader may still come from another package; ensure the image has Vulkan loader + EGL if you want RTX in Docker.)

---

## 3. Verify Vulkan in the container

Run with the same runtime and mounts the run script uses:

```bash
docker run --rm --runtime=nvidia --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
  newton:latest vulkaninfo --summary
```

**Success:** Output lists one or more NVIDIA GPUs under “Devices” (e.g. `deviceName = NVIDIA GB10`) and there is no `ERROR_INCOMPATIBLE_DRIVER`.

**Failure:** If you see `unknown or invalid runtime name: nvidia`, the nvidia runtime is not registered — re-run the host setup (especially `nvidia-ctk runtime configure` and Docker restart). If you still see `Could not get 'vkCreateInstance' ... for ICD libGLX_nvidia.so.0`, the container is not using the nvidia runtime or the toolkit is not injecting the driver libs (check toolkit version and Docker config).

---

## 4. Optional: harmless loader messages

When running `vulkaninfo` in the container you may see “ERROR” lines such as:

- `libvulkan_asahi.so: cannot open shared object file`
- `libvulkan_radeon.so: cannot open shared object file`
- (and similar for other vendors)

These come from **other** ICD JSON files in the mounted `icd.d` (e.g. Mesa/other GPUs). The container does not have those vendor libs, so the loader skips them. Only the NVIDIA ICD is used, and that one loading successfully is what matters. You can ignore these messages.

`'DISPLAY' environment variable not set` and `XDG_RUNTIME_DIR not set` are expected when not attached to a display (e.g. plain `vulkaninfo`). They do not prevent Vulkan or the RTX viewer from working when you run the full example with a display (e.g. X11 forwarded).

---

## 4b. omni.fabric.internal warnings (mesh points)

When using the RTX viewer with deforming meshes you may see repeated:

```text
[Warning] [omni.fabric.internal] Unsupported type encountered during VtValue extraction
[Warning] [omni.fabric.internal] Attempting to set an invalid data source to array attribute
```

These come from Omniverse Fabric when the viewer updates mesh vertex positions (`points`) each frame. The Fabric’s VtValue path doesn’t fully support the DLTensor format we use for point3f arrays, so it logs a warning but often still uses the data. If the scene renders correctly (meshes move as expected), you can ignore these warnings. If rendering is wrong or the viewer crashes, try `--viewer gl` instead. Reducing Omniverse log level (e.g. `--/log/level=error` if your entry point supports it) can hide them.

---

## 5. Multiple ICDs / omni.rtx error

If you see:

```text
[Error] [omni.rtx] Multiple Installable Client Drivers (ICDs) are found for the same GPU ...
```

the Vulkan loader is seeing more than one NVIDIA ICD (e.g. `nvidia_icd.json` in both `/etc/vulkan/icd.d` and `/usr/share/vulkan/icd.d`), so the same GPU is reported twice and OVRTX can crash or misbehave.

**Fix (pick one):**

1. **Use a single ICD (recommended for Docker)**  
   The run script sets `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` when that file exists on the host, so the container uses only that ICD. Ensure the host has `nvidia_icd.json` in `/usr/share/vulkan/icd.d` (and that we mount that dir). If the error still appears, try on the host:
   ```bash
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
   ```
   before running the example (or in the same environment as the container).

2. **Leave only one ICD on the host**  
   Have `nvidia_icd.json` in only one of these:
   - `/etc/vulkan/icd.d` — ICDs from non-distro packages (e.g. NVIDIA `.run` installer)
   - `/usr/share/vulkan/icd.d` — ICDs from distro packages  
   Remove or rename the duplicate (e.g. `sudo mv /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json.bak`) so the loader finds a single NVIDIA driver. Updating the system can re-add the other; then purge distro NVIDIA packages and install the driver via the official `.run` package if you want a clean single-ICD setup.

---

## 6. Caching the first RTX frame (Docker)

The first time the RTX viewer runs, the NVIDIA driver compiles shaders and builds a pipeline cache; that can take tens of seconds. To reuse that work across container runs, the run script enables a **persistent shader cache** when an NVIDIA GPU is detected:

- It creates `NEWTON_DIR/.cache/rtx-shader-cache` on the host (if needed) and mounts it into the container.
- It sets **`__GL_SHADER_DISK_CACHE_PATH`** so the NVIDIA driver writes and reads the cache from that directory.

You are not running the full Omniverse platform—only the OVRTX library for RTX rendering—so this is the only cache we can persist (the driver’s own shader cache). After the first run fills it, later runs of the **same** example can be faster. The cache is keyed by driver version and GPU; if you update the driver or use a different machine, the first run will be slow again until the cache is repopulated. The `.cache/` directory is gitignored.

### Why can the first frame still be slow?

Even with the cache, the first frame often stays slow for one or more of these reasons:

1. **What the cache covers**  
   `__GL_SHADER_DISK_CACHE_PATH` is an **OpenGL** shader cache. The RTX viewer uses **Vulkan** (OVRTX) for the actual ray-tracing. The driver may reuse some internal caches for Vulkan pipelines when this env is set, but the heavy cost is often **Vulkan pipeline compilation and acceleration structure (BLAS/TLAS) build**, which are not fully covered by the OpenGL shader cache.

2. **Per-scene work**  
   Each run builds acceleration structures for the current scene (meshes, instances). That work is geometry-dependent and is done every time you load a scene; it is not cached by a shader cache.

3. **Different examples**  
   Switching to another example means a different scene and possibly new shaders/pipelines. The first frame for that example will again be slow until its pipelines and structures are built.

So the disk cache helps most when you **repeat the same example** on the same machine and driver; it does not remove the cost of the first frame for a **new** example or the one-time cost of building the RT scene. For the fastest possible first frame when testing, use `--viewer gl` (OpenGL viewer), which avoids RT initialization entirely.

### How to make specific scenes load again faster

1. **Use the persistent shader cache (already enabled)**  
   The run script mounts `NEWTON_DIR/.cache/rtx-shader-cache` and sets `__GL_SHADER_DISK_CACHE_PATH`. The first time you run an example, the driver fills that cache. The **second and later runs of the same example** can be faster when the driver reuses compiled shaders from the cache.

2. **Run the same example repeatedly**  
   Cache benefit is per-scene: same example ≈ same pipelines/shaders, so the same cache entries are hit. A different example is a different scene and may need new compilation until its entries are cached.

3. **Vulkan pipeline cache (application-level)**  
   Vulkan allows saving a **pipeline cache** to disk and loading it on the next run; OVRTX does not expose this in its public Python API, so we cannot enable it from here.

4. **Acceleration structures**  
   Ray tracing builds BLAS/TLAS for the current scene each time; that work is not stored in the shader cache, so the first frame still does some per-scene work.

**Summary:** Run the same example at least twice to warm the driver cache; the second run can be faster. Keep the same driver and GPU so the cache stays valid.

---

## 7. Run the RTX viewer

After Vulkan works in the container (step 3), run an example with the RTX viewer:

```bash
./.devcontainer/run-examples.sh 1 --viewer rtx
```

Use your normal display/X11 setup so the viewer window can open, or `--headless` if the example supports it.
