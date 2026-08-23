# RoboTwin-X0 Official Smoke Report — `put_object_cabinet`

Date: 2026-08-23  
Status: **FAIL** (`rtwx_x0_smoke_passed=false`)  
Prereg: `REPORT/REG/RTWX/RTWX0_SMOKE_PREREG.md`  
Artifacts: `runs/rtwx_x0/smoke/summary.json`  
Does not: capacity \(R_P\); RGB in \(s\); play_once / CuRobo planning

## Result

| gate | result |
|---|---|
| G-assets | **PASS** (`objects/` 125 dirs, `embodiments/` 5 robots) |
| G-cabinet | **FAIL** |
| G-norgb / G-state / G-step | not reached |

`setup_demo` dies in `Base_Task.setup_scene`:

```text
RuntimeError: failed to find a rendering device
```

at `sapien.SapienRenderer()`. Official RoboTwin always constructs a Vulkan
renderer and cameras, even when \(s\) is oracle state and `render_freq=0`.

P0 (`rtwx_drawer1.v1`) used `PhysxCpuSystem` only and did not need a
renderer. That gap is real: **assets are in, official task boot is not**.

## Environment note

- GPU visible: RTX 5070 Laptop (`nvidia-smi`).
- CUDA libs present under `/usr/lib/wsl/lib`.
- No NVIDIA Vulkan ICD (`nvidia_icd.json` / `libGLX_nvidia`) on this WSL.
- Mesa lavapipe ICD loads but SAPIEN then fails with
  `vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent`.

Smoke also stubs `CuroboPlanner` (CuRobo not compiled). That is allowed
by the prereg because `play_once` is out of scope; it is **not** the
failing gate.

## Unlock

Formal 3-task RoboTwin-X0 stays **locked** until a later smoke gets
`G-cabinet`. Next hardware/driver step is a working NVIDIA Vulkan ICD in
this WSL, then re-run:

```text
PYTHONPATH=/home/dong/Projects/APR-WM \
  /home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 rtwx-x0-smoke \
  --output /home/dong/Projects/APR-WM/runs/rtwx_x0/smoke
```
