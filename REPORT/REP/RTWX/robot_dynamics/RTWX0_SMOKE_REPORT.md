# RoboTwin-X0 Official Smoke Report — `put_object_cabinet`

Date: 2026-08-27
Host: AutoDL (`/root/RoboTwin`, RTX 5090, NVIDIA Vulkan ICD non-empty)
Status: **PASS** (`rtwx_x0_smoke_passed=true`)
Prereg: `REPORT/REG/RTWX/RTWX0_SMOKE_PREREG.md`
Artifacts: `runs/rtwx_x0/smoke/summary.json`
Does not: capacity \(R_P\); RGB in \(s\); play_once / CuRobo planning; unlock
R10; retune gates

## Result (this machine)

| gate | result |
|---|---|
| G-assets | **PASS** (`objects/` + `embodiments/`; `036_cabinet` present) |
| G-cabinet | **PASS** (`setup_demo` returns) |
| G-norgb | **PASS** (keys: `q_left`, `q_right`, `x_o`, `R_o`, `q_cabinet`, `qd_cabinet`) |
| G-state | **PASS** (\(d_s=27\in[20,80]\)) |
| G-step | **PASS** (20 `scene.step()`, finite) |
| G-label | **PASS** (`official_robotwin_task_executed=true`; no \(R_P\)) |

`unlocks_rtwx_x0_formal_runner=true`.

CuRobo is absent (`curobo_stub=true`); gripper interpolation uses the
prereg-allowed no-op planner (linspace `plan_grippers`). `pytorch3d` missing
is logged by RoboTwin camera FPS and is **not** a smoke gate.

## Renderer (previous FAIL, now closed)

Earlier AutoDL FAIL: hollow NVIDIA stack (0-byte ICD / no `/dev/nvidia*`).
This re-run: `nvidia-smi` RTX 5090, `/dev/nvidia*`, ICD 140 bytes, Vulkan
`deviceName=NVIDIA GeForce RTX 5090`. `SapienRenderer()` succeeded.

## Asset hole (instrument, not a gate retune)

`objects.zip` still has empty dirs for some cabinet objects
(`057_toycar`, `073_rubikscube`, `075_bread`, `077_phone`, `081_playingcards`
have collision/visual stubs and **no** `model_data*.json`). Official
`np.random.choice` on seed 0 hit `073_rubikscube`. Smoke retries seeds in
`[0, 32)` on that `ValueError` only. **setup_seed=2**, object `047_mouse`.
Gates unchanged. Formal must skip or repair those empty object dirs; this
does not rewrite G-assets.

## Unlock

Writing the 3-task formal runner is **unlocked**. R10 stays **LOCKED**.
