# RoboTwin-X0 Official Smoke — `put_object_cabinet`

Date: 2026-08-23  
Status: **FROZEN**; **FAIL** (`rtwx_x0_smoke_passed=false`; renderer)  
Depends on: `REPORT/REG/RTWX/RTWX0_PREREG.md`, P0 PASS  
Does not: capacity \(R_P\); 3-task formal matrix; RGB as \(s\); play_once / CuRobo planner

## One question

\[
\boxed{
\text{Does official } \texttt{put\_object\_cabinet} \text{ boot on this machine
with oracle proprio/object/articulation state and no RGB in } s\text{?}
}
\]

## Protocol

- Host: `/home/dong/Projects/RoboTwin`, `demo_clean.yml`, embodiment `aloha-agilex`.
- Call `setup_demo` only (no `play_once`).
- If CuRobo is not installed, inject a no-op `CuroboPlanner` so the
  official scene still boots; log `curobo_stub`. This smoke is **not** a
  planner test.
- Extract \(s\) from real joint qpos, object pose, cabinet \(q,\dot q\).
- Step the SAPIEN scene \(\ge 20\) times.

## Gates

| id | requirement |
|---|---|
| **G-assets** | `assets/objects` and `assets/embodiments` exist |
| **G-cabinet** | `put_object_cabinet.setup_demo` returns |
| **G-norgb** | extracted \(s\) has no camera/RGB keys or image tensors |
| **G-state** | \(s\) includes robot q, object pose, cabinet q; \(20\le d_s\le 80\) |
| **G-step** | 20 `scene.step()` without exception |
| **G-label** | `runs/rtwx_x0/smoke/`; `official_robotwin_task_executed=true`; no \(R_P\) |

`rtwx_x0_smoke_passed` iff all gates.

Smoke **unlocks writing** the 3-task formal runner. It does **not** itself
report a capacity curve.
