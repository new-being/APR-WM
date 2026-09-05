# TASK-X0 Report — `put_object_cabinet`

Date: 2026-08-27  
Host: AutoDL (`/root/RoboTwin`, NVIDIA Vulkan ICD)  
Status: **RUN COMPLETE**; **`task_x0_passed=false`**; pattern **`coverage_hole`**  
Prereg: `REPORT/REG/TASKX/TASKX0_PREREG.md` (family `TASKX_PREREG.md`)  
Artifacts: `runs/task_x0/put_object_cabinet/{summary.json,run.log,nohup.out}`  
Does not: TASK-X1 / diffusion train; R10; RoboTwin-X0 `physics_predict`; RGB in \(s\); X0R random-wrench; \(p\to F_{\mathrm{physics}}\)

## One-line result

Oracle \(p_t=G_{\mathrm{frozen}}(s_{\le t},g)\) was run on official RoboTwin `demo_clean` qpos/endpose (50 hdf5). CuRobo is missing; native `play_once` does not plan. Subsampled qpos replay recovered **1/50** `check_success`. G0/G1/G2 fail. G3 causal (`future_leak=false`). **Not** an X1 unlock.

**Honest class:** data/execution chain — **instrument insufficient** (`coverage_hole`). This does **not** support “\(p_t\) has no decision value.”

## Command

```text
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
PYTHONPATH=/root/APR-WM \
  /root/miniconda3/envs/Robotwin/bin/python -m aprwm_v0 task-x0 \
  --task put_object_cabinet \
  --output /root/APR-WM/runs/task_x0/put_object_cabinet \
  --robotwin-repo /root/RoboTwin
```

Long collect: `nohup` (PID logged in `nohup.out`). Header freeze before parse: \(N_{\mathrm{demo}}=50\), \(N_{\min}=32\), \(\varepsilon_s=0.05\times\mathrm{RMS}(s)\).

## Frozen instrument (not retuned)

| item | value |
|---|---|
| task | `put_object_cabinet` |
| \(k\) | approach, grasp, transport, insert, release, done |
| \(c\) | object_grasped, object_lifted, near_cabinet, inside_target, released |
| \(s\) | oracle robot qpos + EE + object pose + cabinet \(q,\dot q\) (no RGB) |
| \(A\) | \(\Delta\) qpos (action, not \(F_{\mathrm{physics}}\)) |
| data | official `demo_clean` hdf5 qpos/endpose + seed.txt replay |

## Gate table

| id | requirement | result |
|---|---|---|
| **G0 coverage** | each \(k\): \(N_k\ge 32\) | **FAIL** — counts: approach 251, grasp 0, transport 0, insert 3, release 0, done 10 |
| **G1 non-redundancy** | \(\|s_i-s_j\|<\varepsilon_s\) but \(p_i\neq p_j\), alias action gap \(>\) same-\(p\) NN | **FAIL** — `n_alias_pairs=0`; \(\varepsilon_s=0.0387\) |
| **G2 entropy** | \(\mathrm{NLL}(A\mid s,g,p)<\mathrm{NLL}(A\mid s,g)\); \(\Delta\mathrm{NLL}\) 95% CI not crossing 0 | **FAIL** — \(\Delta\mathrm{NLL}=-0.518\), CI \([-1.532,\;\approx 0]\) |
| **G3 recoverability** | \(p_t=G(s_{\le t},g)\) only; leak not required; `future_leak=false` | **PASS** — `future_leak=false`, `leak_needed_for_coverage=false` |
| **G-label** | `runs/task_x0/`; no \(S_{\mathrm{task}}\); `oracle_progress=true` | **PASS** |

**`task_x0_passed`:** false  
**pattern:** `coverage_hole`  
**`S_task` claimed:** false  
**Unlocks TASK-X1 / R10:** false

## Paths

| item | path |
|---|---|
| runner | `aprwm_v0/task_x0.py` |
| CLI | `python -m aprwm_v0 task-x0 --task … --output …` |
| tests | `tests/test_task_x0.py` (`ALL_TASK_X0_UNIT_OK`) |
| impl lock | `runs/task_x0/.impl_lock` |
| output | `/root/APR-WM/runs/task_x0/put_object_cabinet/` |
| official hdf5 | `/root/RoboTwin/data/demo_clean/put_object_cabinet/aloha_agilex/data/*.hdf5` (50) |

## Demo / CuRobo blockers

1. **CuRobo missing** (`ModuleNotFoundError: curobo`). Embodiment `planner: curobo`. Runner uses `mplib_screw` adapter. **pytorch3d** also missing (RoboTwin prints `missing pytorch3d`).
2. **Native `play_once`:** `setup_demo` boots; first `grasp_actor` raises `AssertionError: target_pose cannot be None for move action` (grasp pose / mplib check). `n_steps_raw=0`. Not a successful official collector pass.
3. **Official hdf5** stores `state/` and `action/` **qpos + endpose only** (plus RGB, unused). **No object pose, no cabinet \(q\)**. Oracle \(c_t\) needs those; they are recovered only if qpos replay physically grasps in SAPIEN.
4. **Qpos replay:** 50 hdf5 × matching `seed.txt`. Scene RNG often disagrees with recorded `{A}` (e.g. coffee-box vs rubikscube on seed 0). Missing `model_data` for some objects (`077_phone`, `081_playingcards`). **`check_success` true on 1/50** (episode 7, seed 73). That episode still lacks grasp/transport coverage under frozen \(G\).
5. Subsampled hdf5 (~15 env steps / frame) does not reproduce contact when drive-targets are replayed.

## Claims ceiling (honored)

X0 may say: demos exist; G is causal; coverage is a hole; \(p\) did not drop held-out action NLL on the one recovered episode. **Not** planning success. **Not** TASK-X1.
