# TASK-X0 `stamp_seal` — oracle progress instrument

Date: 2026-08-27
Status: **RAN**; `task_x0_passed=false`; pattern **`p_no_nll`**
Prereg: `REPORT/REG/TASKX/TASKX0_PREREG.md`, `TASKX_PREREG.md`
Artifacts: `runs/task_x0/stamp_seal/{summary.json,run.log,cli.log}`
Does not: TASK-X1; diffusion train; R10; RGB in \(s\); RoboTwin-X0 `physics_predict`;
\(S_{\mathrm{task}}\) claim

## One-line result

Official `demo_clean` demonstrations (\(N_{\mathrm{demo}}=50\), frozen in the runner
header before parse) yield causal oracle progress with **G0 / G1 / G3 pass**.
The kNN probe does **not** give \(\mathrm{NLL}(A\mid s,g,p)<\mathrm{NLL}(A\mid s,g)\)
(paired \(\Delta\mathrm{NLL}\) CI is negative). Stamp TASK-X0 is **not** PASS.

**Honest class:** genuine negative of *this* probe — \(p_t\) did not further
compress action uncertainty (`p_no_nll`; G2 CI entirely negative). G0/G1/G3
PASS; not a coverage-hole claim; not “task progress has no planning value.”

## Command

```text
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
PYTHONPATH=/root/APR-WM \
  /root/miniconda3/envs/Robotwin/bin/python -m aprwm_v0 task-x0 \
  --task stamp_seal \
  --output /root/APR-WM/runs/task_x0/stamp_seal \
  --robotwin-repo /root/RoboTwin
```

## Data (not play_once)

| item | value |
|---|---|
| source | official hdf5 qpos/endpose + sapien init \((s_0,g)\) + causal rigid-attach |
| \(N_{\mathrm{demo}}\) | **50** (declared before parse) |
| frames | 7279 |
| CuRobo | missing (`curobo_stub=true`); **play_once not run** |
| RGB | false |
| future labels | false |

HDF5 has no object pose. \(G\) is not an EE-mean proxy. Init seal/target come from
`setup_demo` on official `seed.txt`. Attach uses current EE, gripper, and seal only
(pre-grasp geometry: pinch from above, \(\Delta z\sim 0.19\,\mathrm{m}\)).

## Gates

| id | requirement | result |
|---|---|---|
| **G0 coverage** | each \(k\): \(N_k\ge 32\) | **PASS** |
| **G1 non-redundancy** | \(\|s_i-s_j\|<\varepsilon_s=0.05\mathrm{RMS}(s)\) alias pairs; median \(\|A\|\) gap \(>\) same-\(p\) NN | **PASS** (10 pairs; 0.059 \(>\) 0.045) |
| **G2 entropy** | \(\Delta\mathrm{NLL}>0\), 95% CI not crossing 0 | **FAIL** (\(\Delta=-0.484\), CI \([-0.679,-0.290]\)) |
| **G3 recoverability** | \(p_t=G(s_{\le t},g)\); leak not required | **PASS** (`future_leak=false`) |
| **G-label** | `runs/task_x0/`; `oracle_progress=true`; no \(S_{\mathrm{task}}\) | **PASS** |

Phase counts: approach 3279, grasp 1610, align 252, press 1638, complete 500.

\(\varepsilon_s=0.0361\) from train \(\mathrm{RMS}(s)=0.721\).

## Pattern / ceiling

`p_no_nll`: \(p\) is a closed causal instrument with coverage and \(s\)-aliasing,
but the frozen kNN probe does not show lower action uncertainty given \(p\).
X0 must **not** claim planning success. TASK-X1 stays **LOCKED** for this task
until family PASS (all three tasks). R10 stays **LOCKED**.

## Blockers (logged, not faked)

- CuRobo / pytorch3d missing; official `play_once` coverage **not** claimed.
- No `_traj_data` pkl replay path.
- G2 failure is a gate result, not a missing-demo artifact.
