# TASK-X0 Cup Report — `place_empty_cup`

Date: 2026-08-27  
Status: **RAN**; **CUP `task_x0_passed=false`** (`pattern=p_no_nll`)  
Prereg: `REPORT/REG/TASKX/TASKX0_PREREG.md`  
Artifacts: `runs/task_x0/place_empty_cup/{summary.json,run.log,nohup.out}`  
Does not: diffusion / TASK-X1; RGB in \(s\); RoboTwin-X0 `physics_predict`; unlock R10; \(S_{\mathrm{task}}\) claim

## One-line result

On **50** official RoboTwin hdf5 qpos–endpose demos (replayed into the simulator), oracle \(G_{\mathrm{frozen}}(s_{\le t},g)\) **covers all cup phases (G0)**, shows **\(s\)-aliasing (G1)**, and is **causal (G3)**. Held-out kNN NLL **does not drop** when \(p\) is added (**G2 fail**; \(\Delta\mathrm{NLL}<0\), 95% CI crosses 0). Native `play_once` success is **unavailable** (CuRobo missing); replay `check_success=0/50`. Not faked.

**Honest class:** genuine negative of *this* probe — \(p_t\) did not further compress action uncertainty (`p_no_nll`). Not an instrument-hole verdict; not “task progress has no planning value.”

## Gate table (cup only)

| id | result | detail |
|---|---|---|
| **G0** coverage | **PASS** | \(N_k\ge 32\): approach 3957, grasp 1168, lift 1158, transport 1553, place 356, release 184, done 241 |
| **G1** non-redundancy | **PASS** | \(\varepsilon_s=0.05\times\mathrm{RMS}(s)=0.0312\); 27 alias pairs; median \(\|A\|\) alias 0.00791 \(>\) same-\(p\) NN 0.00693 |
| **G2** entropy | **FAIL** | \(\mathrm{NLL}(A\mid s,g,p)=-52.02 > \mathrm{NLL}(A\mid s,g)=-52.49\); \(\Delta\mathrm{NLL}=-0.471\); CI \([-1.004, 0.267]\) crosses 0 |
| **G3** recoverability | **PASS** | `future_leak=false`; leak not required for coverage |
| **G-label** | logged | `oracle_progress=true`; no \(S_{\mathrm{task}}\) |
| **`task_x0_passed` (CUP)** | **false** | G0∧G1∧G3, **not G2** |

Header freeze (before parse): \(N_{\mathrm{demo}}=50\), \(N_{\min}=32\).

## Blockers

- **CuRobo / pytorch3d missing.** Scene boot uses `curobo_stub` + mplib screw fallback. Native `play_once` planning is not a successful demo source.
- **Official hdf5 has qpos/endpose (and RGB, unused), not object pose.** `_traj_data` pickles absent. Subsampled qpos replay **does not re-grasp** the cup (`n_success_episodes=0`). \(G\) is causal on robot EE / gripper / coaster goal plus whatever cup pose the sim shows.
- Do not read this as X1 unlock.

## Command

```text
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
PYTHONPATH=/root/APR-WM \
  /root/miniconda3/envs/Robotwin/bin/python -m aprwm_v0 task-x0 \
  --task place_empty_cup \
  --output /root/APR-WM/runs/task_x0/place_empty_cup \
  --robotwin-repo /root/RoboTwin
```
