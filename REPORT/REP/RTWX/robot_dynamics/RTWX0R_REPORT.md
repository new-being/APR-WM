# RoboTwin-X0R Report — Cabinet Excitation & Dynamics Interface

Date: 2026-08-27
Host: AutoDL (`/root/RoboTwin`, NVIDIA Vulkan ICD)
Status: **RUN COMPLETE**; **`rtwx_x0r_passed=false`**; **`instrument_ready=false`**
Prereg: `REPORT/REG/RTWX/RTWX0R_PREREG.md`
Command: `python -m aprwm_v0 rtwx-x0r`
Artifacts: `runs/rtwx_x0r/{header.json,run.header.txt,summary.json,latent.json,trajectories.npz,run.log}`
Does not: re-score X0 `physics_capacity_shift`; cup/stamp; RGB / X1; capacity \(R_P\); unlock R10

Next cell: **RTWX-X0S** (`REPORT/REG/RTWX/RTWX0S_PREREG.md`) — structure
audit, not \(\phi\)-tuning, not capacity.

## Honest bottleneck (do not mix FAILs)

This is **not** “APR-WM failed on RoboTwin.” \(\Delta s\neq 0\) and
SAPIEN \((q,\dot q,\ddot q,\tau)\) already exist. The recorded pattern
name `excitation_failure` is the X0R **gate** label (identity still
beats Euler). The scientific bottleneck is:

\[
\boxed{\textbf{the explicit cabinet dynamics model is structurally wrong.}}
\]

If it were only mistuned \(\phi\), scene \(\hat\phi\) would beat the
hardcoded \(\phi\). It did not. Do not ask “how to estimate \(\phi\)”
next. Ask which generalized-force / state terms (and whether logged
\(\tau\) is the integrator’s force) are missing. Do **not** relabel
`runs/rtwx_x0r/summary.json`.


## One-line result

Official `put_object_cabinet` was executed as a **new instrument** (cabinet
joint state only). **G1 PASS** (SAPIEN exposes \(q,\dot q,\ddot q,\tau\)).
**G0 / G2 / G3 FAIL.** Pattern **`excitation_failure`**. Capacity claim
**WITHHELD** (forbidden in this cell). R10 and TASK-X1 stay **LOCKED**.

Motion is **not** the X0 A/C identity-copy regime (\(E_1\sim 10^{-3}\)).
Identity NRMSE on cabinet \((q,\dot q)\) is \(0.585\). The frozen G0
second clause still fails: a closed-form Euler map with logged \(\phi\)
is **worse** than identity, so copying state is still the stronger
baseline.

## Command

```text
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
PYTHONPATH=/root/APR-WM \
  /root/miniconda3/envs/Robotwin/bin/python -m aprwm_v0 rtwx-x0r \
  --output /root/APR-WM/runs/rtwx_x0r \
  --robotwin-repo /root/RoboTwin
```

CuRobo / pytorch3d missing (`curobo_stub` as in X0); linspace planner stub.

## Frozen before first collect (not retuned)

Logged in `run.header.txt` / `header.json` **before** episode 1.

| item | value |
|---|---|
| task | `put_object_cabinet` only |
| train / test | **48 / 24** (P0-like; not 1000-ep X0 formal) |
| steps / cal prefix | 80 / 24 |
| macro-substeps | 8 (\(dt_{\mathrm{eff}}=0.032\,\mathrm{s}\)) |
| \(\epsilon_{\mathrm{exc}}\) | **0.08** (\(H=10\)) |
| \(\delta\) | **0.05** |
| \(\tau_1\) / \(\tau_{10}\) | **0.80 / 2.50** |
| latent widths | 16, 32, 64, 128; 40 epochs |
| seed | 8101 |
| \(s\) | cabinet \((q,\dot q)\) only (\(d_s=6\), 3 DoF) |
| \(u\) | applied generalized force via `set_qf` (\(d_u=3\)) |

Gates were chosen a priori so X0 A/C \(E_1\sim 10^{-3}\) copy **cannot**
pass G0. They are **not** copied from X0 formal task-B \(E_1=0.158\).

## Gate table

| gate | result | evidence |
|---|---|---|
| **G0** excitation | **FAIL** | median \(\|s_{t+H}-s_t\|/\mathrm{scale}(s)=1.174>0.08\) (**Δs PASS**); \(E_{\mathrm{id}}=0.585\), \(E_{\mathrm{oracle\text{-}Euler}}=2.045\); identity is **stronger** than Euler+\(\phi_{\mathrm{true}}\), so \(E_{\mathrm{id}}>E_{\mathrm{phy}}+\delta\) **FAIL** |
| **G1** physics closure | **PASS** | `get_qpos`, `get_qvel`, `get_qacc`, `set_qf`/`get_qf`, `compute_passive_force`; \(n_{\mathrm{dof}}=3\); finite-difference \(\ddot q\) also logged. **Not** `not_force_auditable` |
| **G2** \(\phi\) ID | **FAIL** | per-scene \(D_{\mathrm{cal}}\to\hat\phi\): median \(E_1(\hat\phi)=5.27\) **worse** than hardcoded \(\phi=(1.0,0.08,0.3)\) at \(3.47\). LS hits box bounds |
| **G3** latent rollout | **FAIL** | largest (\(H=128\)): \(E_1=0.562<0.80\) but \(E_{\mathrm{roll10}}\sim 4.65\times 10^6 \not< 2.50\). Finite, not the X0 “best-\(E_1\)=worst-roll” coincidence (best \(E_1\) is width 32; worst roll is 16) |

## Pattern

```text
pattern              = excitation_failure
rtwx_x0r_passed      = false
instrument_ready     = false
capacity_claim       = false
not_force_auditable  = false   (G1 passed)
```

G0 is a conjunction. The **state-change** half passed; the **identity-is-weak**
half failed because the structured \((q,\dot q,\tau)\to\ddot q\) Euler used
as “oracle physics” does not match 3-DoF cabinet + gravity/limits.
Do not read this as “cabinet does not move.”

## Latent (diagnostic; not a capacity curve)

| hidden | \(P\) | \(E_1\) | \(E_{\mathrm{roll10}}\) |
|---:|---:|---:|---:|
| 16 | 534 | 0.630 | \(1.10\times 10^7\) |
| 32 | 1574 | **0.559** | \(4.78\times 10^6\) |
| 64 | 5190 | 0.575 | \(4.10\times 10^6\) |
| 128 | 18566 | 0.562 | \(4.65\times 10^6\) |

Open-loop compounding, same qualitative failure as X0 latents, now on
the cabinet joint state rather than a 27-D proprioception blob.

## Claims ceiling (honored)

X0R **may** say: official cabinet trajectories have non-trivial \(\Delta s\)
on \((q,\dot q)\); force/joint accounting is **exposed** (G1); \(\hat\phi\)
from calibration did **not** help; latent is **not** rollout-sane.

X0R **may not** say: physics reduces latent capacity; X0
`physics_capacity_shift` is validated; pick/stamp are solved; \(R_P\).

## Blockers (do not unlock a capacity cell)

1. **Physics model ≠ plant.** Independent per-DoF Euler with
   \((m,b,\mu)\) loses to identity. Need a residual that actually
   accounts generalized force on the 3-DoF cabinet (limits, gravity,
   coupling), or plant-replay oracle — **without** retuning
   \(\epsilon_{\mathrm{exc}},\delta,\tau_1,\tau_{10}\).
2. **\(\phi\) unidentified.** Least-squares from \(D_{\mathrm{cal}}\)
   does not improve held-out vs hardcoded \(\phi\).
3. **Latent \(E_{\mathrm{roll10}}\) explodes** at every width.

G1 is **not** the blocker. Next is **RTWX-X0S** (minimal family vs
identity on cabinet \(\ddot q\)), not “estimate \(\phi\)” and not RGB.

## Tests

`tests/test_rtwx_x0r.py` (numpy backend; no full SAPIEN collect): CLI,
r10 refuse, frozen ε outside X0 copy, G1 `not_force_auditable` path,
G3 best-\(E_1\)=worst-roll forbid, header-before-metrics.

## Ledger

```text
CAP-X3            = PASS
PLAN-X            = CLOSED at iso_sufficient

RoboTwin-X0-P0    = PASS
RoboTwin-X0-smoke = PASS
RoboTwin-X0       = RAN
                    metrics finite
                    capacity claim WITHHELD

RoboTwin-X0R      = RAN
                    G1 PASS (force-auditable)
                    G0 FAIL (identity still beats Euler physics)
                    G2 FAIL (phi_unidentified)
                    G3 FAIL (latent_rollout_unstable)
                    pattern = excitation_failure
                    rtwx_x0r_passed = false
                    instrument_ready = false
                    next = RTWX-X0S structure audit (not φ-ID, not capacity)

RTWX-X0S          = RAN / FAIL timing_mismatch
RTWX-X0F          = RAN / FAIL force_channel_unresolved
TASK-X0           = RAN; FAIL (3/3); X1 LOCKED
TASK-X1           = LOCKED
TASK-XL           = DESIGN FROZEN ONLY; LOCKED
PLAN-X2           = LOCKED
RoboTwin-X1       = LOCKED
R10               = LOCKED
```
