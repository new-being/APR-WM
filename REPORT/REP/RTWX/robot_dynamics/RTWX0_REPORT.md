# RoboTwin-X0 Formal Report — Oracle-State Physics vs Latent Capacity

Date: 2026-08-27
Host: AutoDL (`/root/RoboTwin`, RTX 5090, NVIDIA Vulkan ICD)
Status: **RUN COMPLETE**; **capacity claim WITHHELD**
Prereg: `REPORT/REG/RTWX/RTWX0_PREREG.md`
Smoke: `runs/rtwx_x0/smoke/summary.json` (`rtwx_x0_smoke_passed=true`)
Artifacts: `runs/rtwx_x0/formal/{summary.json,task_A.json,task_B.json,task_C.json,run.log}`
Invalid prior run (NaN `E_roll10`): `runs/rtwx_x0/formal_nan_invalid/`
Does not: RGB in \(s\); policy / play_once success; CuRobo planning; unlock R10;
retune frozen episode counts or gates after seeing curves

## One-line result

The 3-task collection and capacity sweep **finished**. After the rollout
NaN fix, every reported \(E_1\) and \(E_{\mathrm{roll10}}\) is **finite**.
The runner wrote `pattern=physics_capacity_shift` and
`rtwx_x0_passed=true`. That flag is **not** a scientific PASS of the
boxed question. Honest reading (frozen): **run complete, capacity
substitution withheld.** **X0R = RAN / `excitation_failure`**
(`instrument_ready=false`; G1 PASS). TASK-XL after `instrument_ready`,
not after this FAIL. Retry = learned \(z_g,z^p\), not TASK-X1.

## Command

```text
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
PYTHONPATH=/root/APR-WM \
  /root/miniconda3/envs/Robotwin/bin/python -m aprwm_v0 rtwx-x0 \
  --output /root/APR-WM/runs/rtwx_x0/formal \
  --smoke-summary /root/APR-WM/runs/rtwx_x0/smoke/summary.json \
  --robotwin-repo /root/RoboTwin
```

Wall clock (this re-run): ~08:07–08:32 UTC+8. CuRobo / pytorch3d missing
as in smoke (`curobo_stub=true`); linspace planner stub used.

## Frozen protocol (not retuned)

| item | value |
|---|---|
| train / val / test episodes | 1000 / **200 listed, never collected** / 300 |
| steps | 120 |
| tasks | A `place_empty_cup` (`cup`); B `put_object_cabinet` (`object`); C `stamp_seal` (`seal`) |
| \(\phi\) train box | \(m\in[0.8,1.2]\), \(\mu\in[0.2,0.4]\) (set on actor; **not identified**) |
| \(\phi\) test | \(m=1.5\), \(\mu=0.5\) |
| physics \(\phi\) at eval | **hardcoded** \((1.0, 0.08, 0.3)\), not learned |
| latent widths | 16, 32, 64, 128; 40 epochs |
| residual widths | 0, 8, 16, 32 |
| rollout | \(H=10\) only (prereg also listed \(H=50\); not run) |

Action is random sinusoidal wrench, not native `qpos`. State is oracle
proprioception + object pose (+ cabinet \(q,\dot q\) on B). No RGB.

## Recorded numbers

NRMSE on full \(s\). Physics-only is hybrid `hidden=0`.

### Task A — `place_empty_cup` (\(d_s=23\))

| model | \(P\) | \(E_1\) | \(E_{\mathrm{roll10}}\) |
|---|---:|---:|---:|
| physics-only | 0 | **0.00326** | **0.0288** |
| latent 16 | 1095 | 0.923 | 7.44 |
| latent 32 | 2679 | 0.693 | 9.06 |
| latent 64 | 7383 | 0.251 | 11.70 |
| latent 128 | 22935 | 0.0957 | 16.18 |
| residual 8 | 495 | 0.368 | 55.4 |
| residual 16 | 1095 | 0.161 | 25.0 |
| residual 32 | 2679 | 0.0237 | 76.1 |

### Task B — `put_object_cabinet` (\(d_s=27\))

| model | \(P\) | \(E_1\) | \(E_{\mathrm{roll10}}\) |
|---|---:|---:|---:|
| physics-only | 0 | **0.158** | **0.250** |
| latent 16 | 1243 | 0.826 | 6.19 |
| latent 32 | 2971 | 0.774 | 7.90 |
| latent 64 | 7963 | 0.414 | 8.09 |
| latent 128 | 24091 | 0.212 | 21.88 |
| residual 8 | 571 | 0.392 | 28.6 |
| residual 16 | 1243 | 0.285 | 43.6 |
| residual 32 | 2971 | 0.164 | 24.4 |

### Task C — `stamp_seal` (\(d_s=23\))

| model | \(P\) | \(E_1\) | \(E_{\mathrm{roll10}}\) |
|---|---:|---:|---:|
| physics-only | 0 | **0.00328** | **0.0299** |
| latent 16 | 1095 | 0.921 | 4.87 |
| latent 32 | 2679 | 0.684 | 9.43 |
| latent 64 | 7383 | 0.232 | 9.83 |
| latent 128 | 22935 | 0.101 | 14.22 |
| residual 8 | 495 | 0.367 | 65.0 |
| residual 16 | 1095 | 0.163 | 29.5 |
| residual 32 | 2679 | 0.0261 | 52.8 |

Runner: `physics_left_of_latent=true` on A/B/C because
\(E_{\mathrm{roll10}}(\mathrm{phy})\le\min_h E_{\mathrm{roll10}}(\mathrm{latent})\).
That comparison is **not** the prereg primary (capacity curve at
**matched** \(E_{\mathrm{roll}}\)).

## Why the boxed question is not answered

1. **Near-identity physics on A/C.** \(E_1\approx 3\times 10^{-3}\) is the
   signature of \(s'\approx s\). `physics_predict` **copies** robot joints
   and Euler-integrates only object translation with `dt=1/250`. Arms are
   not actuated. Smoke already recorded `s_change_l2=0.0` after 20
   `scene.step()` with no wrench. If the applied wrench does not move the
   actor, “physics beats latent” is copy-the-state vs a drifting MLP.

2. **Latent one-step improves with width; rollout gets worse.** H=128 is
   best \(E_1\) and **worst** \(E_{\mathrm{roll10}}\) on every task. That
   is open-loop compounding, not a usable capacity ranking.

3. **Hybrid residual does not help.** Adding \(R_\psi\) **increases**
   \(E_1\) versus physics-only on all three tasks (B residual-32:
   0.164 vs 0.158). Rollouts of residuals are an order of magnitude worse.
   There is no \(\rho_r=\dim(z_r)/\dim(z^\star_{\mathrm{latent}})\) at
   matched quality.

4. **\(\phi\) is not learned.** Train/test mass–friction split is applied
   to SAPIEN actors; the predictor uses a fixed \(\phi\). This is not
   CAP-X3-style per-scene identification.

5. **Protocol holes (frozen, recorded).** `n_val_ep=200` unused; M2 \(H=50\)
   unused; M3 task coordinates unused; trajectories not saved, so object
   displacement cannot be audited after the fact.

6. **NaN fix (this re-run).** Concatenated transitions were previously
   rolled across episode boundaries; MLP explosion produced NaN
   `E_roll10` and a bogus `latent_sufficient`. Metrics are now finite
   (non-finite windows score \(+\infty\)). That repair does **not** make
   the physics left-shift a capacity result.

## Pattern vs ledger (honest reading; frozen 2026-08-27)

Runner `physics_capacity_shift` is a **program gate only**. It is **not**
a PASS of the prereg boxed question.

The leading fact is not that physics numbers look good. On A/C,
`physics_predict` copies robot joints and Euler-updates object
translation only; smoke already logged `s_change_l2=0` with no wrench.
\(E_1^{\mathrm{phy}}\approx 0.0033\) is the identity-copy regime:
state barely moves, so copying state looks accurate. Task B is more
informative (articulated cabinet actually moves:
\(E_1=0.158\), \(E_{\mathrm{roll10}}=0.250\) vs latent-128
\(0.212\), \(21.88\)), but still **not** a capacity claim:
\(\phi\) unidentified, residual hurts, val unused, \(H=50\) unrun,
latent rollout worsens with width.

Failure is **not** “APR-WM cannot work on RoboTwin.” Three instrument
layers:

1. **Excitation too weak** — identity is a strong baseline; \(\Delta s_t\)
   must be non-trivial.
2. **Physics interface does not cover RoboTwin dynamics** — no joint
   dynamics, contact, true generalized force, or per-scene \(\phi\).
   Not CAP-X3 “right structure, parameters to estimate.”
3. **Latent baseline is not a capacity curve** — \(H=128\) is best
   one-step and worst rollout on all three tasks.

\[
\boxed{
\text{physics side not sufficiently identified}
\quad+\quad
\text{latent side not rollout-stable}
}
\]

Comparing “who needs less latent” is meaningless until both sides exist.

**Do not open RoboTwin-X1 (RGB).** **X0R is NEXT** and highest IG
(`REPORT/REG/RTWX/RTWX0R_PREREG.md`): cabinet only. Pause pick/stamp.
Priority freeze: X0R dynamics instrument \(>\) TASK-X retry \(>\)
visual \(>\) diffusion. TASK-X0 family FAIL does not jump the queue.

### Paper methodological boundary

\[
\boxed{
\textbf{state prediction accuracy is meaningless as a physics-capacity
comparison when the underlying trajectories are nearly static.}
}
\]

Capacity benchmarks must report **excitation / state-change magnitude**.
Otherwise an identity model can look like a strong world model.

## Ledger

```text
CAP-X3            = PASS
PLAN-X            = CLOSED at iso_sufficient

RoboTwin-X0-P0    = PASS
RoboTwin-X0-smoke = PASS
RoboTwin-X0       = RAN
                    metrics finite
                    capacity claim WITHHELD
                    reason:
                    weak excitation /
                    identity-physics confound /
                    phi not identified /
                    latent rollout instability

RoboTwin-X0R      = FAIL / excitation_failure; G1 PASS; G0/G2/G3 FAIL
RTWX-X0S          = RAN / FAIL timing_mismatch
RTWX-X0F          = RAN / FAIL force_channel_unresolved
TASK-X0
├── cup      = FAIL  p_no_nll
├── cabinet  = FAIL  coverage_hole
└── stamp    = FAIL  p_no_nll
TASK-X1           = LOCKED
TASK-XL           = DESIGN FROZEN ONLY; LOCKED
PLAN-X2           = LOCKED
RoboTwin-X1       = LOCKED
R10               = LOCKED
```
