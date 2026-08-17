# APR-WM V6R0 Preflight: Frozen Mechanism, Realistic Physics, Oracle State

> Status update: the interface failure diagnosed here was repaired and passed
> the ordered C0/C1 checks in [REPORT/REP/R0/V6R01_REPORT.md](REPORT/REP/R0/V6R01_REPORT.md). The original
> results below remain unchanged as the historical no-go record.

## Executive decision

V6R0 has a runnable realism-bridge scaffold and a completed three-seed headless SAPIEN preflight. It is **not yet an official RoboTwin Open Laptop result**, and the present proxy fails the prerequisite for a valid R0-C experiment:

\[
\boxed{\text{C0 must be structurally adequate before C1/C2 can test revision}}
\]

In the current hinge proxy, pure physics has lower rollout error than both the no-revision residual model and frozen V6 in every aggregate regime/horizon comparison. C1 exact operator recovery is `1.39%`, while C0 false revision is `2.78%`. The correct decision is therefore to stop before a larger seed/window sweep, repair the physics–representation interface, and rerun the same preflight.

This is a useful failure: it localizes the first simulation-to-robotics transport problem upstream of routing, selection, and acceptance.

## 1. Implemented scope

The new `v6r0-smoke` path freezes the V6 revision mechanism and adds:

- a headless SAPIEN articulated hinge with oracle joint state;
- local state extraction as `[q, qdot, applied torque]`;
- C0 parameter variation, C1 in-library nonlinear drag, and C2 hidden-memory hysteresis;
- expert-action rollouts and deterministic counterfactual branch manifests;
- horizons `H={1,4,8,16}`;
- frozen V4 proposal, V5 selection, and V6 sequential validation logic;
- mechanism, rollout, and residual-assimilation plots;
- a RoboTwin capability gate that refuses to label a proxy run as an official task run.

The run summary records:

```text
scope: R0-C headless SAPIEN hinge proxy
official_robotwin_task_executed: false
mechanism_frozen_from_v6: true
operator_library_frozen: true
oracle_state: true
visual_perception: false
learned_policy: false
```

The preflight uses seeds `13/23/33`, 24 windows per regime per seed, eight synthetic query branches per window, and observation noise `0.01`. It is a development diagnostic, not the proposed formal `2 tasks × 5 seeds × 20 windows × 5 branches` matrix.

## 2. Environment readiness

The local RoboTwin checkout contains the three task definitions and SAPIEN now runs in an isolated `/home/dong/miniconda3/envs/RoboTwin` environment. The capability gate reports:

| Capability | Status |
|---|---:|
| RoboTwin checkout | ready |
| SAPIEN / Gymnasium / HDF5 | ready |
| physics-only proxy | ready |
| `mplib` | missing |
| XPolicyLab submodule | not initialized |
| object assets | missing |
| embodiment assets | missing |
| official task smoke | **not ready** |

The official Hugging Face archives inspected during setup total approximately `14.9 GB` compressed: background textures `10,970,687,027` bytes, objects `3,737,778,549` bytes, and embodiments `219,859,313` bytes. Extraction requires additional storage and the observed connection made this a multi-hour download, so the assets were not fetched implicitly.

There is also a task-definition mismatch that must be resolved before using Beat Block Hammer as proposed. RoboTwin currently creates its target block with `is_static=True` in `/home/dong/Projects/RoboTwin/envs/beat_block_hammer.py`. Consequently, block velocity and impact-transfer residuals are not valid native metrics for that task. R0 should either use contact impulse and hammer rebound, or declare a separate dynamic-block intervention rather than silently changing the benchmark.

## 3. Three-seed mechanism results

Means below are across the three development seeds; intervals are a bootstrap over only three seed means and must not be treated as formal uncertainty estimates.

| Metric | Mean | Bootstrap 95% interval |
|---|---:|---:|
| Tangent detection AUROC | 0.6045 | [0.4913, 0.6675] |
| Magnitude detection AUROC | 0.4407 | [0.4002, 0.4740] |
| Natural C1 top-3 recall | 63.89% | [54.17%, 70.83%] |
| Controlled C1 candidate coverage | 100% | [100%, 100%] |
| C1 selection accuracy | 25.00% | [16.67%, 33.33%] |
| Acceptance given correct C1 selection | 4.17% | [0%, 12.50%] |
| C1 exact operator recovery | 1.39% | [0%, 4.17%] |
| C0 false revision | 2.78% | [0%, 8.33%] |
| C2 unknown rejection | 98.61% | [95.83%, 100%] |
| Mean validation samples | 4.02 | [4.00, 4.06] |
| C1 residual assimilation ratio | 9.72% | [5.88%, 16.67%] |

Tangent evidence is directionally better than magnitude, but its interval crosses chance and the downstream revision chain is unusable. High C2 rejection is not sufficient evidence of success because an overly conservative validator can obtain it while failing C1.

## 4. Counterfactual rollout results

State RMSE normalizes joint position and velocity. These values are means across seeds.

| Regime | Horizon | Physics | No revision | Frozen V6 |
|---|---:|---:|---:|---:|
| C0 parameter-only | 1 | 0.0213 | 0.0327 | 0.0326 |
| C0 parameter-only | 4 | 0.0696 | 0.0740 | 0.0740 |
| C0 parameter-only | 8 | 0.1651 | 0.1699 | 0.2005 |
| C0 parameter-only | 16 | 0.4139 | 0.4152 | 1.0208 |
| C1 in-library drag | 1 | 0.0183 | 0.0285 | 0.0270 |
| C1 in-library drag | 4 | 0.0456 | 0.0538 | 0.0528 |
| C1 in-library drag | 8 | 0.0677 | 0.0769 | 0.0768 |
| C1 in-library drag | 16 | 0.1357 | 0.1405 | 0.1440 |
| C2 outside-library | 1 | 0.0295 | 0.0396 | 0.0396 |
| C2 outside-library | 4 | 0.1033 | 0.1077 | 0.1077 |
| C2 outside-library | 8 | 0.2172 | 0.2205 | 0.2205 |
| C2 outside-library | 16 | 0.4359 | 0.4377 | 0.4772 |

Frozen V6 has one C0/H16 instability: mean stable fraction is `99.48%`, with one seed at `98.44%`. The rollout implementation records instability before clamping the numerical state, so the CSV remains finite without hiding the event.

The central result is unambiguous: **pure physics wins all 12 aggregate comparisons**. The residual does not provide a useful temporary correction, and accepted revision does not improve long-horizon prediction.

## 5. Preregistered claim audit

| Hypothesis | Preflight status | Reason |
|---|---|---|
| H1: interaction-local residual routing remains useful | Not evaluated | The one-DOF proxy has no meaningful interaction-level routing comparison. |
| H2: tangent evidence transports better than magnitude | Weak directional signal only | Mean AUROC is higher, but three-seed interval crosses 0.5 and the detector is not reliable. |
| H3: accepted revision improves multi-step rollout | **Failed** | Frozen V6 never beats pure physics in the aggregate rollout table. |
| H4: residual use falls after correct assimilation | **Failed operationally** | Assimilation is only 9.72%, while exact recovery is 1.39%; the small decline cannot support the claim. |
| H5: contact-heavy task benefits more | Not evaluated | Official Open Laptop and Beat Block Hammer were not executable. |

No V6R0 scientific claim should be made from this run.

## 6. Failure localization

The state adapter correctly exposes `[q, qdot, torque]`, but the frozen two-dimensional V6 operator interface currently feeds only `[torque, qdot]` to an acceleration target. This drops position and does not explicitly express articulated dynamics:

\[
M(q)\ddot q + h(q,\dot q)=\tau + r_\tau.
\]

As a result, SAPIEN's native articulated effects appear as a structured `|qdot|qdot` residual even in C0. The injected negative C1 drag then partially cancels that native residual instead of creating a clean new structure. In development fits, the normalized `|qdot|qdot` coefficient changed from roughly `+0.918` in C0 to `+0.672` in C1. Thus the experiment lacks a legitimate structural ground truth:

\[
\text{observed residual}
\neq
\text{injected C1 operator alone}.
\]

This invalidates proposal recall, selection, acceptance, and assimilation as tests of mechanism transport. Changing V6 thresholds or increasing sample counts would optimize against a malformed control condition.

## 7. Required minimum revision before R0

The next step is an adapter correction, not V7 and not RGB:

1. retain the requested local state `[q, qdot, torque]` throughout the model interface;
2. predict a physics-consistent generalized-force residual
   \(r_\tau=M(q)\ddot q+h(q,\dot q)-\tau\), or include the full known rigid-body terms in the base predictor;
3. require C0 tangent-orthogonal residual to be statistically indistinguishable from measurement/discretization noise under the same interventions;
4. only after that invariant holds, inject C1 and rerun the frozen V6 selection/acceptance chain;
5. install RoboTwin assets, initialize XPolicyLab, and add `mplib` before calling any run an official Open Laptop smoke;
6. redefine Beat Block Hammer metrics around native contact impulse/hammer rebound, or preregister a separately named dynamic-block variant.

The go/no-go gate for the next run is:

\[
\boxed{
\text{C0 adequacy first}
\;\Longrightarrow\;
\text{then test C1 recovery and rollout gain}
}
\]

R0-N, Handover stress tests, and all RGB/slot work remain paused until this gate passes.

## 8. Reproduction and artifacts

Run the capability check:

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r0-doctor \
  --robotwin-repo /home/dong/Projects/RoboTwin
```

Run the current diagnostic proxy:

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r0-smoke \
  --output runs/v6r0/smoke \
  --device cpu \
  --seeds 13 23 33 \
  --episodes-per-regime 24
```

Artifacts are under `runs/v6r0/smoke/`: capability and run summaries, seed and aggregate metrics, horizon rollouts, deterministic branch metadata, and three diagnostic plots.

The full local regression suite passes: `54 passed`.
