# APR-WM V6R0.2: Frozen V6 Force-space Reintegration

> Status update: the subsequent five-seed C0/C1/C2/native formal bridge is an
> overall no-go because native accepted revisions have negative rollout
> utility; see [V6R03_REPORT.md](V6R03_REPORT.md).

## Executive conclusion

The three-seed V6R0.2 development run passes all declared gates. In the controlled C0/C1 SAPIEN hinge setting, the complete trigger–proposal–selection–acceptance–assimilation loop remains stable after moving the residual to physics-consistent generalized-force coordinates.

The central result is:

\[
\boxed{
\text{force-space closure}
\rightarrow
\text{stable frozen-V6 reintegration}
}
\]

C0 false revision is `0%`, and the maximum aggregate frozen-V6 minus pure-physics rollout RMSE is `1.88e-4`, below the preregistered `0.002` non-inferiority margin. In C1, frozen V6 beats the no-revision residual baseline at every horizon; its H16 improvement is `0.02565`. Exact structural recovery and residual assimilation are both `100%`.

These are development results on a clean in-library discrepancy, not a population-level safety guarantee or an official RoboTwin result.

## 1. Scope and frozen mechanism

V6R0.2 uses only:

- C0: adequate articulated physics with varying density and known linear damping;
- C1: the same system plus hidden generalized drag `-0.12 |qdot|qdot`;
- oracle local state `[q,qdot,torque]`;
- generalized-force residuals from the validated R0.1 adapter;
- the unchanged eight-operator V4 dictionary;
- V5 posterior-weighted three-probe selection;
- V6 dual old/runner-up sequential BF-style acceptance;
- the V3 tangent-evidence trigger before residual fallback or validation.

C2, R0-N, RGB, official RoboTwin tasks, and asset downloads remain disabled.

The only necessary interface split is that parameter evidence and structural operators now receive different physics-consistent coordinates:

\[
J_\theta
=
\frac{\partial r_\tau}{\partial(\rho,c)},
\qquad
\phi_k=\phi_k(q,\dot q).
\]

The posterior update, action score, proposal complexity, acceptance thresholds, and stopping boundaries are unchanged.

## 2. Protocol

The run uses seeds `13/23/33`, 24 episodes per regime and seed, 8 passive samples, 16 discovery samples, 32 diagnostic actions, 3 selection probes, and at most 32 validation samples.

Counterfactual horizons correspond to:

| H | Physical time |
|---:|---:|
| 1 | 16 ms |
| 4 | 64 ms |
| 8 | 128 ms |
| 16 | 256 ms |

Rollout queries are restricted to `q in [0.35,0.75]`, `|qdot|<=0.4`, and `|torque|<=0.2`. This keeps the fixed 256 ms experiment inside the unconstrained hinge interior. Joint-limit contact is a different structural regime and is intentionally not mixed into C1.

Three predictors are compared from identical query states:

- `physics`: no residual force;
- `no_revision`: fitted parameter-tangent correction plus generic RBF residual;
- `frozen_v6`: accepted explicit operator, triggered fallback while unresolved, or pure physics when no structural trigger exists.

The last routing rule is essential. The full V3–V6 sequence is:

\[
\text{no trigger}\to\text{physics},
\quad
\text{trigger, no acceptance}\to\text{fallback},
\quad
\text{accepted}\to\text{explicit operator}.
\]

## 3. Mechanism chain

| Metric | Three-seed mean | Seed range |
|---|---:|---:|
| Tangent C0/C1 AUROC | `1.000` | `1.000–1.000` |
| Magnitude C0/C1 AUROC | `1.000` | `1.000–1.000` |
| C1 trigger recall | `100%` | `100–100%` |
| Natural C1 top-3 candidate coverage | `100%` | `100–100%` |
| C1 selection accuracy | `100%` | `100–100%` |
| Acceptance given correct C1 selection | `100%` | `100–100%` |
| C1 exact operator recovery | `100%` | `100–100%` |
| Estimated operator coefficient | `-0.12011` | `-0.12032–-0.11981` |
| Mean C1 validation samples | `4.94` | `4.67–5.33` |

Magnitude and tangent both saturate because this C1 is deliberately high-SNR and exactly represented. V6R0.2 therefore tests loop stability and assimilation, not tangent superiority.

## 4. C0 safety gate

| Metric | Result | Gate |
|---|---:|---:|
| C0 structural trigger | `2.78%` | diagnostic only |
| C0 false revision | `0%` | `<1%` point estimate |
| C0 defer | `2.78%` | diagnostic only |
| Max aggregate V6–physics RMSE | `1.88e-4` | `<=0.002` |
| Rollout stability | `100%` | finite/stable |

Two C0 episodes in seed 33 crossed the tangent trigger and were deferred after validation; neither was accepted. This shows the distinction between a noisy structural alarm and a false permanent revision. It also leaves an evidence-cost issue for a larger study: the three-seed bootstrap range for the trigger/defer rate is `0–8.33%`.

C0 rollout means are:

| H | Pure physics | No revision | Frozen V6 | V6 − physics |
|---:|---:|---:|---:|---:|
| 1 | `0.000195` | `0.000383` | `0.000202` | `+0.000008` |
| 4 | `0.000746` | `0.001547` | `0.000771` | `+0.000025` |
| 8 | `0.001414` | `0.003381` | `0.001458` | `+0.000044` |
| 16 | `0.002552` | `0.007420` | `0.002740` | `+0.000188` |

Thus unconditional residual fallback would damage adequate physics, whereas trigger-gated V6 remains within the declared margin.

## 5. C1 rollout gain

| H | Pure physics | No revision | Frozen V6 | No revision − V6 |
|---:|---:|---:|---:|---:|
| 1 | `0.001585` | `0.002240` | `0.000196` | `0.002044` |
| 4 | `0.007573` | `0.007511` | `0.000727` | `0.006785` |
| 8 | `0.023187` | `0.013010` | `0.001307` | `0.011702` |
| 16 | `0.083069` | `0.027677` | `0.002024` | `0.025653` |

Frozen V6 improves over no revision at every horizon. At H16 it reduces error by approximately `92.7%` relative to the no-revision residual baseline and `97.6%` relative to pure physics.

The no-revision fallback is slightly worse than physics at H1. This is another reason to require routing: a flexible correction fitted from finite noisy context is not guaranteed to help at every query, even when a structural discrepancy exists.

## 6. Assimilation

Before structural acceptance, every triggered C1 episode uses the generic residual fallback. All C1 episodes are correctly revised, after which fallback use falls to zero:

\[
A_{res}
=
1-\frac{R_{after}}{R_{before}}
=1.0.
\]

Because rollout error simultaneously falls at every horizon, this is genuine assimilation in the experiment's operational sense: the identified drag moves from generic neural correction into the explicit dynamics library without degrading prediction.

## 7. Two implementation diagnostics before the final run

Two intermediate outputs are preserved because they clarify the boundary:

1. `runs/v6r02/smoke` used broad interventions that could hit the PhysX hinge limit. Pinocchio rollout lacks that constraint, so even C0 pure physics reached approximately `0.85` H16 RMSE. Those rollouts are invalid for force-space comparison.
2. `runs/v6r02/local_window` removed limit contact but still sent every non-accepted episode to residual fallback. C0 false revision was `0%`, yet noisy fallback accumulated a `0.00492` maximum V6–physics gap and failed non-inferiority.

The final `runs/v6r02/trigger_gated` run changes neither selector nor acceptance. It restores the frozen V3 trigger semantics and passes. This localizes the two pitfalls as rollout-domain mismatch and missing routing control, not acceptance failure.

## 8. Claim audit and next step

Supported at development scale:

- the complete frozen loop can run in generalized-force coordinates;
- C0 is not materially damaged under the declared margin;
- C1 revision improves multi-step rollout over both baselines;
- accepted structure assimilates generic residual use.

Not established:

- a population false-revision probability below `1%`—72 C0 episodes with zero events are insufficient for that guarantee;
- robustness to lower C1 SNR, wrong candidates, outside-library C2, contacts, unknown parameters, or native RoboTwin tasks;
- performance outside the constraint-free 256 ms local window.

The R0.2 result authorizes returning to V6R0 with C2 and a constraint-aware counterfactual protocol. R0-N and asset download should still wait until that controlled C0/C1/C2 matrix passes.

## 9. Reproduction

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r02 \
  --output runs/v6r02/trigger_gated \
  --device cpu \
  --seeds 13 23 33 \
  --episodes-per-regime 24
```

The complete regression suite passes: `61 passed`.
