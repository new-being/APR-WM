# APR-WM V6R0.5: Dynamical-safety Diagnostics

## Executive conclusion

R0.5 identifies a strong mechanistic hypothesis but does **not** select an
acceptance threshold or start R0.6.

Across the one-step force-accepted revisions, unsafe H16 outcomes are ranked
consistently by three structure-aware signals:

| Diagnostic risk score | Selection AUROC | Confirmation AUROC |
|---|---:|---:|
| Maximum positive revision power | `0.988` | `1.000` |
| Negative effective-damping violation | `0.917` | `1.000` |
| Maximum macro-map Jacobian spectral radius | `0.833` | `1.000` |
| Endogenous support distance | `0.905` | `0.914` |
| Parameter Mahalanobis displacement | `0.738` | `0.871` |
| Euclidean parameter displacement | `0.202` | `0.586` |
| One-step force gain | `0.536` | `0.471` |

The dominant unsafe mode is not simply “large parameter refit.” It is an
**anti-dissipative structural revision**: all ten accepted `abs_v_v` candidates
have positive coefficients, inject mechanical power, and are H16-unsafe.

However, operator identity is strongly confounded with outcome. An
`abs_v_v`-identity baseline itself scores `0.917/1.000`, and the strict
H2/H4/H8-accepted population contains only one H16 failure. The current data
therefore support passivity/Jacobian constraints as the leading R0.6
hypothesis, but do not establish a general cross-operator safety predictor.

RoboTwin remains blocked and no assets were downloaded.

## 1. Question and frozen protocol

R0.5 asks only:

\[
\boxed{
\text{Which diagnostics rank locally useful revisions that become H16-unsafe?}
}
\]

It replays the deterministic R0.4-A episodes without modifying proposals,
selection, fitting, acceptance, rollout horizons, or thresholds. No diagnostic
threshold is fitted.

Two R0.4 cohorts remain separate:

- selection cohort: `2001, 2011, 2021, 2031, 2041`;
- confirmation cohort: `4001, 4011, 4021, 4031, 4041`.

Each contains 24 native SAPIEN episodes per seed.

## 2. Two nested diagnostic populations

The primary population contains all revisions that passed independent
one-step force acceptance. This corresponds to the R0.3 failure mode:

| Cohort | Force accepted | H16 safe | H16 unsafe |
|---|---:|---:|---:|
| Selection | 20 | 14 | 6 |
| Confirmation | 19 | 14 | 5 |

The stricter population additionally passed independent H2/H4/H8 rollout and
short-window stability checks:

| Cohort | Short accepted | H16 safe | H16 unsafe |
|---|---:|---:|---:|
| Selection | 12 | 12 | 0 |
| Confirmation | 12 | 11 | 1 |

A revision is labeled unsafe when its candidate H16 gain over fallback is
non-positive or any H16 query is numerically unstable. Thus the primary
population supports meaningful AUROC estimates, while the strict R0.4
counterexample population is underpowered: selection AUROC is undefined and
confirmation AUROC has only one positive label.

## 3. Diagnostics

All risk directions are fixed before labels are inspected.

### Parameter displacement

Both Euclidean displacement and posterior-normalized displacement are recorded:

\[
C_\theta
=(\theta'-\theta)^\top\Sigma_\theta^{-1}(\theta'-\theta).
\]

### Local discrete dynamics

At initial and model-induced short-rollout states, the candidate macro map is
finite-differenced with respect to `[q, qdot]`. The diagnostic is:

\[
\max_{s\in\mathcal S_{short}}\rho(J_F(s)).
\]

This is a ranking feature, not a stability certificate.

### Effective damping and revision power

For total candidate correction `r_tau`, local effective damping is

\[
c_{eff}=c-\frac{\partial r_\tau}{\partial \dot q}.
\]

The violation score is `max(0,-min c_eff)`. The passivity proxy is the maximum
unexplained positive revision power:

\[
\max_{s\in\mathcal S_{short}}
\left[r_\tau(s)\dot q\right]_+.
\]

Because native R0-N discrepancy is hidden dissipation, persistent positive
power is physically inconsistent unless an omitted external energy source is
declared.

### Endogenous support shift

Model-induced short-rollout states are normalized and compared with the nearest
force-evidence state. The maximum nearest-neighbor distance measures whether a
candidate drives itself away from its evidence support.

The benchmark also records equilibrium-force proxy, signed/absolute structural
coefficient, short-rollout gain, force gain, and operator identity.

## 4. Primary AUROC results

| Signal, larger risk unless marked | Selection | Confirmation | Pooled |
|---|---:|---:|---:|
| Positive revision power | `0.988` | `1.000` | `0.990` |
| Negative damping violation | `0.917` | `1.000` | `0.955` |
| Jacobian spectral radius | `0.833` | `1.000` | `0.916` |
| Support distance | `0.905` | `0.914` | `0.919` |
| Lower short-rollout gain | `0.976` | `0.857` | `0.919` |
| Mahalanobis displacement | `0.738` | `0.871` | `0.789` |
| Structural coefficient magnitude | `0.643` | `0.714` | `0.682` |
| Euclidean parameter displacement | `0.202` | `0.586` | `0.357` |
| Lower force-validation gain | `0.536` | `0.471` | `0.519` |
| Equilibrium-force proxy | `0.583` | `0.429` | `0.506` |

Only two confirmation seeds contain both safe and unsafe force-accepted
revisions, so seed-level AUROC is defined for two of five seeds. Cohort AUROC
must not be presented as a five-seed confidence interval.

The main negative findings are also useful:

- force-validation gain is near chance;
- raw parameter correction norm is not a useful risk ordering;
- Mahalanobis normalization contains information, but is weaker than direct
  dynamical/passivity diagnostics and remains uncalibrated;
- equilibrium shift is irrelevant for the dominant velocity-dependent failure.

## 5. Operator confounding and the actual failure mechanism

Operator outcomes are:

| Cohort | Operator | Force accepted | H16 unsafe | Short accepted |
|---|---|---:|---:|---:|
| Selection | `abs_v_v` | 5 | 5 | 0 |
| Selection | `position_damping` | 13 | 0 | 11 |
| Selection | `saturation` | 1 | 1 | 0 |
| Selection | `velocity_coupled` | 1 | 0 | 1 |
| Confirmation | `abs_v_v` | 5 | 5 | 1 |
| Confirmation | `position_damping` | 12 | 0 | 10 |
| Confirmation | `saturation` | 1 | 0 | 1 |
| Confirmation | `x2` | 1 | 0 | 0 |

Every accepted `abs_v_v` coefficient is positive (`0.210–0.454` selection,
`0.191–0.628` confirmation). Since

\[
r_\tau=\alpha\dot q|\dot q|,
\qquad \alpha>0,
\]

the candidate force is aligned with velocity and injects energy. Conversely,
all 25 accepted `position_damping` coefficients are negative and all are safe.

Thus the three leading diagnostics are physically coherent, but they are also
partly detecting this operator/sign split. The result cannot yet show that
Jacobian or power ranks safe and unsafe candidates *within the same operator
family*.

The sole candidate that passes H2/H4/H8 and later fails H16 is confirmation
seed 4011, episode 1:

| Feature | Value |
|---|---:|
| Operator | `abs_v_v` |
| H16 gain | `-8.9938` |
| H16 stable queries | `62.5%` |
| Mahalanobis displacement | `4134.3` |
| Maximum Jacobian spectral radius | `1.6766` |
| Negative damping violation | `0.7833` |
| Maximum positive revision power | `2.3037` |
| Support distance | `0.7842` |

This failure is much more directly described as an anti-damping structural
trap than as a large-Euclidean-parameter compensation trap: its Euclidean
parameter displacement is only `0.263`.

## 6. Scientific conclusion

R0.5 supports:

\[
\boxed{
\text{current native unsafe revisions}
\approx
\text{anti-dissipative operator/sign choices}
}
\]

and rejects the simpler diagnosis:

\[
\boxed{
\text{unsafe revision}
\approx
\text{large Euclidean parameter refit}
}
\]

It also preserves the broader principle:

\[
\boxed{
\text{local predictive evidence}
\neq
\text{structure-preserving dynamics}
}
\]

This is a diagnostic result, not an R0.6 go result. The next method should be
preregistered as one structure-preserving revision rule rather than a tuned
score mixture. The leading candidate is a passivity constraint on the
zero-input structural correction over the declared support:

\[
r_{\tau,M}(q,\dot q,0)\dot q\le 0,
\]

with Jacobian and H16 stability retained as evaluation metrics, not threshold
ingredients. Before making a general claim, fresh evaluation must include
matched operator families with both safe and unsafe coefficient signs; this is
needed to separate physical constraint value from operator-identity lookup.

R0.6 should still require C0 protection, C1 exact dissipative recovery, native
H16 non-inferiority, and accepted stability above `99%` on untouched seeds.

## 7. Reproduction

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r05 \
  --output runs/v6r05/diagnostics \
  --device cpu \
  --selection-seeds 2001 2011 2021 2031 2041 \
  --confirmation-seeds 4001 4011 4021 4031 4041 \
  --episodes 24
```

The complete regression suite passes: `70 passed`.
