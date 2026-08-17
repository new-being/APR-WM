# APR-WM V6R0.1: Physics-consistent Generalized-force Interface

> Status update: frozen V6 was subsequently reintegrated in force space and
> passed the controlled C0/C1 gates; see [REPORT/REP/R0/V6R02_REPORT.md](REPORT/REP/R0/V6R02_REPORT.md).

## Executive conclusion

V6R0.1 passes its two ordered development gates:

1. under adequate C0 physics, the generalized-force residual closes at the numerical/noise floor on all three seeds;
2. only after that gate passed, the C1 nonlinear drag was restored and recovered cleanly in force space.

The result supports the diagnosis that the V6R0 no-go came from representation/interface misspecification rather than evidence that V3–V6 fails on articulated dynamics. It does **not** yet validate the full frozen V6 revision loop, official RoboTwin tasks, C2 open-set rejection, or rollout improvement.

## 1. Interface correction

The robotics adapter now preserves the oracle local state

\[
s=(q,\dot q,\tau)
\]

and defines the unexplained component in generalized-force coordinates:

\[
r_\tau
=
M(q)\ddot q+h(q,\dot q)-\tau_{known}.
\]

`tau_known` includes SAPIEN gravity/Coriolis compensation, the commanded hinge torque, and the declared linear damping. The C1 operator is excluded from `tau_known`, so its force

\[
-\alpha |\dot q|\dot q
\]

appears directly in `r_tau`.

The implementation also exposed a second interface issue. In SAPIEN `3.0.0b1`, the built revolute joint retained default friction `0.05` even though the builder was given `friction=0.0`. V6R0 had therefore treated an undeclared engine force as model inadequacy. R0.1 explicitly clears joint friction and link damping before creating the Pinocchio inverse-dynamics model. A direct dynamics probe then reduced

\[
M(q)\ddot q+h(q,\dot q)-\tau_{applied}
\]

from order `1e-1` to order `1e-8` at representative states.

This refines the failure diagnosis:

\[
e_{observed}
=
e_{repr/interface}
+e_{undeclared\ engine\ physics}
+e_{param}
+e_{struct}.
\]

Both of the first two terms must be controlled before tangent evidence is interpretable.

## 2. Ordered protocol

The run uses development seeds `13/23/33`, 24 independently parameterized hinges per seed, and 48 broad intervention states per hinge. Density varies from `110–230` and declared damping from `0.025–0.075`.

Phase A collects **only C0**. It does not execute operator proposal, selection, revision, sequential acceptance, C2, or rollout. C0 must pass every seed on:

- clean force RMSE `<= 1e-5`;
- absolute bias z-score `<= 3.5` relative to the injected `0.002` force-noise floor;
- observed variance/noise variance in `[0.8, 1.2]`;
- maximum absolute correlation with `q` or `qdot <= 0.1`.

Only if every seed passes does Phase B create C1 data. Phase B still does not restore full V6; it checks tangent separation and frozen-library operator identifiability in generalized-force coordinates. Its development gate requires mean tangent AUROC `>=0.8`, every seed above `0.5`, and mean C0 structural-trigger rate `<=1%`.

## 3. Phase A: C0 closure

| Metric | Three-seed mean | Seed range |
|---|---:|---:|
| Clean generalized-force RMSE | `1.248e-7` | `1.208e-7–1.278e-7` |
| Observed force bias | `2.478e-5` | `-6.802e-5–1.011e-4` |
| Absolute bias z-score | `1.190` | `0.701–1.715` |
| Variance / known noise variance | `1.0407` | `1.0264–1.0515` |
| Correlation with q | `0.0061` | `-0.0105–0.0161` |
| Correlation with qdot | `0.0100` | `-0.0293–0.0534` |
| Maximum absolute state correlation | `0.0311` | `0.0105–0.0534` |
| C0 structural-trigger rate | `0%` | `0–0%` |

All three seeds pass. The clean residual is roughly two orders of magnitude below the already strict `1e-5` threshold. The observed variance matches the injected force-noise variance, and neither position nor velocity shows a systematic residual trend.

Therefore:

\[
\boxed{\text{adequate physics becomes adequate again in force space}}
\]

## 4. Phase B: minimal C1 recovery

After the C0 gate passed, C1 added only

\[
r_\tau^{C1}=-0.12|\dot q|\dot q.
\]

The force-space parameter tangent uses

\[
J_\theta
=
\frac{\partial r_\tau}{\partial(\rho,c)},
\]

with a local density tangent from inverse dynamics and damping tangent `qdot`. Projection and the eight-operator V4 dictionary are unchanged conceptually; proposal, active selection, and acceptance are not run.

| Metric | Three-seed mean | Seed range |
|---|---:|---:|
| Tangent C0/C1 AUROC | `1.000` | `1.000–1.000` |
| Magnitude C0/C1 AUROC | `1.000` | `1.000–1.000` |
| C1 structural-trigger recall | `100%` | `100–100%` |
| Drag operator top-1 recovery | `100%` | `100–100%` |
| Drag operator top-3 recovery | `100%` | `100–100%` |
| Estimated drag coefficient | `-0.11984` | `-0.11997–-0.11959` |
| Coefficient MAE vs `-0.12` | `0.00111` | `0.00089–0.00125` |
| Orthogonal force RMSE before drag fit | `0.02543` | `0.02535–0.02551` |
| Orthogonal force RMSE after drag fit | `0.00198` | `0.00196–0.00199` |

The post-fit residual returns to the injected noise floor `0.002`. Here magnitude and tangent both saturate because C1 is deliberately clean and high-SNR. This run therefore verifies coordinate/interface recovery; it does not establish a new advantage for tangent evidence over magnitude.

## 5. Claim audit

| Claim | Status |
|---|---|
| C0 is closed in physics-consistent coordinates | Supported on three development seeds |
| Representation/interface misspecification caused the prior no-go | Strongly supported by the before/after closure probe |
| C1 force operator is identifiable after closure | Supported in the minimal controlled setting |
| Full V6 leaves C0 prediction unchanged | Not tested |
| Full V6 C0 false revision is below 1% | Not tested |
| Full V6 improves C1 multi-step rollout | Not tested |
| C2 open-set behavior transports | Not tested |
| Official RoboTwin Open Laptop works | Not tested; assets remain intentionally absent |

The `1.0` C1 scores must be read as a unit/integration check, not a formal benchmark result: only three development seeds are used, the operator is exactly in the frozen dictionary, and the signal-to-noise ratio is high.

## 6. Next gate

R0.1 now authorizes reintegration of the frozen V6 mechanism in force coordinates. The next experiment should restore only C0/C1 and test:

1. C0 frozen-V6 rollout is statistically non-inferior to pure physics;
2. C0 false revision is below `1%` at the development point estimate;
3. C0/C1 tangent AUROC remains above `0.8` with an interval not crossing `0.5`;
4. accepted C1 revision reduces multi-step error and residual fallback use.

C2, R0-N, RoboTwin assets, Beat Block Hammer, and RGB remain downstream of that gate. No acceptance threshold should be retuned unless the force-space reintegration reveals an acceptance-specific failure.

## 7. Reproduction

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r01-closure \
  --output runs/v6r01/closure \
  --device cpu \
  --seeds 13 23 33 \
  --episodes 24 \
  --samples-per-episode 48
```

Outputs include `summary.json`, seed and aggregate metrics, all force samples, a C0 state-dependence plot, and a C1 recovery plot. The complete local regression suite passes: `57 passed`.
