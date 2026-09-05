# APR-WM V6R0.3: Formal Held-out SAPIEN Bridge

## Executive conclusion

V6R0.3 is an informative **overall no-go**. Five held-out seeds confirm the R0.2 C0/C1 result and show strong C2 open-set rejection, but R0-N exposes a new failure:

\[
\boxed{
\text{positive one-step force likelihood}
\not\Rightarrow
\text{positive rollout utility}
}
\]

The frozen V6 accepts native operators in `21.67%` of R0-N episodes. Those accepted revisions improve independent force-space validation MSE by `+0.00492` on average, yet their valid-window H16 rollout gain is `-2.91`; aggregate frozen-V6 H16 RMSE rises to `2.032` versus `0.333` for no revision. R0-N therefore fails the preregistered positive-utility gate by a wide margin.

RoboTwin assets should not be downloaded yet. The formal bridge has done its job: it localizes the next bottleneck to rollout-safe acceptance/model parameterization under native discrepancy.

## 1. Frozen formal protocol

No R0.2 threshold or mechanism was tuned. The experiment uses five seeds disjoint from development:

```text
1001, 1011, 1021, 1031, 1041
```

Each seed contains 24 episodes in each of four regimes:

- `C0`: adequate physics;
- `C1`: in-library generalized drag `-0.12 |qdot|qdot`;
- `C2`: outside-library latent-history force;
- `R0-N`: no injected operator; PhysX native joint friction and link damping are retained while the Pinocchio predictor omits them.

The frozen loop remains:

\[
\text{tangent trigger}
\rightarrow
\text{V4 proposal}
\rightarrow
\text{V5 selection}
\rightarrow
\text{V6 sequential acceptance}
\rightarrow
\text{explicit operator or fallback}.
\]

All regimes use 8 passive observations, 16 discovery observations, 32 action candidates, 3 selection probes, at most 32 validation samples, and 8 counterfactual queries at `H={1,4,8,16}`.

## 2. Validity mask

Every truth window records whether the PhysX hinge entered a `0.02 rad` margin around its joint limits:

\[
\mathcal W_{valid}
=
\{w:\text{no unmodeled joint-limit activation}\}.
\]

Out-of-support windows remain in the dataset and count against coverage, but are excluded from prediction RMSE. This prevents a constraint mismatch from being reported as force-model failure without hiding how often it occurs.

Mean H16 valid-window coverage is:

| Regime | H16 valid fraction |
|---|---:|
| C0 | `80.21%` |
| C1 | `94.58%` |
| C2 | `82.08%` |
| R0-N | `89.06%` |

The minimum seed/regime/horizon coverage is `75%`, above the declared `70%` gate. H1/H4 coverage is `100%`; almost all exclusions arise only at H16.

## 3. Gate summary

| Gate | Result | Status |
|---|---:|---|
| C0 max V6–physics RMSE | `2.42e-5 <= 0.002` | Pass |
| C0 false revision | `0% < 1%` point estimate | Pass |
| C1 rollout gain at every H | yes | Pass |
| C1 assimilation | `100%` | Pass |
| C2 unknown rejection | `95.83% >= 90%` | Pass |
| C2 forced wrong revision | `4.17% <= 5%` | Pass, narrow |
| R0-N H16 no-revision − V6 | `-1.700` | **Fail** |
| R0-N assimilation | `21.67% > 0` | Pass mechanically |
| Minimum valid support coverage | `75% >= 70%` | Pass |

`overall_go=false` because assimilation without positive predictive utility is not success.

The C2 wrong-revision seed-bootstrap interval is `1.67–6.67%`; although the point estimate passes, the upper bound crosses the 5% target. Likewise, zero C0 events across 120 held-out episodes do not establish a population-level `<1%` guarantee.

## 4. C0/C1 replication on held-out seeds

C0 remains protected. Mean valid-window RMSE is:

| H | Physics | No revision | Frozen V6 |
|---:|---:|---:|---:|
| 1 | `0.000770` | `0.001202` | `0.000772` |
| 4 | `0.002930` | `0.004355` | `0.002937` |
| 8 | `0.005490` | `0.008222` | `0.005501` |
| 16 | `0.008412` | `0.014305` | `0.008436` |

C1 again closes the complete revision/assimilation chain:

- trigger, proposal coverage, selection, exact recovery, and assimilation: all `100%`;
- recovered coefficient: `-0.12003` versus truth `-0.12`;
- H16 frozen-V6 RMSE: `0.00531` versus `0.23933` no revision and `0.67482` physics.

| H | Physics | No revision | Frozen V6 |
|---:|---:|---:|---:|
| 1 | `0.020351` | `0.009669` | `0.000766` |
| 4 | `0.095369` | `0.041556` | `0.002596` |
| 8 | `0.261346` | `0.105043` | `0.004131` |
| 16 | `0.674821` | `0.239333` | `0.005311` |

This held-out replication supports the narrow claim that force-space V3–V6 transport is real for a clean in-library structure.

## 5. C2: safe unknown, weak fallback

The history force is invisible to the explicit state `[q,qdot,torque]` and absent from the operator library. Results are:

- trigger recall: `100%`;
- unknown rejection: `95.83%`;
- forced wrong revision: `4.17%`;
- residual fallback usage: `95.83%`.

The system usually refuses to force a known operator onto the history-dependent discrepancy. That answers the open-set identity question positively at the point-estimate level.

However, the generic fallback also sees no history variable. It therefore cannot predict the sign of the hidden force and is worse than physics:

| H | Physics | No revision | Frozen V6 |
|---:|---:|---:|---:|
| 1 | `0.019805` | `0.037528` | `0.037177` |
| 4 | `0.075889` | `0.131185` | `0.130001` |
| 8 | `0.144129` | `0.234461` | `0.232843` |
| 16 | `0.261429` | `0.409915` | `0.411778` |

Thus:

\[
\boxed{
\text{unknown rejection}
\not\Rightarrow
\text{useful residual fallback}
}
\]

The three-state semantics remain correct, but fallback needs a positive-utility condition and a state representation capable of expressing the missing history. Unknown should not automatically mean “trust the residual.”

## 6. R0-N failure localization

R0-N uses native PhysX dissipation rather than an injected formula. It triggers on every episode, and the generic no-revision residual improves H16 physics from `0.397` to `0.333`. The discrepancy is therefore detectable and partially learnable.

Frozen V6 accepts an explicit revision in `21.67%` of native episodes. Accepted operator frequencies over all native episodes are:

- `position_damping`: `16.67%`;
- `abs_v_v`: `5.00%`;
- all other operators: `0%`.

For accepted revisions:

- independent validation-MSE gain: `+0.00492`;
- mean absolute structural coefficient: `0.311`;
- mean parameter-correction norm: `9.02`;
- H16 rollout gain relative to no revision: `-2.91`;
- H16 stability: `85.45%`.

Aggregate rollout is:

| H | Physics | No revision | Frozen V6 |
|---:|---:|---:|---:|
| 1 | `0.033577` | `0.036912` | `0.033565` |
| 4 | `0.112644` | `0.110769` | `0.121103` |
| 8 | `0.209556` | `0.195498` | `0.937147` |
| 16 | `0.396863` | `0.332779` | `2.032384` |

The selected operator plus large parameter-tangent correction can fit independent one-step forces while producing an unstable vector field under repeated integration. The failure is not candidate availability or validation leakage: validation is independent and its local gain is genuinely positive. The acceptance objective is simply misaligned with long-horizon dynamical utility in this native regime.

## 7. Scientific conclusion

V6R0.3 supports:

\[
\text{physics closure}
+
\text{clean structural assimilation}
+
\text{open-set rejection}.
\]

It rejects the stronger claim that the current acceptance rule safely assimilates arbitrary native dynamics:

\[
\boxed{
\text{force-space fit}
\not\Rightarrow
\text{stable explicit dynamics}
}
\]

The next lightweight experiment should remain below RoboTwin and isolate two questions:

1. can acceptance include constraint-aware short-rollout utility/stability without sacrificing C0 safety and C1 power;
2. should unresolved fallback be used only when its held-out predictive utility exceeds staying with physics, especially when the state is not closed under hidden history.

No operator expansion, RGB, IG, or asset download is justified before those two failures are resolved.

## 8. Reproduction

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r03 \
  --output runs/v6r03/formal \
  --device cpu \
  --seeds 1001 1011 1021 1031 1041 \
  --episodes-per-regime 24
```

The complete regression suite passes: `63 passed`.
