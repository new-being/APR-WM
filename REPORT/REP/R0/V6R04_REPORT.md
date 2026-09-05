# APR-WM V6R0.4: Dynamical Acceptance and History-aware Fallback

## Executive conclusion

V6R0.4 is an **overall no-go for RoboTwin**, but its two isolated experiments
have different outcomes:

1. **R0.4-A fails confirmatory evaluation.** Short-rollout acceptance is much
   safer than one-step force acceptance, but success at `H={2,4,8}` does not
   reliably transfer to H16. On fresh confirmatory seeds, H16 RMSE is `0.6390`
   versus `0.3691` for no revision, and accepted-revision stability is
   `98.125% < 99%`.
2. **R0.4-B passes its controlled representation test.** When an outside-library
   delayed force is made Markov by exposing three action lags, H16 RMSE falls
   from `0.01666` for physics to `0.00243`; the 12-feature memoryless model is
   worse than physics at `0.04995`.

The result sharpens both R0.3 principles:

\[
\boxed{
\text{finite-horizon validation}
\not\Rightarrow
\text{longer-horizon dynamical safety}
}
\]

and

\[
\boxed{
\text{history-dependent discrepancy}
+
\text{memoryless fallback}
=
\text{representation failure}
}
\]

RoboTwin assets were not downloaded.

## 1. Isolation protocol

Only one mechanism changes in each experiment.

- **R0.4-A (R0-N only):** force-space representation, tangent trigger,
  proposals, selection, parameter fit, generic fallback, and validity mask are
  frozen. Acceptance additionally sees independent short rollouts.
- **R0.4-B (C2 only):** revision is removed from the question. The experiment
  compares physics, a generous instantaneous residual basis, that same basis
  augmented with an explicit three-step action history, and oracle history.

The cohorts are disjoint from R0.3 and from each other:

| Cohort | Seeds | Role |
|---|---|---|
| A selection/ablation | 2001, 2011, 2021, 2031, 2041 | identify the viable predefined acceptance ablation |
| A confirmation | 4001, 4011, 4021, 4031, 4041 | fixed `rollout_stability` policy; no retuning |
| B held-out | 3001, 3011, 3021, 3031, 3041 | fixed history/fallback comparison |

Each seed contains 24 independently parameterized SAPIEN hinge episodes.

## 2. R0.4-A mechanism

Four predefined policies are compared:

1. `force_only`: frozen V6 force-space acceptance;
2. `rollout`: force acceptance plus improvement over both physics and fallback
   on independent `H={2,4,8}` branches;
3. `rollout_stability`: the above plus a hard veto for non-finite state,
   excessive velocity, or leaving the declared support;
4. `rollout_stability_parameter`: the above plus the provisional hard bound

\[
C_\theta
=(\theta'-\theta)^\top\Sigma_\theta^{-1}(\theta'-\theta)
\le 9.21.
\]

Final H1/H4/H8/H16 queries are independent of force validation and short
rollout validation. The short gate requires a margin of `1e-4` against both
incumbents at every validation horizon.

## 3. R0.4-A: selection result

The first cohort shows why rollout validation is necessary:

| Policy | Revision rate | H16 RMSE | Accepted H16 gain | Accepted H16 stability |
|---|---:|---:|---:|---:|
| Force only | `16.67%` | `1.9478` | `-2.3299` | `89.625%` |
| + short rollout | `10.00%` | `0.3366` | `+0.1977` | `100%` |
| + stability veto | `10.00%` | `0.3366` | `+0.1977` | `100%` |
| + parameter hard veto | `0%` | `0.3514` | `0` | `100%` |
| No revision | — | `0.3514` | — | `100%` |

The short-rollout gate removes the catastrophic accepted revisions on this
cohort and retains a nontrivial `10%`. The stability veto does not change its
decision, meaning the rollout comparison already removed the candidates that
failed the declared short-window stability test.

The parameter displacement diagnostic is extremely large: mean
`C_theta=4815.8` over force-accepted candidates. The provisional chi-square
threshold lets only `5%` of force-accepted candidates pass by itself and none
after intersection with rollout acceptance. It is therefore a safe abstention
rule here, but not a useful revision policy.

Crucially, the recommended policy was chosen after viewing this ablation. This
cohort is not used as confirmatory evidence.

## 4. R0.4-A: fresh confirmation fails

With `rollout_stability` fixed before running five new seeds:

| H | Physics | No revision | Force only | Rollout + stability |
|---:|---:|---:|---:|---:|
| 1 | `0.03389` | `0.03855` | `0.03548` | `0.03616` |
| 4 | `0.10924` | `0.11612` | `0.10978` | `0.10809` |
| 8 | `0.20313` | `0.21652` | `1.01745` | `0.20898` |
| 16 | `0.38821` | `0.36910` | `1.61108` | `0.63903` |

The fixed policy still accepts `10%` of episodes, but:

- mean accepted H16 gain is `-0.26088`;
- accepted H16 stability is `98.125%`;
- aggregate H16 stability is `99.6875%`, which hides instability concentrated
  in accepted branches;
- seed 4011 alone rises from `0.3058` no-revision H16 RMSE to `1.8002` after
  rollout-aware revision.

The aggregate H8 result is still positive relative to no revision
(`0.20898 < 0.21652`), yet H16 fails. This is direct evidence that validating
the same vector field over H8 does not establish H16 safety under the sampled
state distribution.

### R0.4-A gate decision

| Gate | Result | Status |
|---|---:|---|
| H16 revision <= no revision | `0.6390 <= 0.3691` | **Fail** |
| Stability among accepted revisions | `98.125% >= 99%` | **Fail** |
| Accepted H16 gain | `-0.2609 >= 0` | **Fail** |

R0.4-A is therefore **no-go**. Short rollout is useful evidence, not a
sufficient acceptance certificate.

The hard parameter veto returns exactly to no revision and avoids the failure,
but accepts nothing. Its large values may be an important compensation-trap
signal; they cannot be treated as chi-square calibrated without a valid
posterior/coordinate model. It should remain a diagnostic or separately
calibrated graded penalty, not be threshold-tuned on these held-out outcomes.

## 5. R0.4-B mechanism

The controlled C2 term is

\[
r_{\tau,t}=-0.10\,\tau_{t-1}.
\]

It is outside the frozen explicit operator library. The previous command is
available in the observed trajectory but absent from the instantaneous state,
so the experiment cleanly distinguishes an unclosed state from insufficient
instantaneous model capacity.

The memoryless residual receives a 12-dimensional basis containing constants,
linear and quadratic state/action terms, cross terms, and nonlinear transforms.
The history residual receives the identical basis plus
`[tau[t-1], tau[t-2], tau[t-3]]`. Both use the same ridge estimator, 48 training
force samples, and 24 independent validation samples per episode. A local
utility gate enables a fallback only when its validation force MSE beats
physics by the declared margin.

## 6. R0.4-B results

Clean independent force-space error is:

| Predictor | Force RMSE | Fallback usage |
|---|---:|---:|
| Physics | `0.045742` | `0%` |
| Memoryless | `0.054646` | `100%` raw |
| History | `0.001340` | `100%` raw |
| Utility-gated memoryless | `0.045540` | `9.17%` |
| Utility-gated history | `0.001340` | `100%` |
| Oracle history | `1.19e-7` | `100%` |

Sequential rollout is:

| H | Physics | Memoryless | Utility memoryless | History / utility history | Oracle history |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.004928` | `0.006325` | `0.005066` | `0.000430` | `0.000393` |
| 4 | `0.009578` | `0.017061` | `0.010382` | `0.000944` | `0.000705` |
| 8 | `0.012470` | `0.028440` | `0.014082` | `0.001482` | `0.000925` |
| 16 | `0.016658` | `0.049951` | `0.020678` | `0.002427` | `0.001184` |

All methods have `100%` numerical stability and validity coverage in this
controlled window.

Adding observed history reduces force RMSE by `97.1%` relative to physics and
H16 state RMSE by `85.4%`. A high-capacity instantaneous basis not only fails
to close the discrepancy but amplifies rollout error. This supports the narrow
claim:

\[
\boxed{
\text{this recoverable delayed-force C2 is representation-limited,
not instantaneous-basis-limited}
}
\]

The qualification matters: this does not show that arbitrary latent hysteresis
can be reconstructed from a three-step window. It shows that when the missing
variable is encoded in observable history, exposing that history closes the
controlled C2 that a larger instantaneous model cannot close.

The memoryless local utility gate removes most harmful calls, but its `9.17%`
false-use rate still yields H16 `0.02068 > 0.01666` for physics. Therefore the
R0.4-A lesson also applies to fallback: **force-space utility is not itself a
rollout-utility certificate**.

## 7. Combined decision and next scientific question

| Required gate | Result | Status |
|---|---:|---|
| Revision H16 non-inferiority | `0.6390 > 0.3691` | **Fail** |
| Accepted revision stability | `98.125% < 99%` | **Fail** |
| C2 history fallback < physics | `0.00243 < 0.01666` | Pass |
| C2 history fallback < memoryless | `0.00243 < 0.04995` | Pass |
| C0 false revision | frozen R0.3 result `0%` | Inherited, not rerun |
| C1 exact recovery | frozen R0.3 result `100%` | Inherited, not rerun |

Because the revision gate fails, the combined decision remains:

\[
\boxed{\text{V6R0.4 overall no-go for RoboTwin}}
\]

The next experiment should not add operators, repeated revision, RGB, or IG.
It should isolate why H8-safe native revisions can fail at H16. Candidate
directions are longer/adversarial validation branches, invariant or energy
constraints, and regularized joint `(M, theta)` fitting that prevents the
revision compensation trap. Parameter displacement needs calibration on a
development cohort before it can become a meaningful penalty.

For fallback, the next integration should preserve the successful history
encoder but evaluate its utility in rollout space and include a no-discrepancy
control to measure false fallback use.

## 8. Reproduction

```bash
# Acceptance ablation / selection cohort
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r04a \
  --output runs/v6r04a/formal --device cpu \
  --seeds 2001 2011 2021 2031 2041 --episodes 24

# Fixed-policy confirmatory cohort
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r04a \
  --output runs/v6r04a/confirmatory --device cpu \
  --seeds 4001 4011 4021 4031 4041 --episodes 24

# History-aware C2 fallback
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r04b \
  --output runs/v6r04b/formal --device cpu \
  --seeds 3001 3011 3021 3031 3041 --episodes 24
```

The complete regression suite passes: `68 passed`.
