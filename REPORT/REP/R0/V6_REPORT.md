# APR-WM V6: Adaptive Evidence Acquisition for Model Revision

## Executive conclusion

V6 freezes V5's operator library, controlled candidate coverage, and posterior-weighted three-probe selector. It studies only the final question in the revision loop:

> When is there enough independent evidence to replace the current physics explanation with the selected structure?

Five held-out seeds support four conclusions:

1. V5's high-noise acceptance collapse was partly real evidence scarcity and partly an uncalibrated absolute-RMSE threshold whose floor was below the observation noise;
2. revision confidence requires evidence that the selected model beats both the old physics and the runner-up candidate—beating the old model alone admits useful but structurally wrong operators;
3. a dual sequential plug-in Bayes-factor policy keeps mean false revision below `1%` at all tested noise levels while using `10.21–16.17` validation samples on average instead of 32;
4. bounded false revision is not free: at noise `0.10`, fixed-32 validation obtains more recovery but exceeds the target false-revision rate, while the sequential policy sacrifices power to stay below it.

The V6 result is therefore a constrained frontier:

\[
\boxed{
\min \mathbb E[N_{validation}]
\quad\text{subject to}\quad
P(\text{false revision})\lesssim 1\%
}
\]

It does not show that adaptive validation universally increases acceptance. It shows how validation power, revision safety, and evidence cost trade against one another.

## 1. Controlled scope

V6 retains the V5 dynamics and eight-operator dictionary. Cubic and velocity-coupled truth are represented; thresholded oscillatory truth remains outside the library. The in-library trigger and correct-candidate availability are controlled, and candidate selection is fixed to posterior-weighted disagreement with three diagnostic probes.

Every episode then receives a broad intervention-distribution validation stream of at most 32 observations. Validation is independent of fitting and selection data. Adequate and outside-library episodes are also passed through acceptance so false structural revision can be measured directly.

Thresholds were developed only on seed 13. Formal results use disjoint seeds `401/411/421/431/441`, 2,048 episodes per seed, and observation noise `0/0.025/0.05/0.10/0.15`.

This remains a controlled acceptance experiment, not end-to-end discovery.

## 2. A correction to the V5 acceptance interpretation

V5 required observed candidate RMSE below `0.095`. For noisy observations,

\[
\mathbb E[(y-\hat y)^2]
=
\mathbb E[(f-\hat f)^2]+\sigma_{obs}^2.
\]

At `σ=0.10`, even a perfect predictor has expected observed RMSE near `0.10`, so the fixed `0.095` threshold rejects it. V6 estimates structural excess error as

\[
\widehat{E}_{model}
=
\max\left(
\frac1N\sum_t(y_t-\hat y_t)^2-\sigma_{obs}^2,
0
\right).
\]

This uses known synthetic observation variance. In a real system, noise variance would itself need estimation and calibration.

At noise `0.10`, calibrating fixed-8 validation raises exact recovery from `3.20%` to `28.42%`, a paired increase of `+25.22` percentage points, CI `[+23.66,+26.74]`. But false revision also rises from `0.26%` to `3.41%`, CI for the increase `[+2.68,+3.66]` points.

Thus V5 correctly located acceptance as the operational bottleneck, but overstated pure evidence scarcity: part of the collapse was a dimensional calibration error. Correcting it restores power and simultaneously exposes the safety problem.

## 3. Acceptance contains two hypotheses

An early V6 development baseline tested only whether the selected structure beat the old linear physics. It accepted many wrong selected operators that improved prediction without being the true structure.

The final protocol therefore requires both:

\[
H_{old}:
L(M_{new})<L(M_{old})
\]

and

\[
H_{unique}:
L(M_{new})<L(M_{runner\text{-}up}).
\]

This distinguishes:

- **revision necessity**: the old explicit physics should change;
- **revision identity**: it should change to this candidate rather than another plausible candidate.

The generic residual is not treated as a structural hypothesis. It remains the prediction buffer when revision is rejected or deferred. This preserves its V3–V5 role as temporary correction and scientific evidence rather than allowing it to define structural truth.

## 4. Strategies

- `fixed8_raw`: eight validation samples and the original observed-RMSE threshold;
- `fixed8/16/32_calibrated`: fixed budgets with noise-corrected adequacy and dual old/runner-up improvement tests;
- `sequential_lcb`: checks dual lower confidence bounds every four samples;
- `sequential_bf`: checks dual plug-in predictive likelihood ratios every four samples;
- `always_accept`: unsafe upper bound on acceptance power;
- `oracle_acceptance`: accepts exactly the correctly selected in-library hypotheses.

For the BF-style strategy,

\[
\log BF_{old,t}
=
\sum_{s=1}^t
\log\frac{p(y_s\mid M_{new})}{p(y_s\mid M_{old})},
\]

\[
\log BF_{runner,t}
=
\sum_{s=1}^t
\log\frac{p(y_s\mid M_{new})}{p(y_s\mid M_{runner})}.
\]

The stopping statistic is the smaller of the two, after a complexity prior. It accepts above `6.9`, rejects below `−2.94` or when the noise-corrected adequacy test fails, and otherwise collects more evidence up to 32 samples.

Because parameters are plug-in posterior means and optional stopping is not handled with an e-process or a fully integrated marginal likelihood, this is a **Bayes-factor surrogate**, not a calibrated or anytime-valid Bayesian test.

## 5. Fixed-budget calibration and evidence quantity

| Noise σ | Strategy | Acceptance power | Exact recovery | False revision | Validation samples |
|---:|---|---:|---:|---:|---:|
| 0.05 | Fixed-8 raw | 42.58% | 35.48% | 0.30% | 8 |
| 0.05 | Fixed-8 calibrated | 49.47% | 41.23% | 0.48% | 8 |
| 0.05 | Fixed-32 calibrated | 53.98% | 44.99% | 0.17% | 32 |
| 0.10 | Fixed-8 raw | 4.48% | 3.20% | 0.26% | 8 |
| 0.10 | Fixed-8 calibrated | 39.61% | 28.42% | 3.41% | 8 |
| 0.10 | Fixed-32 calibrated | 44.29% | 31.78% | 1.66% | 32 |
| 0.15 | Fixed-8 calibrated | 33.81% | 23.17% | 5.75% | 8 |
| 0.15 | Fixed-32 calibrated | 33.76% | 23.13% | 2.55% | 32 |

At noise `0.05`, moving from 8 to 32 calibrated samples adds `+3.76` exact-recovery points, CI `[+3.06,+4.37]`, while reducing false revision by `0.32` points.

At noise `0.15`, extra evidence does not improve mean recovery (`−0.03` points; CI crosses zero), but it reduces false revision by `3.20` points, CI `[−3.57,−2.73]`. More validation data can therefore buy safety even when it no longer buys power.

## 6. Adaptive validation under the empirical 1% target

| Noise σ | Selection accuracy | BF acceptance power | BF exact recovery | BF false revision | Mean samples | Defer |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 100.00% | 100.00% | 100.00% | 0.82% | 11.39 | 24.44% |
| 0.025 | 92.53% | 85.46% | 79.08% | 0.85% | 13.99 | 20.97% |
| 0.050 | 83.34% | 54.22% | 45.20% | 0.60% | 16.17 | 30.85% |
| 0.100 | 71.76% | 25.58% | 18.35% | 0.47% | 12.09 | 17.93% |
| 0.150 | 68.49% | 14.08% | 9.65% | 0.07% | 10.21 | 10.89% |

All five formal point estimates satisfy the development target of mean false revision below `1%`. The bootstrap upper bounds at noise `0` and `0.025` are `1.21%` and `1.12%`, so these results do **not** establish a strict population-level 1% guarantee. Such a guarantee would require explicit calibration data and a valid sequential testing construction.

The non-monotonic sample count is important. High noise does not become easier: many hypotheses fail the adequacy or negative-evidence boundary early, so the policy rejects quickly. Mean evidence cost alone must therefore be read together with acceptance power and defer rate.

## 7. Evidence-cost and safety frontier

At noise `0.05`, sequential BF and fixed-32 have statistically indistinguishable exact recovery:

- sequential BF: `45.20%` recovery with `16.17` samples;
- fixed-32: `44.99%` recovery with `32` samples;
- paired recovery difference: `+0.21` points, CI `[−0.18,+0.61]`;
- sample reduction: `15.83`, CI `[15.44,16.18]`.

The sequential method's false revision is higher (`0.60%` versus `0.17%`) but remains below the empirical 1% target.

At noise `0.10`, fixed-32 reaches `31.78%` recovery but has `1.66%` false revision. Sequential BF reduces false revision to `0.47%` and samples from 32 to `12.09`, but recovery falls to `18.35%`. This is the clearest bounded-risk trade-off: the additional fixed-budget power is obtained by violating the target.

At noise `0.15`, sequential BF uses `10.21` samples and reaches only `9.65%` recovery, but false revision is `0.07%`. Fixed-32 reaches `23.13%` recovery with `2.55%` false revision.

No policy dominates across validation cost, power, and revision error.

## 8. Sequential BF versus sequential LCB

The LCB policy is consistently conservative and often evidence-inefficient. At noise `0.10`, BF relative to LCB:

- raises exact recovery by `+3.41` points, CI `[+3.01,+3.86]`;
- changes false revision by `−0.01` points, CI crossing zero;
- uses `5.52` fewer samples, CI `[5.18,5.77]`;
- reduces defer by `15.40` points.

At noise `0.15`, BF has slightly higher recovery (`+0.45` points), lower false revision (`−0.45` points), and uses `8.48` fewer samples.

At low noise, BF gains much more power but spends part of the false-revision allowance. This is why LCB conservatism should be assessed as a point on the frontier, not described as intrinsically safer or better.

## 9. Resource objective

V6 adds validation evidence cost to the previous resource objective:

\[
\mathcal J
=
L_{pred}
+\lambda_r C_r
+\lambda_a C_a
+\lambda_M C_M
+\lambda_v C_v.
\]

The fixed prices are experiment-level proxies, not measured energy or wall-clock cost. At noise `0.05`, sequential BF has objective `0.1112`, below fixed-32's `0.1198`, primarily because it halves validation cost with similar recovery. At noise `0.10–0.15`, strict safety leaves more systems on residual fallback, and this proxy no longer favors the sequential policy uniformly.

## 10. Scientific interpretation

V6 completes a more precise revision decomposition:

\[
\boxed{
\text{selection confidence}
\neq
\text{revision necessity}
\neq
\text{revision identity}
\neq
\text{validation sufficiency}
}
\]

It also changes the interpretation of “acceptance bottleneck.” The bottleneck is not a single threshold. It contains:

1. noise calibration;
2. evidence that old physics is inadequate;
3. evidence that the selected explanation is unique among plausible candidates;
4. a stopping decision under asymmetric false-revision and missed-revision costs.

Residual now has two stable roles:

\[
\boxed{
\text{Residual}
=
\text{temporary prediction buffer}
+
\text{evidence for self-revision}
}
\]

## 11. What is and is not supported

Supported in this controlled benchmark:

1. known-noise calibration removes a major artificial acceptance ceiling;
2. dual old/runner-up validation is necessary to reject predictively useful but structurally wrong revisions;
3. adaptive evidence acquisition can halve validation cost at moderate noise without reducing recovery significantly;
4. an empirical false-revision constraint changes which policy is preferred at high noise;
5. explicit reject/defer decisions preserve residual fallback when structural evidence is insufficient.

Not established:

1. a theoretical or population-level `1%` false-revision guarantee;
2. exact Bayes factors, calibrated model posteriors, or anytime-valid optional stopping;
3. unknown or heteroscedastic observation-noise estimation;
4. adaptive choice of validation actions rather than only validation duration;
5. repeated revision, rollback, hysteresis, or catastrophic model-update recovery;
6. unrestricted operator generation, compositional structures, long-horizon control, vision, or real robot data.

The next scientifically justified step is no longer another one-shot revision stage. It is repeated self-revision with rollback: test whether accepted structures remain stable under distribution shift, detect harmful revisions, and decide when to revert or reopen `unknown` without structural churn.

## Artifacts

- Raw held-out-seed results: [`strategies.csv`](runs/v6/core/strategies.csv)
- Aggregated metrics: [`strategies_aggregate.csv`](runs/v6/core/strategies_aggregate.csv)
- Paired comparisons: [`paired_differences.csv`](runs/v6/core/paired_differences.csv)
- Validation power: [`validation_power.png`](runs/v6/core/validation_power.png)
- Revision safety: [`revision_safety.png`](runs/v6/core/revision_safety.png)
- Validation-cost frontier: [`validation_cost_frontier.png`](runs/v6/core/validation_cost_frontier.png)
- Noise-calibration ablation: [`noise_calibration.png`](runs/v6/core/noise_calibration.png)
- Run summary: [`summary.json`](runs/v6/core/summary.json)
