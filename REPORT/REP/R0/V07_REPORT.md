# APR-WM V0.7: Utility-supervised Compute Allocation

## Executive conclusion

V0.7 closes the final objective-level question left by V0.5/V0.6: should a router predict where a residual is large, where a synthetic regime says it is necessary, or where executing it actually improves the task loss?

Across five independently trained seeds, utility supervision is beneficial when the residual expert is sufficiently expressive, but it is not universally superior:

- with the 8× expert, utility routing captures `0.9678 ± 0.0067` of oracle-selected utility, versus `0.9424 ± 0.0316` for magnitude routing;
- the paired utility-minus-magnitude improvement is `+0.02537`, bootstrap 95% CI `[+0.00150,+0.05255]`;
- adaptive RMSE improves by `−0.00483`, CI `[−0.00950,−0.00050]`;
- with the 1× expert, every paired utility-versus-magnitude confidence interval crosses zero.

Thus the defensible claim is:

> Utility supervision can improve compute allocation beyond residual-magnitude or regime labels, especially when expert prediction quality makes realized per-edge value heterogeneous. It does not automatically help when the simpler targets already induce nearly the same ranking.

Paired CUDA timing also preserves the V0.6 compute boundary. B16/8× is clearly slower (`0.311×`), B128/8× remains statistically at the crossover (`1.131×`, CI `[0.981,1.226]`), B512/1× gives a mild speedup (`1.247×`), and B512/8× gives a clear speedup (`3.383×`).

## 1. Corrected lineage of router targets

The exact historical distinction matters:

- V0/V0.5 distills a soft target from the magnitude of the frozen/current residual message;
- V0.6 trains against the controlled binary nonlinear-regime label;
- V0.7 compares magnitude, binary necessity, and realized task utility under the same model class, training loss, data budget, and routing budget.

V0.7 freezes each expert before constructing router targets. The labels therefore cannot co-adapt with the router during training.

## 2. Utility definition

For residual target `y`, frozen expert prediction `r`, and an observable anisotropic task-sensitivity vector `w(e)`, per-edge utility is

\[
\Delta L(e)
=
\ell(0,y;w(e))-\ell(r,y;w(e)),
\]

where

\[
\ell(r,y;w)=\sum_d w_d(r_d-y_d)^2.
\]

The sensitivity weights are deterministic functions of current edge features. They create controlled cases in which a large correction is not necessarily the most task-relevant correction. The router sees those features, but it never sees the target or realized utility at evaluation time.

Because deployment uses exact top-10% allocation, all learned objectives use the same top-budget ranking-classification loss:

- `magnitude`: top 10% by `||r||`;
- `necessity`: controlled binary regime label, whose prevalence is 10%;
- `utility`: top 10% by realized `ΔL`;
- `oracle`: privileged evaluation-only ranking by realized `ΔL`.

This is a budget-conditioned ranking implementation of learning expected utility, rather than unrestricted scalar regression. A pilot showed that direct Smooth-L1 regression on the highly skewed utility distribution optimized the bulk of low-value edges and produced worse top-k allocation; aligning training with the deployment decision removed that mismatch.

## 3. Protocol

- Seeds: `13, 23, 33, 43, 53`.
- Interaction density: 80%.
- Residual budget among active edges: 10%.
- Experts: true-compute 1× and 8× V0.6 architectures.
- Expert training: 1,200 steps per seed.
- Router training: 1,000 steps per expert/objective/seed.
- Evaluation: 131,072 generated edges per seed.
- Router architecture: identical 513-parameter current MLP for every objective.
- Confidence intervals: nonparametric bootstrap intervals over five seed-level values.

The ordinary edge RMSE and the task-weighted loss are both reported. Utility is defined by the latter; ordinary RMSE remains an independent check that task weighting did not merely move error into an unreported coordinate.

## 4. Allocation results

Values are mean ± sample standard deviation over five seeds.

### 1× residual expert

| Strategy | Utility AUROC | Oracle utility captured | Adaptive RMSE | Δ task loss vs dense expert |
|---|---:|---:|---:|---:|
| Random | 0.5012 ± 0.0015 | 0.0938 ± 0.0020 | 0.10517 ± 0.00065 | +0.031089 ± 0.000745 |
| Magnitude | 0.9893 ± 0.0066 | 0.9448 ± 0.0268 | 0.03124 ± 0.00459 | −0.000401 ± 0.001429 |
| Necessity | 0.9927 ± 0.0034 | 0.9594 ± 0.0124 | 0.02858 ± 0.00283 | −0.000937 ± 0.000697 |
| Utility | 0.9922 ± 0.0016 | 0.9573 ± 0.0078 | 0.02895 ± 0.00192 | −0.000860 ± 0.000449 |
| Oracle | 1.0000 | 1.0000 | 0.01859 ± 0.00066 | −0.002440 ± 0.000557 |

Utility is numerically better than magnitude on average, but its paired differences are not statistically resolved:

| Utility minus magnitude metric | Mean | Bootstrap 95% CI |
|---|---:|---:|
| Utility AUROC | +0.00293 | [−0.00256, +0.00876] |
| Oracle utility captured | +0.01249 | [−0.01398, +0.03929] |
| Adaptive RMSE | −0.00229 | [−0.00732, +0.00274] |
| Δ task loss | −0.000458 | [−0.001433, +0.000516] |

### 8× residual expert

| Strategy | Utility AUROC | Oracle utility captured | Adaptive RMSE | Δ task loss vs dense expert |
|---|---:|---:|---:|---:|
| Random | 0.4991 ± 0.0029 | 0.0894 ± 0.0111 | 0.10527 ± 0.00068 | +0.031083 ± 0.002413 |
| Magnitude | 0.9881 ± 0.0064 | 0.9424 ± 0.0316 | 0.03095 ± 0.00678 | −0.000588 ± 0.001716 |
| Necessity | 0.9863 ± 0.0072 | 0.9241 ± 0.0401 | 0.03414 ± 0.00645 | +0.000093 ± 0.003728 |
| Utility | **0.9938 ± 0.0022** | **0.9678 ± 0.0067** | **0.02612 ± 0.00198** | **−0.001530 ± 0.002645** |
| Oracle | 1.0000 | 1.0000 | 0.01742 ± 0.00204 | −0.002727 ± 0.002576 |

Paired utility-minus-magnitude results:

| Metric | Mean | Bootstrap 95% CI |
|---|---:|---:|
| Utility AUROC | +0.00574 | [+0.00045, +0.01203] |
| Oracle utility captured | +0.02537 | [+0.00150, +0.05255] |
| Adaptive RMSE | −0.00483 | [−0.00950, −0.00050] |
| Δ task loss | −0.000942 | [−0.001940, −0.000056] |

Utility also beats binary necessity for the 8× expert: utility capture improves by `+0.04372`, CI `[+0.00951,+0.07665]`, and RMSE improves by `−0.00802`, CI `[−0.01413,−0.00158]`.

The 8× dense expert RMSE is `0.03259`, while utility-routed RMSE is `0.02612`. Sparse routing can outperform dense correction because it suppresses small erroneous expert outputs on simple edges; this is not evidence that fewer evaluations are intrinsically more accurate.

## 5. Multi-seed compute regimes

CUDA timing uses 100 paired samples per seed, each averaging 10 forwards after 30 paired warm-ups. Baseline and adaptive blocks alternate order on every sample, controlling GPU clock drift, thermal state, and cache/workspace warm-up. Speedup is the median of paired baseline/adaptive ratios within each seed, then aggregated across seeds.

The table reports the utility router. Other targets have the same architecture and execution count; their small latency differences are measurement/data-dependent variation rather than an algorithmic compute advantage.

| Regime | Mean speedup ± std | Bootstrap 95% CI | Interpretation |
|---|---:|---:|---|
| B16 / 8× | 0.311 ± 0.005 | [0.307, 0.316] | Clearly slower |
| B128 / 8× | 1.131 ± 0.164 | [0.981, 1.226] | Crossover unresolved |
| B512 / 1× | 1.247 ± 0.174 | [1.088, 1.346] | Mild speedup; one seed is below 1× |
| B512 / 8× | 3.383 ± 0.153 | [3.263, 3.503] | Clear speedup |

An earlier sequential timing implementation measured all baseline samples before all adaptive samples and moved B128/8× as high as `1.56×`. A component-level audit showed sensitivity to GPU state and execution order. The final paired protocol restores the point to a confidence interval that crosses break-even. This correction reinforces the V0.6 conclusion:

\[
\text{statistical sparsity}\not\Rightarrow\text{hardware efficiency}.
\]

## 6. Why a separate cost-aware router is redundant here

For one homogeneous expert tier, every selected edge has the same cost `C_E`. At a fixed top-k budget:

\[
\operatorname{rank}\!\left(\frac{\Delta L_i}{C_E}\right)
=
\operatorname{rank}(\Delta L_i)
\]

and

\[
\operatorname{rank}(\Delta L_i-\lambda C_E)
=
\operatorname{rank}(\Delta L_i).
\]

A regression test verifies both equivalences. Training another network with either transformed label would be mathematically redundant. Cost-aware allocation becomes a distinct problem only when at least one of the following is introduced:

1. heterogeneous expert tiers per edge;
2. state-dependent execution cost;
3. a variable route count with an abstention threshold;
4. a global budget shared across interactions, time, or modalities.

This is an important design constraint for V1/V2: compute cost must vary across the available actions before `ΔL/C` or `ΔL−λC` can change the allocation policy.

## 7. Supported claims and limits

Supported:

1. a router can learn realized task-utility ranking without seeing targets at deployment;
2. utility supervision improves allocation over magnitude and binary labels for the 8× expert across five seeds;
3. the improvement is not significant for the 1× expert;
4. actual speedup remains a hardware/expert/batch regime property under a stricter paired timing protocol;
5. cost-aware transformations are ranking-equivalent under constant cost and fixed cardinality.

Not yet established:

1. rollout-level utility, collision-timing utility, or task-reward utility;
2. utility learned without privileged next-state supervision;
3. robustness when physics parameters are unknown;
4. independent hardware-session confidence intervals or energy reduction;
5. heterogeneous expert selection where cost-aware utility becomes nontrivial.

## 8. Transition to V1

V1 should now address unknown physics rather than add visual observations. The minimal parameter set should be `θ=(k,c)` or `θ=(m,μ)`, with the physics expert using a posterior `p(θ|D_{1:t})` rather than ground truth. Evaluation must separate:

\[
e_{phy}=e_{model}+e_{param},
\]

where model-class inadequacy and parameter-estimation error are independently controlled. Router inputs should include posterior uncertainty, allowing the policy to distinguish “invoke residual” from “collect information/update system identification.” That distinction is the central V1 scientific question.

## Artifacts

- Raw five-seed evaluation: [`evaluation.csv`](runs/v07/core/evaluation.csv)
- Aggregated metrics: [`evaluation_aggregate.csv`](runs/v07/core/evaluation_aggregate.csv)
- Paired objective differences: [`paired_differences.csv`](runs/v07/core/paired_differences.csv)
- Raw paired timing: [`latency.csv`](runs/v07/core/latency.csv)
- Timing aggregates: [`latency_aggregate.csv`](runs/v07/core/latency_aggregate.csv)
- Figures: [`utility_capture.png`](runs/v07/core/utility_capture.png) and [`multiseed_speedup.png`](runs/v07/core/multiseed_speedup.png)
- Run summary: [`summary.json`](runs/v07/core/summary.json)
