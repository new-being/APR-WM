# APR-WM V0.5 Scientific Validation Report

## Executive conclusion

V0.5 confirms a real, reproducible **residual-necessity ranking signal**, but it also corrects the original compute-efficiency interpretation.

The strongest defensible conclusion is:

> In this controlled object-interaction world, an adaptive router can identify interactions where a learned residual is useful and preserve always-on residual accuracy across seeds and long rollouts. However, most graph edges are already removable by a trivial contact mask, the current router costs more analytical compute than that mask, and sparse execution of this very small residual MLP is slower than dense GPU execution.

Therefore V0/V0.5 is a mechanism-level proof of concept for **selective residual necessity**, not yet a proof of wall-clock compute savings.

## 1. Five-seed stability

Seeds: `7, 17, 27, 37, 47`; 4,000 optimization steps per learned model. Values are mean ± sample standard deviation. Confidence intervals are nonparametric bootstrap 95% intervals over seed means.

| Model | ID hard rollout RMSE | OOD hard rollout RMSE |
|---|---:|---:|
| Pure Physics | 0.04224 ± 0.00187 | 0.07092 ± 0.00304 |
| Pure Neural GNN | 0.02610 ± 0.00333 | 0.05944 ± 0.00633 |
| Always-on Residual | 0.01186 ± 0.00047 | 0.03320 ± 0.00209 |
| Adaptive | 0.01240 ± 0.00045 | 0.03306 ± 0.00204 |

Paired adaptive minus always-on residual differences:

- ID: `+0.000542 ± 0.000214`, bootstrap CI `[+0.000370, +0.000690]`;
- OOD: `-0.000134 ± 0.000295`, bootstrap CI `[-0.000355, +0.000087]`.

Thus adaptive has a small but stable ID penalty. On OOD, the confidence interval includes zero and the two methods are statistically indistinguishable at five seeds. OOD routing AUROC is `0.9674 ± 0.0074`; analytical learned-compute proxy is `0.10995 ± 0.00032`.

Raw statistics: [`aggregate.csv`](runs/v05/seeds/aggregate.csv) and [`paired_differences.csv`](runs/v05/seeds/paired_differences.csv).

## 2. Budget–accuracy Pareto curve

Budgets are fractions of currently contacting directed edges, evaluated at `0, 1%, 2%, 5%, 10%, 20%, 40%, 60%, 100%`. The experiment compares:

- random allocation;
- a category-free kinematic heuristic based on penetration, closing speed, tangential slip, and friction;
- learned router ranking;
- a privileged greedy oracle ranked by per-edge utility

\[
\Delta L_{ij}
=
\lVert a^{true}_{ij}-a^{phy}_{ij}\rVert^2
-
\lVert a^{true}_{ij}-a^{phy}_{ij}-a^{res}_{ij}\rVert^2.
\]

At 20% OOD contact budget:

| Strategy | Teacher-forced one-step RMSE | Rollout RMSE | Active graph edges |
|---|---:|---:|---:|
| Random | 0.01664 | 0.06398 | 1.00% |
| Heuristic | 0.00744 | 0.03391 | 0.97% |
| Learned | 0.00726 | 0.03257 | 0.96% |
| Greedy oracle | 0.00689 | 0.03218 | 0.96% |

The learned router is close to the oracle in teacher-forced one-step error and clearly outperforms random allocation. The greedy one-step oracle is not a trajectory-level oracle, so learned or heuristic routing can occasionally achieve lower autoregressive rollout error through error cancellation or compounding effects.

The important confound is contact sparsity: activating **all** contact edges executes the residual expert on only about `4.72%` of all graph edges and reaches OOD rollout RMSE `0.03106`. Excluding selection overhead, that is only `4.34%` of dense expert compute. The learned router itself costs about `7.96%` of the dense router-plus-expert FLOPs before any residual is selected. Consequently, the learned method improves allocation at equal activated-edge budget but does not beat contact filtering at equal total analytical compute in this toy environment.

Plots:

- [`pareto_rollout_allocation.png`](runs/v05/core/pareto_rollout_allocation.png)
- [`pareto_rollout_compute.png`](runs/v05/core/pareto_rollout_compute.png)
- [`pareto_one_step_allocation.png`](runs/v05/core/pareto_one_step_allocation.png)

Raw curve: [`pareto.csv`](runs/v05/core/pareto.csv).

## 3. Continuous complexity sweep

Material identity is fixed to `1.0` for every sample. Penetration and impact speed vary continuously over a `25 × 25` grid. Router probability versus physics-only edge error has Spearman correlation:

\[
\rho=0.9363.
\]

This supports the narrower claim that the router responds continuously to interaction regime/model mismatch rather than only reading a discrete material category. It does not prove causal complexity recognition under hidden physical parameters because the rest of the V0 state remains oracle.

Plot: [`continuum.png`](runs/v05/core/continuum.png); data: [`continuum.csv`](runs/v05/core/continuum.csv).

## 4. Error versus rollout horizon

OOD endpoint RMSE:

| Model | H=1 | H=5 | H=10 | H=20 | H=50 |
|---|---:|---:|---:|---:|---:|
| Physics | 0.07386 | 0.05454 | 0.05770 | 0.07838 | 0.15127 |
| Neural | 0.03387 | 0.04224 | 0.05020 | 0.07547 | 0.19860 |
| Always-on Residual | 0.01820 | 0.02065 | 0.02418 | 0.03418 | 0.07181 |
| Adaptive | 0.01789 | 0.02068 | 0.02418 | 0.03466 | 0.07413 |

Adaptive tracks always-on residual closely through horizon 50. Pure neural becomes worse than physics at long horizon despite better short-horizon accuracy, illustrating why one-step metrics alone are insufficient.

Plot: [`horizon.png`](runs/v05/core/horizon.png); data: [`horizon.csv`](runs/v05/core/horizon.csv).

## 5. True conditional execution and CUDA latency

The hard path was corrected so unselected edges never enter the residual MLP. A regression test records the actual expert input count. CUDA timing uses 100 samples, each averaging a block of 20 forwards after 50 warm-up iterations.

Batch 128 median latency:

| Execution mode | Expert-executed graph fraction | Median latency |
|---|---:|---:|
| Pure Physics | 0% | 0.656 ms |
| Dense Always-on Residual | 100% | 1.264 ms |
| Adaptive soft | 100% | 1.386 ms |
| Adaptive hard | 13.1% | 1.934 ms |
| Contact-all conditional | 19.1% | 2.157 ms |
| Heuristic 40% conditional | 7.6% | 2.187 ms |

Sparse execution is real, but it is slower. The expert has only about 13k parameters, so boolean indexing, compaction, small irregular GEMMs, and kernel launches dominate. This is expected on an RTX 5070 and invalidates any claim of an 89% wall-clock speedup.

Plot: [`latency.png`](runs/v05/core/latency.png); data: [`latency.csv`](runs/v05/core/latency.csv).

## 6. Router ablation

Single-seed OOD results:

| Variant | Hard rollout RMSE | Routing AUROC | Compute proxy |
|---|---:|---:|---:|
| Full | 0.03177 | 0.963 | 0.110 |
| No warm-up | 0.03261 | 0.966 | 0.110 |
| No distillation | 0.05286 | 0.873 | 0.110 |
| Soft-gate training | 0.03131 | 0.953 | 0.110 |
| 40% contact budget | 0.03423 | 0.945 | 0.100 |
| 80% contact budget | 0.03150 | 0.966 | 0.118 |

Distillation is essential for hard routing in this implementation. Warm-up helps moderately. Soft-gate training remains competitive after thresholded evaluation when combined with residual-need distillation, so the earlier soft-gate collapse was specifically caused by an unconstrained scaling symmetry, not softness alone. Budget 40% shows the expected accuracy trade-off; 80% nearly matches dense residual.

Raw results: [`ablation.csv`](runs/v05/ablations/ablation.csv).

## Revised research status

Supported:

1. physics mismatch has structured interaction-level variation;
2. residual utility can be learned and ranked substantially better than random;
3. the ranking generalizes OOD across five seeds;
4. sparse residual correction preserves long-horizon accuracy.

Not supported yet:

1. learned routing is more compute-efficient than a contact/kinematic heuristic;
2. analytical sparsity produces wall-clock GPU speedup;
3. the result holds without oracle object state/material;
4. the same residual sparsity exists in realistic robotics environments.

Before V1, the most informative next experiment is a denser interaction benchmark where many edges are physically active but only some require residual correction. The router should also be made cheaper (linear/tiny cascade after contact filtering), while the residual expert should be large enough for conditional execution to amortize packing and launch overhead. Accuracy should then be plotted against **measured latency**, not only analytical FLOPs.

