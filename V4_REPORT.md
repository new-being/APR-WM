# APR-WM V4-min: Residual-guided Active Operator Discovery

## Executive conclusion

V4-min asks what structure should replace a rejected physics model. It separates residual-guided operator proposal, diagnostic selection, and held-out revision acceptance while preserving an explicit `unknown` outcome.

Five held-out seeds support four conclusions:

1. tangent-orthogonal proposal obtains `98.99%` top-3 recall for the two missing operators represented in the library;
2. active diagnostic selection raises exact operator recovery from passive selection's `43.66%` to `60.45%`;
3. the proposal oracle improves exact recovery by only `0.38` percentage points, whereas the selection oracle raises it to `100%`;
4. active discovery reduces residual fallback from `65.24%` before discovery to `25.76%` afterward, while rejecting `90.80%` of truth outside the operator library.

The main V4 result is not that the operator library can fit the data. It is the bottleneck decomposition:

\[
\boxed{
\text{proposal is nearly solved}
\quad\text{but}\quad
\text{selection and acceptance remain limiting}
}
\]

V4 also closes the intended feedback loop:

\[
\text{physics}
\rightarrow
\text{residual evidence}
\rightarrow
\text{operator proposal}
\rightarrow
\text{diagnostic intervention}
\rightarrow
\text{revised physics}.
\]

## 1. Scope and dynamics

The initial model is linear:

\[
F_0(x,v)=kx+cv.
\]

Episodes are sampled uniformly from four true classes:

\[
\begin{aligned}
M_0 &: kx+cv,\\
M_{cubic} &: kx+cv+800x^3,\\
M_{coupled} &: kx+cv+95x^2\tanh(v/0.06),\\
M_{outside} &: kx+cv+0.24\,\mathbf 1[x>0.05]\sin(v/0.032).
\end{aligned}
\]

The finite operator library contains eight single-step proposals:

```text
x², x³, v², xv, |v|v,
position-dependent damping,
x² tanh(v/0.06), saturation
```

The cubic and velocity-coupled truth are in the library. The thresholded oscillatory term is deliberately absent. Every accepted model adds at most one operator:

\[
M'=M_0+\alpha\phi_j.
\]

Thresholds and costs were developed on seed 13. Formal statistics use disjoint seeds `201/211/221/231/241`, with 4,096 episodes per seed.

## 2. Proposal from tangent-orthogonal residual

V3's windowed projection produces

\[
r_\perp=(I-P_{\mathcal T_\theta})r.
\]

Each candidate operator is also residualized against the current parameter tangent space, yielding `φ_j,perp`. Its explanatory gain is

\[
G_j
=\lVert r_\perp\rVert^2
-\min_\alpha
\lVert r_\perp-\alpha\phi_{j,\perp}\rVert^2.
\]

The proposal score is

\[
S_j=G_j-\lambda_p C(\phi_j),
\]

and the top three candidates are retained. Residualizing the operators is important: ranking raw correlations can reward an operator for explaining error that a change in `(k,c)` could already absorb.

## 3. Active selection and held-out acceptance

For each triggered episode, all proposed models predict 32 candidate actions. V4 executes the two actions with the largest variance across candidate predictions. This is a model-disagreement surrogate for information gain, not an exact mutual-information calculation.

The candidate with the lowest diagnostic-probe error is selected. Proposal and selection are therefore separate:

\[
r_\perp
\rightarrow
\{M_1,M_2,M_3\}
\rightarrow
a^{probe}
\rightarrow
\hat M.
\]

Selection still does not imply revision. On eight independent held-out interactions, V4 accepts only when

\[
\Delta J
=L_{heldout}(M_0)
-L_{heldout}(\hat M)
-\lambda_M C(\hat M)
>\tau,
\]

and candidate RMSE is below a fixed absolute threshold. Otherwise it retains `unknown` and uses a bounded generic RBF residual fallback.

This version permits only one revision, so `revision_count` is reported but structural churn and hysteresis are not yet tested.

## 4. Baselines and diagnostic oracles

- `residual_only`: retain the linear model and use fallback after structural rejection;
- `passive_selection`: choose the highest proposal score without a diagnostic action;
- `raw_active_discovery`: active selection, but propose from raw residual correlation;
- `active_discovery`: orthogonal proposal plus active selection and held-out acceptance;
- `oracle_proposal`: force the correct in-library operator into top-3, but leave selection and acceptance unchanged;
- `oracle_selection`: directly provide the correct operator for in-library truth and the correct `unknown` decision outside the library. Coefficients are still estimated from noisy data.

The selection oracle is a structural-fidelity oracle. For outside-library truth it refuses a false explanation and uses residual fallback; it is not guaranteed to minimize short-term RMSE by opportunistic misspecification.

## 5. Proposal, selection, and acceptance decomposition

| Strategy | Top-3 proposal recall | Selection given proposal | Acceptance given correct selection | Exact recovery |
|---|---:|---:|---:|---:|
| Passive selection | 98.99% | 51.21% | 86.09% | 43.66% |
| Raw active | 82.30% | 77.04% | 81.72% | 51.82% |
| Orthogonal active | **98.99%** | 75.94% | 80.42% | **60.45%** |
| Oracle proposal | 100% | 76.07% | 79.97% | 60.83% |
| Oracle selection | — | 100% | 100% | 100% |

Active-minus-passive paired differences:

- selection accuracy given proposal: `+0.24727`, bootstrap CI `[+0.23355,+0.26099]`;
- exact recovery: `+0.16784`, CI `[+0.15488,+0.18306]`;
- diagnostic probes: `+1.31055` per episode, CI `[+1.29697,+1.32813]`.

The active policy's lower conditional acceptance (`−0.05678`) does not mean diagnostic data directly harms a fixed set of candidates. Active selection reaches a different and harder subset of correctly selected episodes; these conditional populations are not identical.

Orthogonal-minus-raw active proposal:

- top-3 recall: `+0.16689`, CI `[+0.16209,+0.17199]`;
- exact recovery: `+0.08632`, CI `[+0.07878,+0.09252]`;
- intervention RMSE: `−0.01334`, CI `[−0.01399,−0.01269]`.

This supports using `r_perp` as a discovery signal rather than merely as a rejection score.

## 6. Oracle bottleneck attribution

Oracle proposal versus learned orthogonal proposal changes:

- proposal recall: `+1.01` percentage points;
- exact recovery: only `+0.38` percentage points, CI `[+0.32,+0.44]`;
- in-library intervention RMSE: `−0.00120`, CI `[−0.00198,−0.00051]`.

Thus candidate generation is not the dominant bottleneck under this eight-operator library and top-3 budget.

Oracle selection versus oracle proposal changes:

- selection accuracy: `+0.23927`;
- correct-candidate acceptance: `+0.20032`;
- exact recovery: `+0.39172`, CI `[+0.38541,+0.39878]`;
- in-library intervention RMSE: `−0.06804`, CI `[−0.06976,−0.06631]`.

The remaining error is concentrated in discriminating correlated candidates and deciding whether finite noisy evidence is sufficient for permanent revision.

## 7. Prediction and resource results

| Strategy | Overall RMSE | In-library RMSE | Outside RMSE | Total objective |
|---|---:|---:|---:|---:|
| Residual only | 0.29133 | 0.27462 | 0.42915 | 0.09144 |
| Passive selection | 0.24437 | 0.16375 | 0.42512 | 0.06434 |
| Raw active | 0.20606 | 0.12931 | 0.36509 | 0.04977 |
| Orthogonal active | **0.19272** | **0.10708** | **0.35033** | **0.04415** |
| Oracle proposal | 0.19239 | 0.10588 | 0.35033 | 0.04402 |
| Oracle selection | 0.18423 | 0.03784 | 0.36102 | 0.04135 |

Relative to passive selection, active discovery reduces overall RMSE by `0.05165` (21.1%), CI `[−0.05229,−0.05116]`, and the resource-regularized objective by `0.02019`, CI `[−0.02049,−0.01984]`.

Relative to residual-only fallback, it reduces RMSE by `0.09861` (33.9%) and objective by `0.04729`.

The objective is

\[
\mathcal J
=L_{intervention}
+\lambda_a C_a
+\lambda_r C_r
+\lambda_M C_M.
\]

These are fixed experiment prices, not wall-clock latency or measured energy.

## 8. Residual as buffer and discovery signal

The tangent trigger sends `65.24%` of all episodes into temporary residual fallback before structural discovery. After active proposal, selection, and acceptance, fallback usage is `25.76%`, a relative reduction of 60.5%.

This is the intended role transition:

\[
\boxed{
\text{residual as temporary predictor}
+
\text{residual as structure-discovery evidence}
}
\]

Accepted explicit operators replace much of the generic fallback instead of merely being added alongside it forever.

## 9. Open-set refusal and a recurring negative result

For outside-library truth, active discovery reports `unknown` in `90.80%` of episodes and forces a library explanation in `4.13%`; the remaining approximately `5.9%` never cross the structural trigger and retain the linear model. Adequate systems have `0%` accepted incorrect expansions in all five formal seeds.

The structural-fidelity oracle has outside RMSE `0.3610`, slightly worse than active discovery's `0.3503`. This is not an oracle contradiction. The active method occasionally accepts a wrong operator that improves short-term prediction through approximation, while the oracle refuses to call that operator the true structure.

V4 therefore reproduces the V1 boundary in a new form:

\[
\boxed{
\text{structural fidelity}
\neq
\text{minimum short-term error under misspecification}
}
\]

Unknown-to-forced-explanation remains avoidable only by explicitly valuing epistemic correctness, not by RMSE alone.

## 10. What is and is not supported

Supported in this controlled benchmark:

1. tangent-orthogonal residual can rank a small structural operator library;
2. diagnostic intervention improves candidate selection and exact recovery;
3. complexity-aware held-out validation prevents accepted expansions on adequate systems;
4. proposal and selection oracles localize the dominant bottleneck;
5. explicit discovery substantially reduces persistent residual fallback;
6. operator recovery and outside-library refusal can coexist.

Not established:

1. unrestricted symbolic regression or generation of an operator absent from the library;
2. multi-operator composition, beam search, or grammar learning;
3. sequential posterior updates after each diagnostic probe;
4. revision hysteresis, exit decisions, or repeated model switching;
5. calibrated probabilities over candidate structures;
6. learned complexity/action/compute prices;
7. long-horizon control, visual perception, or real embodied data.

The next bottleneck is no longer proposal recall. V5-min therefore freezes the library and isolates candidate discrimination, sequential evidence, and acceptance under finite noisy interventions; see [`V5_REPORT.md`](V5_REPORT.md). Compositional operator generation is deferred until this discrimination boundary is understood.

## Artifacts

- Raw held-out-seed results: [`strategies.csv`](runs/v4/core/strategies.csv)
- Aggregated metrics: [`strategies_aggregate.csv`](runs/v4/core/strategies_aggregate.csv)
- Paired comparisons: [`paired_differences.csv`](runs/v4/core/paired_differences.csv)
- Discovery decomposition: [`discovery_decomposition.png`](runs/v4/core/discovery_decomposition.png)
- Per-class intervention RMSE: [`operator_intervention_rmse.png`](runs/v4/core/operator_intervention_rmse.png)
- Residual transition: [`residual_transition.png`](runs/v4/core/residual_transition.png)
- Cost/error Pareto: [`discovery_cost_pareto.png`](runs/v4/core/discovery_cost_pareto.png)
- Run summary: [`summary.json`](runs/v4/core/summary.json)
