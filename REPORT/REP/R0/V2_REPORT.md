# APR-WM V2: Misspecification-aware Epistemic Control

## Executive conclusion

V2 tests whether an agent can actively break the ambiguity between parameter uncertainty and structural model inadequacy. The answer is positive in a controlled compensation trap.

Five-seed results show:

1. a forced linear model learns an inadequate-class pseudo-true stiffness bias of `+0.3727`, while its adequate-class bias is statistically zero;
2. this compensating model has low ID force RMSE (`0.0256`) but high intervention RMSE (`0.2194`);
3. parameter-information probing improves model-class AUROC to `0.9155` and intervention RMSE to `0.1949`;
4. misspecification-aware joint probing reaches AUROC `0.9632` and intervention RMSE `0.1740`;
5. joint probing uses `1.814` probes on average, versus 2 for parameter probing and the fixed diagnostic action;
6. joint probing is statistically indistinguishable from the **action-only diagnostic oracle** in intervention RMSE while costing less;
7. a separate model-class oracle reaches intervention RMSE `0.0491`, showing that substantial value remains in resolving `M`, even after selecting the right diagnostic action;
8. joint probing's adequate-class residual misuse is `0.078%`; strict misuse conditional on an inaccurate parameter posterior is `0.93%`.

The central V2 claim is:

> Error compensation is locally predictive but interventionally brittle. A misspecification-aware epistemic policy can select actions that expose the hidden model-class discrepancy, improve future prediction, and make residual-compute allocation safer.

## 1. Two decision spaces

V2 does not treat physics, residual computation, and probing as three interchangeable actions.

The environment action is

\[
a_t\in\mathcal A
=\{a^{passive},a^{velocity},a^{amplitude}\}.
\]

The internal compute decision is

\[
r_t\in\mathcal R
=\{\text{physics only},\text{physics+residual}\}.
\]

The two are selected sequentially from a shared belief:

\[
b_t=p(\theta,M\mid D_{1:t}),
\]

where `θ=(k,c)` and `M` is adequate or nonlinear/misspecified. The action changes future evidence; the compute decision changes how the current prediction is produced. Probe cost and residual-compute cost are accounted for separately.

In this minimum version, the discrepancy is the class-conditioned basis `δ(x)=Mαx³`, so it is deterministic given `(M,x)` and is not a third independently inferred latent. A learned or nonparametric `p(δ|θ,M,D)` remains outside the present claim.

This minimum experiment holds task reward constant across candidate probes to isolate the epistemic term. It evaluates the resulting cost–prediction Pareto rather than claiming a complete control objective.

## 2. Compensation-trap dynamics

Ground-truth force is

\[
F(x,v)=kx+cv+M\alpha x^3,
\qquad \alpha=800.
\]

Passive context uses a narrow low-amplitude range `x≈0.018–0.024`. Locally,

\[
kx+\alpha x^3
\approx
(k+\alpha x_0^2)x,
\]

so a wrong linear model can absorb the cubic response into an effective stiffness. Amplitude probes use `x≈0.065–0.080`, where the cubic term becomes distinguishable. Velocity probes use large `|v|≈0.7–0.9` and are primarily informative about damping.

A forced-linear passive estimator provides direct pseudo-true evidence:

| True model class | Mean stiffness bias | Bootstrap 95% CI |
|---|---:|---:|
| Adequate | −0.0027 | [−0.0105,+0.0052] |
| Inadequate | **+0.3727** | [+0.3601,+0.3883] |

Its ID RMSE is `0.02557`, but intervention RMSE is `0.21938`; the gap is `0.19380`. Thus the wrong parameter is useful only near the passive observational distribution.

## 3. Belief and probing policies

The joint belief is a discrete particle approximation over two model classes, 45 stiffness values, and 17 damping values. Gaussian observation likelihoods update particle weights after every interaction.

Baselines:

- `linear_compensation`: force `M=adequate`, passive data only;
- `passive`: maintain the joint belief without probing or residual calls;
- `router_no_probe`: joint belief and compute policy, but no environment probe;
- `parameter_ig`: select actions for continuous-parameter information;
- `joint_ig`: combine normalized parameter IG with a weighted model-class IG;
- `oracle_probe`: always execute the maximum-amplitude diagnostic action, but still infer `M` from data;
- `full_oracle`: execute the same diagnostic action and reveal the true model class before prediction. Despite the code name, this is a **model-class oracle**, not an oracle-parameter estimator; `θ` is still inferred.

The V2 implementation uses a moment-matched information-gain surrogate. Parameter IG measures within-model predictive variance reduction. Model IG measures between-model predictive disagreement relative to within-model and observation variance. Joint score is

\[
\tilde{IG}_{joint}
=
\frac{IG_\theta}{\max_a IG_\theta}
+\beta_M
\frac{IG_M}{\max_a IG_M},
\qquad \beta_M=10.
\]

This is a misspecification-weighted approximation to

\[
I((\theta,M);x_{t+1}\mid D_t,a_t),
\]

not an exact mutual-information estimator. The normalization makes the intended multi-objective preference explicit rather than allowing the continuous parameter scale to silently dominate model-class information.

## 4. Physical action allocation

| Strategy | Velocity probes | Amplitude probes | Total probes | Model AUROC | Model Brier |
|---|---:|---:|---:|---:|---:|
| Passive joint belief | 0 | 0 | 0 | 0.5116 ± 0.0086 | 0.2499 ± 0.0002 |
| Parameter IG | 1.1505 | 0.8495 | 2.0000 | 0.9155 ± 0.0029 | 0.1171 ± 0.0018 |
| Joint IG | 0.00005 | 1.8144 | 1.8144 | **0.9632 ± 0.0021** | **0.0748 ± 0.0023** |
| Diagnostic oracle | 0 | 2.0000 | 2.0000 | 0.9629 ± 0.0022 | 0.0750 ± 0.0023 |
| Model-class oracle | 0 | 2.0000 | 2.0000 | 1.0000 | 0.0000 |

Amplitude actions also contain information about stiffness, so `parameter_ig` is not artificially forced to select only velocity probes. The empirical allocation follows its stated objective. Joint IG concentrates on model-discriminating amplitude probes and stops probing in about 18.6% of the second opportunities.

Per executed probe:

| Strategy | Parameter entropy reduction | Model entropy reduction |
|---|---:|---:|
| Parameter IG | **0.8929** | 0.1666 |
| Joint IG | 0.1790 | **0.2679** |
| Diagnostic oracle | 0.1588 | 0.2236 |

The policies allocate epistemic effort differently rather than one uniformly dominating all identification metrics. Parameter IG estimates damping much better (`c` RMSE `0.0981` versus joint IG `0.2931`), while joint IG is much better at detecting the wrong model class.

## 5. Compute allocation

The structural residual correction at a query is `δ=αx³`. Applying it is beneficial under the bad class and equally harmful under the adequate class. Net expected predictive utility is therefore

\[
\mathbb E[\Delta L]
=(2p(M_{bad})-1)\delta^2.
\]

The compute decision executes residual only when

\[
\mathbb E[\Delta L]>C_r
\quad\text{and}\quad
p(M_{bad})>0.99.
\]

This is a variable-cardinality decision, so cost-aware utility now changes the allocation policy; it is no longer equivalent to fixed top-k ranking as in V0.7.

| Strategy | Residual call rate | Structural utility captured | Adequate misuse | Strict misuse |
|---|---:|---:|---:|---:|
| Parameter IG | 0.0826 ± 0.0031 | 0.1640 ± 0.0053 | 0.039% | 0.216% |
| Joint IG | 0.1475 ± 0.0054 | **0.2924 ± 0.0091** | 0.078% | 0.933% |
| Diagnostic oracle | 0.1447 ± 0.0051 | 0.2881 ± 0.0090 | 0.068% | 0.826% |
| Model-class oracle | 0.4987 ± 0.0041 | **1.0000** | **0%** | **0%** |

The 0.99 confidence constraint makes compute allocation deliberately conservative. Consequently structural utility capture is below the fixed-budget V1 result. Joint IG has slightly higher misuse than parameter IG on the difficult posterior-miss subset, but the adequate-class difference is tiny and its paired confidence interval includes zero. The result should not be described as uniformly safer on every conditional subgroup.

## 6. Interventional performance and decision efficiency

| Strategy | ID RMSE | Intervention RMSE | Mean probes | Total decision cost |
|---|---:|---:|---:|---:|
| Linear compensation | 0.02557 | 0.21938 | 0 | 0 |
| Passive joint belief | 0.02586 | 0.22810 | 0 | 0 |
| Parameter IG | 0.02436 | 0.19495 | 2.000 | 0.04330 |
| Joint IG | **0.02279** | **0.17403** | **1.814** | **0.04219** |
| Diagnostic oracle | 0.02271 | 0.17384 | 2.000 | 0.04579 |
| Model-class oracle | 0.01567 | 0.04914 | 2.000 | 0.05995 |

Paired joint-IG minus parameter-IG differences:

- model AUROC: `+0.04769`, CI `[+0.04606,+0.05014]`;
- model Brier: `−0.04235`, CI `[−0.04456,−0.04061]`;
- intervention RMSE: `−0.02092`, CI `[−0.02223,−0.01961]`;
- probe count: `−0.18560`, CI `[−0.18901,−0.18247]`;
- total cost: `−0.00112`, CI `[−0.00126,−0.00097]`;
- adequate misuse: `+0.000385`, CI `[−0.000196,+0.001256]`.

Diagnostic-oracle minus joint-IG intervention RMSE is `−0.00019`, CI `[−0.00061,+0.00016]`; the accuracy difference is unresolved. The oracle uses `+0.18560` more probes and costs `+0.00360`. In this candidate action set, adaptive joint probing reaches the diagnostic action's predictive performance more efficiently.

This diagnostic oracle is not the epistemic upper bound: it knows the best probe shape but must still infer the hidden class. When the true model class is additionally revealed, intervention RMSE falls by `−0.12489` relative to joint IG, CI `[−0.12591,−0.12387]`, model AUROC becomes `1.0`, structural utility capture becomes `1.0`, and adequate-class misuse becomes zero. Its higher cost (`0.05995`) comes from two probes plus residual execution for every inadequate-class episode. The remaining joint-IG gap is therefore model-class decision uncertainty, not failure to identify the diagnostic action.

## 7. Calibration caution

Passive model ECE is numerically low (`0.0057`) despite AUROC near chance and Brier near `0.25`. This is because assigning probability 0.5 to every episode is marginally calibrated when class prevalence is 0.5, but it contains no discrimination. Model adequacy evaluation must report calibration together with Brier, AUROC, and decision performance.

## 8. What V2 supports

Supported in this controlled experiment:

1. a wrong physics class can induce a stable pseudo-true parameter that predicts locally but fails under intervention;
2. parameter-focused and model-focused probes produce different epistemic outcomes;
3. misspecification-aware joint probing improves model detection and intervention prediction over parameter-only probing;
4. environment action allocation and internal compute allocation can be optimized and charged separately;
5. variable-budget residual decisions make cost-aware utility nontrivial;
6. adaptive probing nearly matches a fixed diagnostic action with fewer probes;
7. knowing the correct probe is not equivalent to knowing the model class: the explicit model-class oracle exposes a large remaining epistemic gap.

Not established:

1. exact mutual-information optimization or learned action selection;
2. task/control reward during probing;
3. long-horizon dynamics or contact-mode changes;
4. amortized belief inference outside the discretized spring model;
5. robustness to an unenumerated model class;
6. parameter-tangent versus structural-orthogonal decomposition;
7. visual observations or real embodied systems.

The tangent-space proposal

\[
r=P_{\mathcal T_\theta}r+(I-P_{\mathcal T_\theta})r
\]

is evaluated in the controlled V3-min experiment; see [`REPORT/REP/R0/V3_REPORT.md`](REPORT/REP/R0/V3_REPORT.md). V2 itself intentionally did not use it.

## Artifacts

- Raw five-seed strategies: [`strategies.csv`](runs/v2/core/strategies.csv)
- Aggregated metrics: [`strategies_aggregate.csv`](runs/v2/core/strategies_aggregate.csv)
- Paired comparisons: [`paired_differences.csv`](runs/v2/core/paired_differences.csv)
- Compensation plot: [`compensation_brittleness.png`](runs/v2/core/compensation_brittleness.png)
- Epistemic cost Pareto: [`epistemic_cost_pareto.png`](runs/v2/core/epistemic_cost_pareto.png)
- Probe allocation: [`probe_allocation.png`](runs/v2/core/probe_allocation.png)
- Run summary: [`summary.json`](runs/v2/core/summary.json)
