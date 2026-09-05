# APR-WM V3-min: Tangent-triggered Model-class Revision

## Executive conclusion

V3-min tests whether a hybrid world model can distinguish parameter-explainable error from evidence that its current physics family is inadequate, then either instantiate a more complex physics hypothesis or abstain to an `unknown` state.

The answer is positive but incomplete. On five held-out seeds:

1. tangent evidence reaches structural-detection AUROC `0.9957`, versus `0.7709` for residual magnitude;
2. it reduces false revision on adequate systems from `65.44%` to `0.92%`;
3. three-state model accuracy rises from `70.22%` to `89.61%`;
4. intervention RMSE falls from `0.1751` to `0.1641`, while the cost-regularized objective falls from `0.03812` to `0.03206`;
5. relative to never revising the linear model, tangent revision lowers intervention RMSE by `0.02322` (12.4%) and the total objective by `0.00303`;
6. nevertheless, the model-class oracle reaches `0.0290` intervention RMSE. Correctly declaring `unknown` is therefore not equivalent to discovering the missing dynamics.

The supported claim is:

> A residual component outside the local parameter tangent space is substantially better evidence for structural inadequacy than residual magnitude alone. It can trigger selective model revision without broadly upgrading adequate systems, but open-set detection by itself does not close the model-discovery gap.

## 1. Experimental scope

Every episode initially activates only the linear family

\[
M_0:\quad F(x,v)=kx+cv.
\]

The environment is sampled uniformly from three true classes:

\[
\begin{aligned}
M_{adequate}&: kx+cv,\\
M_{cubic}&: kx+cv+\alpha x^3,\\
M_{unseen}&: kx+cv+\gamma x^2\tanh(v/0.06).
\end{aligned}
\]

Only the cubic basis is available as a dormant structured revision proposal. The velocity-coupled class is absent from that proposal set. It tests open-set rejection and uses a generic fixed RBF residual as fallback.

This is therefore not unlimited open-world symbolic discovery. It tests two narrower abilities:

- activating a dormant structural hypothesis only after evidence warrants it;
- recognizing that none of the available structured hypotheses adequately explains an unseen discrepancy.

Thresholds were developed with seed 13. Formal statistics use five disjoint seeds `101/111/121/131/141`, each with 4,096 episodes.

## 2. Windowed tangent evidence

For a window of `N` interactions, define

\[
r=y-f_\theta(X),
\qquad
J_\theta=\frac{\partial f_\theta(X)}{\partial\theta}\in\mathbb R^{N\times 2}.
\]

The regularized projection is

\[
P_{\mathcal T}
=J_\theta(J_\theta^\top J_\theta+\rho I)^{-1}J_\theta^\top,
\]

with

\[
r_\parallel=P_{\mathcal T}r,
\qquad
r_\perp=(I-P_{\mathcal T})r.
\]

V3 uses

\[
S_M=\frac{\lVert r_\perp\rVert^2}{\lVert r\rVert^2+\epsilon}
\]

together with an absolute noise-floor constraint on `RMS(r_perp)`. Both are needed: the ratio checks direction, while the absolute term prevents small observation noise from being interpreted as structural evidence.

The projection is defined over a multi-observation window. For a single scalar force observation, the tangent/orthogonal distinction would be underdetermined and is not claimed.

## 3. Revision and abstention

After a trigger, the agent fits a cubic extension on the training portion of the evidence window and evaluates it on four held-out interactions. It selects:

- `linear`: no persistent tangent-orthogonal evidence;
- `cubic`: held-out loss improves enough and extension RMSE passes an acceptance threshold;
- `unknown`: misspecification is detected but the cubic proposal fails acceptance.

The `unknown` state invokes a generic 12-basis RBF residual fitted to observed residuals. The decision objective is

\[
\mathcal J
=L_{intervention}
+\lambda_M C_M
+\lambda_r C_r.
\]

Here `C_M=1` for an instantiated cubic model and `C_r=1` for the unknown-state fallback. The scalar prices (`0.006` and `0.01`) are controlled experiment utilities, not measured wall-clock or energy costs.

## 4. Baselines

- `linear_refit`: update `(k,c)` but never revise the model class;
- `magnitude_revision`: trigger the same revision mechanism from total residual RMS;
- `tangent_revision`: trigger from tangent-orthogonal evidence;
- `always_expand`: always attempt cubic revision or declare unknown;
- `oracle_class`: reveal the true family, including the held-out family, but estimate all coefficients from noisy observations.

All non-oracle strategies share the same data, proposal test, parameter fitting, residual fallback, costs, and thresholds. Thus magnitude versus tangent isolates the trigger representation.

## 5. Detection and revision results

| Strategy | Structural AUROC | False revision: adequate | Missed cubic | Unseen unknown recall | Three-state accuracy |
|---|---:|---:|---:|---:|---:|
| Linear refit | 0.9957† | 0% | 100% | 0% | 33.48% |
| Magnitude revision | 0.7709 | 65.44% | **12.50%** | 89.02% | 70.22% |
| Tangent revision | **0.9957** | **0.92%** | 19.49% | 89.08% | **89.61%** |
| Always expand | 0.9957† | 100% | 11.66% | 89.21% | 58.95% |
| Oracle class | — | 0% | 0% | 100% | 100% |

† These policies do not consume the tangent score; it is shown only as a common diagnostic. Oracle AUROC is omitted because oracle labels are supplied rather than inferred.

Paired tangent-minus-magnitude differences:

- structural AUROC: `+0.22482`, bootstrap CI `[+0.22014,+0.22979]`;
- adequate false revision: `−0.64518`, CI `[−0.65234,−0.63599]`;
- model-state accuracy: `+0.19395`, CI `[+0.18750,+0.20039]`;
- unseen recall: `+0.00062`, CI `[−0.00029,+0.00200]`.

The gain is not free. Tangent revision misses `6.99` percentage points more cubic episodes than magnitude, CI `[+6.28,+7.77]`. It is a conservative structural test: it rejects parameter-like errors and thereby sacrifices some sensitivity.

The tangent misspecification score is strongly rank-informative but not calibrated as a posterior: Brier is `0.1474` and ECE is `0.3421`. It must not be interpreted as `p(M_bad|D)` without a separate calibration model.

## 6. Prediction and resource objective

| Strategy | ID RMSE | Intervention RMSE | Adequate | Cubic | Unseen | Total objective |
|---|---:|---:|---:|---:|---:|---:|
| Linear refit | 0.03891 | 0.18731 | **0.02178** | 0.14757 | 0.28855 | 0.03509 |
| Magnitude revision | 0.01918 | 0.17513 | 0.11830 | **0.07345** | 0.26904 | 0.03812 |
| Tangent revision | **0.01900** | **0.16409** | 0.03290 | 0.08519 | **0.26899** | **0.03206** |
| Always expand | 0.01984 | 0.18238 | 0.14750 | 0.07184 | 0.26947 | 0.04186 |
| Oracle class | 0.00723 | 0.02898 | 0.02178 | 0.03146 | 0.03252 | 0.00482 |

Tangent versus magnitude:

- intervention RMSE: `−0.01104`, CI `[−0.01438,−0.00864]`;
- total objective: `−0.00606`, CI `[−0.00732,−0.00518]`.

Tangent versus linear-only:

- intervention RMSE: `−0.02322`, CI `[−0.02594,−0.02000]`;
- total objective: `−0.00303`, CI `[−0.00397,−0.00196]`.

The adequate-class comparison explains the objective result. Residual magnitude frequently confuses parameter-estimation error with structural error and raises adequate intervention RMSE from `0.0218` to `0.1183`. Always expanding is worse (`0.1475`). Tangent projection largely preserves the simple model where it is adequate (`0.0329`).

## 7. The new bottleneck: detection is not discovery

Tangent revision correctly sends `89.08%` of unseen episodes to `unknown`, yet their intervention RMSE remains `0.2690`; the oracle-family value is `0.0325`. The generic residual fallback recognizes neither the correct compositional form nor its parameter semantics.

Overall, oracle-class minus tangent intervention RMSE is `−0.13511`, CI `[−0.13858,−0.13253]`, an 82.3% relative reduction. The remaining gap contains:

- missed or rejected cubic revisions;
- imperfect unknown detection;
- most importantly, failure to turn an `unknown` declaration into the correct new structured hypothesis.

Thus V3 refines the V2 conclusion:

\[
\boxed{
\text{model inadequacy detection}
\neq
\text{model discovery}
}
\]

## 8. What is and is not supported

Supported in this controlled benchmark:

1. parameter-tangent geometry separates structural evidence from total error magnitude;
2. persistent orthogonal evidence can selectively trigger model revision;
3. a held-out proposal test prevents most adequate systems from being unnecessarily upgraded;
4. an explicit unknown state detects most truth outside the revision proposal set;
5. selective revision improves intervention prediction and a complexity-regularized objective.

Not established:

1. learning or synthesizing a new symbolic physics family;
2. calibrated Bayesian inference over an unbounded model space;
3. amortized tangent estimation for neural or high-dimensional physics models;
4. learned task-sensitive model, action, and compute costs;
5. long-horizon control, contacts, perception, or real hardware;
6. wall-clock savings from model revision;
7. robustness beyond the two constructed structural discrepancies.

The controlled V4-min experiment implements this next step with a finite operator library, active candidate discrimination, held-out acceptance, and proposal/selection oracles; see [`REPORT/REP/R0/V4_REPORT.md`](REPORT/REP/R0/V4_REPORT.md).

## Artifacts

- Raw held-out-seed results: [`strategies.csv`](runs/v3/core/strategies.csv)
- Aggregate statistics: [`strategies_aggregate.csv`](runs/v3/core/strategies_aggregate.csv)
- Paired differences: [`paired_differences.csv`](runs/v3/core/paired_differences.csv)
- Revision performance: [`revision_performance.png`](runs/v3/core/revision_performance.png)
- Revision confusion: [`revision_confusion.png`](runs/v3/core/revision_confusion.png)
- Complexity/error Pareto: [`revision_cost_pareto.png`](runs/v3/core/revision_cost_pareto.png)
- Run configuration: [`summary.json`](runs/v3/core/summary.json)
