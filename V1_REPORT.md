# APR-WM V1: Unknown Physics, Posterior Calibration, and Residual Misuse

## Executive conclusion

V1 demonstrates that replacing oracle physics parameters with an estimate does not merely add noise to the V0.7 routing problem. It creates an identifiability conflict between structural model error and parameter error.

Across five seeds:

- the Bayesian `(k,c)` posterior is calibrated when the explicit physics class is correct: nominal 90% marginal coverage is `90.55% ± 0.77%`;
- under structural misspecification, the same posterior becomes severely overconfident: coverage falls to `17.40% ± 0.93%` and calibration error rises from `1.04%` to `63.85%`;
- a posterior-full router captures `95.09% ± 0.45%` of privileged structural utility and reduces strict residual misuse from `10.09%` to `0.80%`;
- nevertheless, its total force RMSE using estimated physics is `0.3542`, worse than the state-only router's `0.3092`;
- with oracle parameters, the ordering reverses: posterior-full RMSE is `0.1912`, substantially better than state-only at `0.2760`.

The apparent contradiction is the central result. In the inadequate class, the misspecified parameter posterior absorbs part of the nonlinear force into biased `(k,c)` estimates. That parameter bias cancels some structural error. A decomposition-safe residual restores the true structural correction and can therefore worsen short-horizon prediction when combined with the biased physics estimate.

The strongest defensible V1 claim is:

> Unknown physics creates a conflict between predictive error minimization and identifiable physics–residual decomposition. Posterior context can sharply reduce residual misuse and recover structural utility, but calibrated uncertainty under the assumed model does not remain calibrated under model misspecification.

## 1. Controlled unknown-physics setup

Each trajectory samples hidden parameters

\[
k\sim U(20,50),\qquad c\sim U(0.6,1.8).
\]

The linear explicit model observes interaction features

\[
x=(\text{penetration},-\text{normal velocity})
\]

and predicts

\[
F_{phy}=x^\top\theta,\qquad \theta=(k,c).
\]

Each trajectory supplies 4, 8, 16, or 32 noisy context interactions. A conjugate Bayesian linear regression posterior computes

\[
p(\theta\mid D)=\mathcal N(\mu_\theta,\Sigma_\theta).
\]

Half of the trajectories use an adequate linear physics class. The other half add a state-dependent nonlinear force above a penetration threshold. Structural class and `(k,c)` are sampled independently.

The residual expert receives query state, posterior mean and standard deviation, context residual RMS, and context count. It is supervised only with

\[
F_{true}-F_{phy}(\theta^*),
\]

not with the error under `μθ`. Therefore parameter-estimation error is excluded from the residual target by construction. Expert and posterior are not jointly optimized, preventing gradient-based residual takeover in this experiment.

## 2. Router variants and utility definitions

All learned routers have a fixed 10% global edge budget:

1. `state_only`: current interaction features only;
2. `posterior_mean`: state plus posterior mean;
3. `posterior_full`: state, mean, uncertainty, predictive residual diagnostic, and context count;
4. `total_utility`: posterior-full inputs, trained against utility under estimated physics.

The first three optimize structural utility:

\[
U_{struct}
=
L(F_{phy}(\theta^*))
-L(F_{phy}(\theta^*)+F_{res}).
\]

The total-predictive comparator optimizes

\[
U_{total}
=
L(F_{phy}(\mu_\theta))
-L(F_{phy}(\mu_\theta)+F_{res}).
\]

`structural_oracle` and `total_oracle` use realized privileged utility only for evaluation. The distinction makes the interpretability–prediction tradeoff measurable rather than rhetorical.

## 3. Parameter calibration

Values are mean ± sample standard deviation over five independently generated and trained seeds.

| Physics class | k RMSE | c RMSE | 90% coverage | Calibration error | Posterior NLL |
|---|---:|---:|---:|---:|---:|
| Adequate | 0.7226 ± 0.0287 | 0.2306 ± 0.0023 | 0.9055 ± 0.0077 | 0.0104 ± 0.0033 | 0.621 ± 0.022 |
| Inadequate | 11.0561 ± 0.1507 | 1.1359 ± 0.0390 | 0.1740 ± 0.0093 | 0.6385 ± 0.0098 | 336.73 ± 12.06 |

Calibration error is the mean absolute gap between observed and nominal marginal coverage at 50%, 80%, 90%, and 95%. The posterior is correct under its assumed likelihood, so adequate-class calibration is close to ideal. Structural misspecification violates that likelihood and produces biased, falsely confident parameter estimates.

This establishes that `Σθ` is not, by itself, a reliable measure of total epistemic uncertainty: it represents uncertainty conditional on the model class being correct.

## 4. The 2×2 error decomposition

Physics-only force errors are evaluated under adequate/inadequate model classes and oracle/estimated parameters:

Define the factorial cells as

\[
E_{00}=E(\text{adequate class},\text{oracle parameters}),\quad
E_{10}=E(\text{adequate class},\text{estimated parameters}),
\]

\[
E_{01}=E(\text{inadequate class},\text{oracle parameters}),\quad
E_{11}=E(\text{inadequate class},\text{estimated parameters}).
\]

The factorial interaction is exactly

\[
I=E_{11}-E_{10}-E_{01}+E_{00}.
\]

It only reduces to `E11−E10−E01` in this controlled force benchmark because `E00=0`. Future observation, numerical, or perception error can make `E00` nonzero.

| Model class | Parameter mode | Force RMSE |
|---|---|---:|
| Adequate | Oracle | 0.0000 |
| Adequate | Estimated | 0.0425 ± 0.0013 |
| Inadequate | Oracle | 0.6363 ± 0.0050 |
| Inadequate | Estimated | 0.4302 ± 0.0032 |

On the MSE scale:

| Component | Mean ± std | Bootstrap 95% CI |
|---|---:|---:|
| Parameter-only error | 0.00181 ± 0.00011 | [0.00173, 0.00190] |
| Structural-only error | 0.40493 ± 0.00632 | [0.40044, 0.41038] |
| Combined error | 0.18505 ± 0.00271 | [0.18283, 0.18720] |
| Interaction term | **−0.22169 ± 0.00476** | [−0.22539, −0.21800] |

Therefore the useful decomposition is not generally additive:

\[
e_{combined}
=e_{param}+e_{struct}+e_{interaction}.
\]

Here the interaction is strongly negative because the biased linear parameters partially fit the omitted nonlinearity. Writing only `e_phy=e_structural±e_param` hides the dominant phenomenon.

## 5. Residual allocation and misuse

Strict misuse is defined as

\[
P(\text{route}=1\mid
\text{adequate model class and }\theta^*
\text{ outside the posterior 90% interval}).
\]

The broader adequate-class misuse rate is also reported.

| Router | Structural utility AUROC | Structural utility capture | Adequate misuse | Strict misuse | Misuse share of calls |
|---|---:|---:|---:|---:|---:|
| State only | 0.9427 ± 0.0026 | 0.7250 ± 0.0117 | 0.0996 ± 0.0013 | 0.1009 ± 0.0023 | 0.0880 ± 0.0035 |
| Posterior mean | 0.9725 ± 0.0023 | 0.8071 ± 0.0090 | 0.0507 ± 0.0042 | 0.0539 ± 0.0053 | 0.0469 ± 0.0033 |
| Posterior full | **0.9940 ± 0.0009** | **0.9509 ± 0.0045** | **0.0090 ± 0.0029** | **0.0080 ± 0.0028** | **0.0070 ± 0.0022** |

Paired posterior-full minus state-only differences:

- structural utility capture: `+0.22589`, CI `[+0.21986,+0.23151]`;
- adequate misuse: `−0.09060`, CI `[−0.09190,−0.08881]`;
- strict misuse: `−0.09280`, CI `[−0.09593,−0.09060]`.

Posterior uncertainty plus a model-mismatch diagnostic therefore provides a strong, stable separation signal. This does not mean posterior variance alone detects model inadequacy; the ablation bundles variance with predictive residual and context size, while `posterior_mean` supplies the no-uncertainty comparison.

## 6. Decomposition fidelity versus prediction RMSE

| Router | Structural capture | RMSE with estimated θ | RMSE with oracle θ |
|---|---:|---:|---:|
| State only | 0.7250 | **0.3092** | 0.2760 |
| Posterior mean | 0.8071 | 0.3466 | 0.2486 |
| Posterior full | **0.9509** | 0.3542 | **0.1912** |
| Structural oracle | 1.0000 | 0.3414 | 0.1673 |
| Total oracle | 0.4519 | **0.2566** | 0.3522 |

Posterior-full versus state-only is unambiguous but changes sign with parameter mode:

- estimated-physics total RMSE: `+0.04501`, CI `[+0.04318,+0.04748]` (worse);
- oracle-parameter RMSE: `−0.08472`, CI `[−0.08600,−0.08344]` (better).

The total oracle achieves the best estimated-physics RMSE by selecting corrections that cooperate with parameter bias. It captures only `45.19%` of structural utility and has much worse oracle-parameter RMSE. It is a predictive oracle, not a decomposition oracle.

The learned total-utility router reaches total-utility AUROC `0.8654` but does not capture positive total utility reliably: mean normalized capture is `−0.2355`. This is a negative result. Realized parameter error is only partly inferable from the posterior summary, and a top-k classifier trained on its utility does not approach the privileged total oracle in this setup.

## 7. Scientific interpretation

V1 answers the original question in three parts:

1. **Can parameter uncertainty be calibrated?** Yes, when the assumed physics class is adequate.
2. **Can uncertainty-aware routing avoid using residual for parameter error?** Yes; strict misuse falls by roughly 9.3 percentage points and structural allocation improves sharply.
3. **Does decomposition-safe routing minimize immediate prediction error under a biased posterior?** No. Bias–structure cancellation can make an identifiable correction look harmful under the estimated model.

This means “residual misuse” cannot be judged from prediction RMSE alone. A model may achieve lower RMSE precisely because its parameter estimator and residual decomposition are both wrong in compensating directions.

## 8. Limits and V2 implications

V1 is still an interaction-force benchmark. It does not yet establish:

- state rollout behavior or contact-mode transitions;
- a learned/amortized posterior under rich dynamics;
- calibration robust to model misspecification;
- an active action that chooses identification rather than residual compute;
- transfer to visual or robotic observations.

V2 should introduce a decision among at least three actions:

\[
\{\text{physics only},\text{invoke residual},\text{take informative action/update ID}\}.
\]

At that point costs and outcomes are heterogeneous, so cost-aware utility becomes nontrivial. A robust posterior or explicit model-class latent variable is also required; merely passing `Σθ` from a misspecified Bayesian model is insufficient.

## Artifacts

- Raw routing metrics: [`routing.csv`](runs/v1/core/routing.csv)
- Paired routing differences: [`routing_paired_differences.csv`](runs/v1/core/routing_paired_differences.csv)
- Parameter metrics: [`posterior_aggregate.csv`](runs/v1/core/posterior_aggregate.csv)
- 2×2 conditions: [`conditions_aggregate.csv`](runs/v1/core/conditions_aggregate.csv)
- Interaction decomposition: [`decomposition_aggregate.csv`](runs/v1/core/decomposition_aggregate.csv)
- Figures: [`parameter_calibration.png`](runs/v1/core/parameter_calibration.png), [`error_decomposition.png`](runs/v1/core/error_decomposition.png), and [`decomposition_prediction_tradeoff.png`](runs/v1/core/decomposition_prediction_tradeoff.png)
- Run summary: [`summary.json`](runs/v1/core/summary.json)
