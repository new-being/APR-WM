# APR-WM V5-min: Noisy Sequential Hypothesis Discrimination

## Executive conclusion

V5-min freezes V4's eight-operator library and top-3 proposal budget. It asks a narrower question: once the true in-library operator is available as a candidate, how should finite noisy interventions be allocated, accumulated, stopped, and accepted?

Five held-out seeds and five observation-noise levels support four conclusions:

1. posterior-weighted, uncertainty-normalized disagreement improves structural selection at moderate noise, but normalization alone does not;
2. a third fixed probe improves exact recovery, while sequential stopping saves substantial evidence at a measurable accuracy cost;
3. learned action selection is already close to the action oracle, so action optimization is not the dominant remaining bottleneck;
4. at noise `0.10–0.15`, candidate selection remains informative but held-out acceptance collapses, making acceptance—not proposal or action selection—the limiting stage.

The central result is therefore conditional rather than universal:

\[
\boxed{
\text{posterior-aware evidence allocation improves discrimination}
\quad\text{but}\quad
\text{finite-sample acceptance sets the high-noise recovery ceiling}
}
\]

LCB acceptance is a useful negative result. In this benchmark it rejects more correct revisions without producing a measurable reduction in already-negligible adequate-physics expansions.

## 1. Controlled scope

The dynamics, operator dictionary, complexity costs, and top-3 proposal budget are unchanged from V4. Cubic and velocity-coupled truth are in the dictionary; thresholded oscillatory truth remains outside it.

V5 deliberately conditions the in-library discrimination experiment on two controls:

- every in-library episode is sent to discrimination, even if the natural tangent trigger misses it;
- after recording natural top-3 recall, the correct operator is inserted if absent, giving controlled candidate coverage of `100%`.

Adequate and outside-library episodes retain the natural trigger. Consequently, exact recovery here is **not** end-to-end discovery accuracy. It measures selection and acceptance conditional on trigger/candidate availability for represented truth. The natural trigger and proposal metrics are reported separately so this control cannot be mistaken for improved proposal machinery.

Thresholds were developed on seed 13. Formal results use disjoint seeds `301/311/321/331/341`, with 2,048 episodes per seed. Each seed shares the same latent systems and action pools across noise levels; observation-noise draws are noise-level specific.

## 2. Candidate posterior and action scores

Each proposed structure has a Bayesian linear posterior over `(k,c,α)`. For action `a` and candidates `i,j`, V5 computes

\[
D_{ij}(a)=
\frac{(\mu_i(a)-\mu_j(a))^2}
{\sigma_i^2(a)+\sigma_j^2(a)+\sigma_{obs}^2}.
\]

The posterior-weighted score is

\[
S(a)=\sum_{i<j}p_i p_jD_{ij}(a).
\]

This is a pairwise information-gain surrogate, not exact mutual information. It favors actions that separate currently plausible candidates while discounting separation caused by parameter or observation uncertainty.

V5 compares raw prediction variance, unweighted normalized separation, posterior-weighted separation, one/two/three fixed probes, sequential stopping, an action oracle, and a structural-selection oracle.

After every probe, candidate parameter posteriors and categorical model probabilities are updated. Sequential variants stop when

\[
p_{(1)}>0.55,
\qquad
p_{(1)}-p_{(2)}>0.10,
\]

or defer after at most three probes.

## 3. Selection is separate from acceptance

The selected candidate is not automatically installed. Point-estimate acceptance uses held-out improvement, an absolute candidate-RMSE threshold, and operator complexity cost. The conservative variant uses a lower confidence bound on improvement relative to the incumbent fallback and also requires positive gain over the base physics model:

\[
\operatorname{LCB}(\Delta L)
=\bar{\Delta L}-z\operatorname{SE}(\Delta L)-\lambda_M C(M').
\]

Failure to select confidently produces `defer`; failure to validate a selected candidate produces `unknown`. This preserves selection, acceptance, and rejection as distinct decisions.

## 4. Natural proposal and controlled coverage

| Noise σ | Natural in-library trigger | Natural top-3 recall | Controlled candidate coverage |
|---:|---:|---:|---:|
| 0.000 | 59.13% | 100.00% | 100.00% |
| 0.025 | 66.20% | 99.92% | 100.00% |
| 0.050 | 83.27% | 99.11% | 100.00% |
| 0.100 | 30.74% | 87.66% | 100.00% |
| 0.150 | 10.71% | 75.72% | 100.00% |

The natural metrics show why the control is necessary. At high noise, an unconstrained sweep would silently mix trigger/proposal degradation with candidate discrimination. V5 does not claim to repair those natural declines.

## 5. Noise-aware discrimination

The table reports selection accuracy conditional on candidate availability, followed by exact recovery after acceptance.

| Noise σ | Raw-2 sel./exact | Normalized-2 sel./exact | Weighted-2 sel./exact | Weighted-3 sel./exact |
|---:|---:|---:|---:|---:|
| 0.000 | 100.00 / 99.86 | 100.00 / 99.86 | 100.00 / 99.80 | 100.00 / 99.84 |
| 0.025 | 87.70 / 87.11 | 87.16 / 86.74 | **90.64 / 90.16** | **92.48 / 91.99** |
| 0.050 | 78.21 / 74.88 | 76.26 / 72.77 | **80.55 / 76.49** | **82.69 / 79.42** |
| 0.100 | 66.91 / 19.15 | 66.28 / 19.06 | **70.52 / 20.16** | **73.06 / 22.30** |
| 0.150 | 63.45 / 2.48 | 62.30 / 2.26 | **64.66 / 2.32** | **68.27 / 2.76** |

At noise `0.025`, weighted-2 improves selection over normalized-2 by `+3.48` percentage points, paired bootstrap CI `[+2.73,+4.22]`, and exact recovery by `+3.42` points, CI `[+2.63,+4.20]`. At noise `0.05`, the gains are `+4.29` points, CI `[+3.77,+4.89]`, and `+3.72` points, CI `[+2.93,+4.65]`.

Unweighted normalization is not sufficient: it is slightly worse than raw disagreement at `0.025–0.15`. The gain comes from combining uncertainty normalization with posterior relevance, not from dividing by variance alone.

A third weighted probe adds `+2.94` exact-recovery points at noise `0.05`, CI `[+1.57,+4.52]`, but costs `+0.737` probes per episode. Evidence quantity therefore still matters.

## 6. Sequential evidence-cost frontier

| Noise σ | Weighted-3 exact | Sequential exact | Weighted-3 probes | Sequential probes | Sequential defer |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 99.84% | 99.80% | 2.138 | **0.736** | 0.00% |
| 0.025 | 91.99% | 85.28% | 2.160 | **0.791** | 0.09% |
| 0.050 | 79.42% | 71.07% | 2.212 | **0.906** | 1.40% |
| 0.100 | 22.30% | 16.01% | 1.859 | **0.958** | 10.37% |
| 0.150 | 2.76% | 1.93% | 1.598 | **0.938** | 18.51% |

At noise `0.05`, sequential stopping saves `1.306` probes per episode, CI `[1.293,1.315]`, but loses `8.35` exact-recovery points, CI `[7.37,9.35]`. Its entropy reduction per executed probe is `0.647`, versus `0.271` for weighted-3, a `2.39×` evidence-efficiency gain.

Sequential selection is therefore a Pareto option, not an accuracy-dominating replacement. Its current stopping thresholds are too aggressive when later evidence can reverse an early posterior lead.

## 7. Where the high-noise failure occurs

For weighted-3:

| Noise σ | Selection accuracy | Acceptance given correct selection | Exact recovery |
|---:|---:|---:|---:|
| 0.050 | 82.69% | 96.04% | 79.42% |
| 0.100 | 73.06% | 30.53% | 22.30% |
| 0.150 | 68.27% | 4.04% | 2.76% |

Selection remains well above chance as noise increases, but the eight-sample held-out acceptance test almost stops installing correct structures. This establishes a sharper bottleneck:

\[
\boxed{
\text{high-noise failure}
\approx
\text{insufficient acceptance evidence},
\quad
\text{not loss of candidate ranking alone}
}
\]

The action oracle supports the same attribution. Relative to learned sequential action selection, at noise `0.05` it changes selection by only `+0.25` points and exact recovery by `+0.27` points; the exact-recovery CI crosses zero. Even at noise `0.15`, its `+3.34` selection-point advantage becomes only `+0.04` exact-recovery points because acceptance is binding.

## 8. LCB acceptance is conservative without a safety payoff

At noise `0.05`, adding LCB acceptance to the same sequential selector reduces exact recovery from `71.07%` to `68.22%`, a paired change of `−2.85` points, CI `[−3.25,−2.46]`. At zero noise it reduces recovery by `7.97` points.

Adequate-physics incorrect expansion is already `0%` for both variants through noise `0.05` and at most `0.04%` in the full sweep. LCB raises outside-library rejection conditional on a natural trigger only from `95.85%` to `96.13%` at noise `0.05`.

Thus the current LCB does not improve the relevant safety frontier. It double-counts limited-sample uncertainty relative to an already strict absolute-RMSE test and should not be adopted as the default merely because it is statistically conservative.

## 9. Unknown preservation and the detection boundary

Outside-library rejection must be decomposed into natural detection and conditional rejection:

| Noise σ | Natural outside trigger | Unknown rejection, all outside | Unknown rejection given trigger |
|---:|---:|---:|---:|
| 0.000 | 86.07% | 81.43% | 94.60% |
| 0.025 | 88.95% | 84.56% | 95.07% |
| 0.050 | 95.03% | 91.08% | 95.85% |
| 0.100 | 48.27% | 47.69% | 98.82% |
| 0.150 | 14.43% | 14.39% | 99.74% |

Once an outside episode reaches discrimination, the explicit `unknown` outcome is preserved and becomes more conservative with noise. The collapsing unconditional rate at high noise is instead a trigger miss: most outside systems retain the current linear model without entering revision. These are operationally different failures and require different remedies.

## 10. Prediction versus structural recovery

At noise `0.05`, weighted-3 achieves the best non-oracle exact recovery (`79.42%`) and lowers in-library intervention RMSE to `0.0710`, compared with raw-2's `0.0760`. Its extrapolation RMSE is also lower (`0.1835` versus `0.2030`).

However, structural selection gains do not translate monotonically into overall short-horizon RMSE at every noise level. The structural oracle can also have worse overall RMSE than a non-oracle because it refuses outside-library approximations that happen to predict locally. V5 therefore retains the V1/V4 distinction:

\[
\boxed{
\text{structural fidelity}
\neq
\text{minimum short-term predictive error under misspecification}
}
\]

Residual fallback remains a buffer for rejected or outside hypotheses. At noise `0.05`, weighted-3 reduces structural fallback from `73.56%` before discrimination to `25.41%` afterward; at noise `0.15`, strict acceptance leaves it almost unchanged (`52.99%` to `51.17%`).

## 11. What is and is not supported

Supported in this controlled benchmark:

1. posterior-weighted normalized disagreement improves noisy candidate ranking;
2. additional evidence improves recovery, while sequential stopping exposes a measurable cost/accuracy frontier;
3. action selection is not the dominant bottleneck under the current action pool;
4. selection and acceptance fail at different noise regimes and must be evaluated separately;
5. explicit defer/unknown decisions prevent forced structural explanations after a revision trigger;
6. LCB acceptance is not automatically better than a point-estimate test.

Not established:

1. end-to-end high-noise discovery without controlled trigger/candidate availability;
2. calibrated categorical model probabilities or exact information gain;
3. optimal sequential stopping or adaptive acquisition cost;
4. acceptance with an adaptive evidence budget;
5. unrestricted operator generation or multi-operator composition;
6. long-horizon control, vision, or real embodied data.

The justified next step is not a larger operator library. V6 therefore freezes proposal and selection, then studies adaptive held-out evidence under explicit false-revision and missed-revision costs; see [`V6_REPORT.md`](V6_REPORT.md).

## Artifacts

- Raw held-out-seed results: [`strategies.csv`](runs/v5/core/strategies.csv)
- Aggregated metrics: [`strategies_aggregate.csv`](runs/v5/core/strategies_aggregate.csv)
- Paired comparisons: [`paired_differences.csv`](runs/v5/core/paired_differences.csv)
- Noise discrimination: [`noise_discrimination.png`](runs/v5/core/noise_discrimination.png)
- Evidence-cost frontier: [`evidence_cost.png`](runs/v5/core/evidence_cost.png)
- Acceptance/open-set safety: [`acceptance_safety.png`](runs/v5/core/acceptance_safety.png)
- Natural-versus-controlled proposal: [`proposal_control.png`](runs/v5/core/proposal_control.png)
- Run summary: [`summary.json`](runs/v5/core/summary.json)
