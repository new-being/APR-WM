# APR-WM V1 Plan: Unknown Physics Without Visual Confounds

## Scientific question

> Can interaction-level physics–residual decomposition remain identifiable and compute-useful when the explicit physics parameters are uncertain?

V1 should isolate this question before adding perception or active exploration.

## Minimal scope

Start with spring stiffness and damping:

\[
\theta=(k,c).
\]

These parameters directly control the existing base force law and avoid simultaneously changing object representation, graph construction, and observation quality. Mass/friction can be a second ablation after the mechanism is stable.

The predictor is

\[
\hat s_{t+1}
=F_{phy}(s_t,a_t;\hat\theta_t)
+g_tF_{res}(e_t),
\]

with

\[
q_\phi(\theta\mid D_{1:t})
=\mathcal N(\mu_{\theta,t},\operatorname{diag}(\sigma^2_{\theta,t})).
\]

Router inputs include current interaction features, predicted physics error features, posterior mean, and posterior uncertainty.

## Required error decomposition

Construct a controlled 2×2 evaluation:

| Physics class | Parameters | Purpose |
|---|---|---|
| Adequate | Oracle | irreducible numerical/reference floor |
| Adequate | Estimated | isolate parameter error `e_param` |
| Inadequate | Oracle | isolate model-class error `e_model` |
| Inadequate | Estimated | realistic combined case |

This makes the decomposition operational:

\[
e_{phy}=e_{model}+e_{param}+e_{interaction}.
\]

The additive expression is diagnostic rather than an assumption of orthogonality; interaction terms should be measured explicitly by the 2×2 design.

## Baselines

1. physics with oracle parameters;
2. physics with point-estimated parameters;
3. physics with posterior parameters;
4. posterior physics + always-on residual;
5. posterior physics + utility router without uncertainty input;
6. posterior physics + utility router with uncertainty input;
7. privileged router labeled separately by model-class and parameter-error utility.

The comparison between 5 and 6 tests whether uncertainty changes allocation rather than merely improving prediction features.

## Preventing residual takeover

The residual must not become an unrestricted patch for poor system identification. Use:

- an identification-only warm-up with the residual disabled;
- alternating or staged updates for the posterior and residual expert;
- stop-gradient from residual loss into the parameter posterior in the primary experiment;
- residual magnitude/call regularization;
- held-out parameter recovery metrics independent of next-state RMSE;
- an ablation allowing joint gradients to quantify takeover rather than silently assuming it away.

## Router decisions

V1 should initially keep a fixed residual budget so it remains comparable to V0.7. Report two privileged utilities during analysis:

\[
U^{model}_{ij}
=L(F_{phy}(\theta^*))-L(F_{phy}(\theta^*)+F_{res}),
\]

\[
U^{param}_{ij}
=L(F_{phy}(\hat\theta))-L(F_{phy}(\theta^*)).
\]

They distinguish “residual helps despite correct parameters” from “the parameter estimate is wrong.” The learned router does not receive these privileged labels at deployment.

Variable-budget decisions such as “residual versus gather information” belong to V2, because only then does cost-aware utility change the action ranking nontrivially.

## Metrics

- parameter posterior RMSE and negative log likelihood;
- posterior interval coverage/calibration;
- one-step and rollout state RMSE;
- model-utility and parameter-error AUROC;
- selected residual utility and oracle-utility capture;
- residual calls caused by model inadequacy versus parameter uncertainty;
- paired CUDA latency in the V0.7 slower/crossover/faster regimes;
- stability over at least five seeds.

## Minimal experiment sequence

1. Static hidden `(k,c)` per trajectory, passive context, adequate physics class.
2. Add controlled nonlinear model inadequacy while keeping `(k,c)` hidden.
3. Test uncertainty-aware routing and residual-takeover ablations.
4. Only after V1 succeeds, allow actions to change information gain in V2 active system identification.

## V1 success criterion

V1 succeeds if posterior-aware utility routing preserves the V0.7 accuracy–compute frontier while measurably separating model-class residual demand from parameter uncertainty. Lower next-state error alone is insufficient if parameter recovery collapses or the residual absorbs parameter-estimation mistakes.
