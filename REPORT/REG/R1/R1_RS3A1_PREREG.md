# R1-RS3A.1 Preregistration — Cross-Fitted Structural Excess-Risk Evidence

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R1/R1_RS3A_REPORT.md`  
Does not change: `RS3A_GO=false`, `RS2_GO=false`, `RS2A_GO`, `RS2B_GO`,
`RS1C_GO`, C0, \(s^\star\), \(D_0\), \(C_{\mathrm{tol}}\), RS1C policy  
Locks: **RS3B**, **RS3C**, and any further local-null whitening of \(S_\perp\)

## Placement

\[
\boxed{
RS3A\_GO=false
\rightarrow
R1\text{-RS3A.1 (this stage)}
\qquad
RS3B\text{/}C\text{ locked}
}
\]

Stage name:

\[
\boxed{
R1\text{-RS3A.1 — Cross-Fitted Structural Excess-Risk Evidence}
}
\]

中文名：**交叉拟合的结构性超额风险证据**。

This is the last strong attempt at an **intervention-free scalar**
structural-evidence ruler at this representation. It is not a runtime
detector, not a contact threshold, and not RS3C.

## Why not RS3A.1 as smarter whitening

RS3A refuted

\[
\text{local H0 whitening of }r_\perp
\Rightarrow
\text{stable evidential meaning across }\mathcal I.
\]

Forbidden here: matched H0 length, Welch/\(F\)/\(\chi^2\) edits, robust
\(\hat\sigma_\perp\), quieter contact `phase`. Those still ask

\[
\text{how abnormal is residual relative to an }\mathcal I\text{-local null?}
\]

## Scientific question

\[
\boxed{
\text{Does extra frozen structural capacity reduce held-out predictive
risk by a similar amount under Mode-A and contact, without domain ID?}
}
\]

中文：不要问 residual 相对 local noise 有多异常；问 **增加结构容量后，
held-out 预测风险能减少多少**，且该减少量在两种干预下是否同语义。

Success is **operating-point transport**, not contact AUROC/recall.

If this also fails, the intended scientific close is that a domain-free
scalar ruler may not exist at this level, and the programme should turn
to \(\pi(E,C,V\mid\mathcal I)\). That close is **not** executed in this
stage.

## Learner-visible statistic

From the **probe / excitation window only** (no quiet-H0 construction),
draw a common budget

\[
K=K_f+K_h=100+100=200
\]

indices without replacement. If \(n_{\mathrm{probe}}<K\), use all points
and a half/half split. Two-fold cross-fit: each half is \(D_{\mathrm{fit}}\)
once and \(D_{\mathrm{hold}}\) once.

### \(M_{\mathrm{par}}\)

On \(D_{\mathrm{fit}}\), fit only incumbent parameters \(\hat\theta\)
(same `_fit_base_posterior` / tangent as frozen RS1A). Held-out loss

\[
L_{\mathrm{par}}=\mathrm{mean}\big((y-\hat y_{\hat\theta})^2;D_{\mathrm{hold}}\big).
\]

### \(M_{\mathrm{flex}}\)

Sequential, not a new physics operator: keep the same \(\hat\theta\), then
fit the **frozen V3 12-RBF** generic residual on the \(D_{\mathrm{fit}}\)
parameter residual, with Door-box centres (not V3 disk-world centres):

\[
q\in\{-0.4,0.0,0.4,0.8\},\quad
v\in\{-0.8,0.0,0.8\},\quad
\sigma_q=0.25,\ \sigma_v=0.45.
\]

Ridge \(2\times10^{-3}\) (V3/V6 `residual_ridge`). Prediction clamped to
\([-0.4,0.4]\). Same architecture, capacity, ridge, clamp, and \(K\) on
every episode. No domain ID, no true \(\phi\).

\[
L_{\mathrm{flex}}=\mathrm{mean}\big((y-\hat y_{\hat\theta}-\hat r_{\mathrm{RBF}})^2;D_{\mathrm{hold}}\big).
\]

\[
E_{\mathrm{XR}}
=
\frac{L_{\mathrm{par}}-L_{\mathrm{flex}}}{L_{\mathrm{par}}+L_{\mathrm{flex}}+\epsilon}
\in[-1,1]
\]

averaged over the two folds.

Meaning:

- \(E_{\mathrm{XR}}\approx0\): extra capacity does not stably help held-out;
- \(E_{\mathrm{XR}}\gg0\): the incumbent class leaves held-out error that
  this frozen extra capacity can explain.

Log frozen \(D_0\) on the full probe as a comparator only. Do not use
\(S_\perp\) as a candidate score. Oracle \(X_{\phi,\perp}\) may be logged,
not gated.

## Matrix

New seeds, held out from RS3A / RS2:

\[
\{13101,13111,13121,13131,13141\}.
\]

Regimes \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\). **C2 excluded.**
Mode-A \(A/A_0\in\{1.0,1.5\}\); contact four frozen jobs including
`pull_release@\(s^\star\)`. \(N=90\).

Smoke: seed `13101` × C0 × {Mode-A \(1.5A_0\), contact `fast_pull`}. No GO.

## Hypotheses / GO

C0 is the **negative control** against a memorizer / contact-nuisance
residual.

### H1a — C0 excess-risk near zero (both domains)

\[
\boxed{
\begin{aligned}
\mathrm{median}\,E_{\mathrm{XR}}(C0,\mathrm{ModeA}) &\le 0.15\\
\mathrm{median}\,E_{\mathrm{XR}}(C0,\mathrm{contact}) &\le 0.15
\end{aligned}
}
\]

If contact C0 is large, \(M_{\mathrm{flex}}\) is fitting contact nuisance,
not structural inadequacy.

### H1b — shared threshold specificity (not a second fitted ruler)

\(\tau_E=\) Mode-A C0 sample maximum of \(E_{\mathrm{XR}}\) (99% quantile
at \(n=10\)).

\[
\boxed{
\begin{aligned}
\mathrm{FPR}(E_{\mathrm{XR}}>\tau_E\mid C0,\mathrm{ModeA}) &\le 0.01\\
\mathrm{FPR}(E_{\mathrm{XR}}>\tau_E\mid C0,\mathrm{contact}) &\le 0.05
\end{aligned}
}
\]

H1b alone is **not** evidence-scale transport (RS3A H1 lesson). H1a
blocks that over-read.

### H2 — operating-point transport (adapter pair, C1)

On C1, Mode-A \(1.5A_0\) vs `pull_release@\(s^\star\)`, same \(g(\cdot)\)
as RS3A:

\[
\boxed{H2:\quad g(E_{\mathrm{XR}})<g(D_0)}
\]

Not an AUROC/recall gate. Report \(g(D_0)\) for the same cells.

\[
\boxed{RS3A.1\_GO = H1a \land H1b \land H2}
\]

`RS3A.1_GO=true` does not set `RS3A_GO` or `RS2_GO`, does not unlock
RS3B/C, and does not authorize domain-specific \(\tau_E\).

## Failure interpretation

| Pattern | Read as |
|---------|---------|
| all pass | excess-risk semantics transport better than \(D_0\) amplitude |
| ¬H1a | frozen RBF is a nuisance/memorizer at this \(K\); stop |
| H1a ∧ ¬H1b | C0 medians look small but contact tail exceeds Mode-A \(\tau_E\) |
| H1 ∧ ¬H2 | C0 control holds; C1 operating points still split by \(\mathcal I\) |
| AUROC high, H2 fail | discrimination without transport; do not promote |

If GO is false, **do not** open RS3A.2 with another scalar. The
scientifically honest next question would be intervention-conditioned
epistemics, under a new preregistration, not more rulers.

## Non-goals

- smarter \(S_\perp\) / local null repair;
- RS3B \(R_{\mathrm{abs}}\) / RS3C policy;
- C2 / latch;
- true \(\phi\) or domain ID in the score;
- expanding RBF capacity per domain.
