# R1-RS2A Preregistration — Contact-Mediated Structural Identifiability

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R1/R1_RS2_REPORT.md` (`RS2_GO=false`, `RS2-C0_PASS=true`)  
Does not change: `RS1C_GO`, `RS1B_GO`, RS2-C0, \(s^\star\), thresholds, VoI, \(C_{\mathrm{tol}}\), RS1C policy  
Locks: **R1-RS2B** (consequence transport) is not opened here

## Placement

\[
\boxed{
RS2\text{-Formal GO=false}
\rightarrow
R1\text{-RS2A (this stage)}
\qquad
R1\text{-RS2B locked}
}
\]

Stage name:

\[
\boxed{
R1\text{-RS2A — Contact-Mediated Structural Identifiability}
}
\]

中文名：**接触介导条件下的结构可辨识性**。

This is a **diagnosis** of the RS2 allocation failure, not a retune of RS2.

## Frozen (do not reopen)

- \(D_0\) definition;
- Mode-A cell thresholds and \(C_{\mathrm{tol}}\);
- \(s^\star=2.0\) and script \(\to\) amp bins;
- RS1C decision policy, VoI, passivity, operator library;
- RS2-C0 geometry, scripts, and \(J_c^\top f_c\) accounting;
- no revision, no install, no H32, no accept bit.

RS2B (contact-mediated consequence transport) stays locked. C2 is excluded
from the primary matrix because Formal already showed \(D_0\) large and
`detect=1`; that is a \(C\)-transport problem, not a visibility problem.

## Scientific question

\[
\boxed{
\text{Why does matched operator exposure fail to preserve inadequacy
detectability under contact?}
}
\]

Operational question:

\[
\boxed{
\text{Is contact transfer governed by tangent-visible exposure rather
than raw operator exposure?}
}
\]

RS2 already established the policy-level fact:

\[
\boxed{
\text{RS1C decision policy does not transfer from direct excitation
to contact-mediated excitation}
}
\]

with C0 excluding \(J^\top f\) accounting error. RS2A asks whether that
non-transfer is explained by **parameter-tangent geometry**.

## Why not retune the threshold

Detector \(D_0\) is not RMS of the hidden residual \(r\). It is

\[
\boxed{
D_0=\|r_\perp\|=\|(I-P_{\mathcal T})r\|
}
\]

with \(\mathcal T=\mathrm{col}(J_\theta)\). Matching

\[
X_\phi=\mathbb E[v^4]
\]

need not match \(\|r_\perp\|\). The falsifiable claim is that contact
preserves operator exposure but not **structural visibility**.

## Definitions

Hidden operator signature on a discovery / probe window:

\[
\Phi=[\phi(v_1),\ldots,\phi(v_T)]^\top,\qquad \phi(v)=|v|v.
\]

Raw operator exposure (frozen RS1A.3):

\[
X_\phi=\frac1T\|\Phi\|^2=\mathbb E[v^4].
\]

Let \(J_\theta\) be the incumbent parameter tangent on the same window
(frozen two-column features: density-scaled \(\ddot q\), \(\dot q\)).
Ridge projector as in V3 / \(D_0\):

\[
\Phi_\perp=(I-P_{\mathcal T})\Phi.
\]

Tangent-visible exposure（参数切空间外可见激励量）:

\[
\boxed{
X_{\phi,\perp}=\frac1T\left\|(I-P_{\mathcal T})\Phi\right\|^2
}
\]

Structural visibility ratio（结构可见率）:

\[
\boxed{
\kappa_\perp=\frac{X_{\phi,\perp}}{X_\phi+\epsilon}
}
\]

If \(r\approx J_\theta\theta+\alpha\Phi\), then

\[
\boxed{
D_0\approx|\alpha|\sqrt{X_{\phi,\perp}}.
}
\]

## Hypotheses

### H1 — raw exposure transported on the adapter pair

The adapter was designed to match Mode-A \(1.5A_0\) with
`pull_release` at \(s^\star=2.0\). Primary pair:

\[
\text{contact: pull\_release @ }s^\star
\quad\text{vs}\quad
\text{Mode-A }1.5A_0.
\]

\[
\boxed{H1:\quad
\mathrm{rel.\ error}(\mathrm{median}\,X_\phi^{contact},X_\phi^{ModeA})<20\%
}
\]

on C1 pooled for this pair. This restates the adapter’s job; it is not a
new detector.

Report, but do not require, the Formal **bin-assigned** pairs
(`slow_pull`\(\leftrightarrow 1.0A_0\), `fast_pull`\(\leftrightarrow 1.5A_0\)).
Nearest-bin assignment is not equality.

### H2 — structural visibility does not transport

On the same primary pair, C1 only:

\[
\boxed{H2:\quad
\mathrm{median}\,\kappa_\perp^{contact}
<
\mathrm{median}\,\kappa_\perp^{ModeA}
}
\]

If H1 holds and H2 holds, contact absorbed more of \(\Phi\) into
\(\mathrm{col}(J_\theta)\).

### H3 — \(X_{\phi,\perp}\) predicts \(D_0\) across domains

On all C1 cells (both domains):

\[
\boxed{H3a:\quad
\rho(D_0,|\alpha|\sqrt{X_{\phi,\perp}})
>
\rho(D_0,|\alpha|\sqrt{X_\phi})
}
\]

Spearman \(\rho\). And

\[
\boxed{H3b:\quad
\mathrm{NRMSE}(D_0,|\alpha|\sqrt{X_{\phi,\perp}})
<
\mathrm{NRMSE}(D_0,|\alpha|\sqrt{X_\phi}).
}
\]

The scientific target is approximate cross-domain calibration

\[
D_0\approx|\alpha|\sqrt{X_{\phi,\perp}}
\]

on Mode-A and contact together. If that holds, detectability is governed by
excitation orthogonal to the incumbent model manifold — not by a
contact-specific threshold.

## Matrix

Regimes: \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\). **C2 excluded.**

Seeds: RS2 Formal seeds \(\{11101,11111,11121,11131,11141\}\) (diagnosis of
that experiment, not a new policy GO).

Mode-A: frozen sine \(f=0.20\,\mathrm{Hz}\), \(A/A_0\in\{1.0,1.5\}\).

Contact: frozen C0 scripts at scale \(1.0\), plus VoI extra
`pull_release` at \(s^\star=2.0\).

\[
\begin{aligned}
N_{\mathrm{ModeA}}&=5\times 3\times 2=30\\
N_{\mathrm{contact}}&=5\times 3\times 4=60\\
N&=90.
\end{aligned}
\]

Smoke: seed `11101` × C1-H × {Mode-A \(1.5A_0\), contact `fast_pull`}.
Smoke does not evaluate H1–H3.

## Logged every episode (no decision)

\[
X_\phi,\ X_{\phi,\perp},\ \kappa_\perp,\ D_0,\ |\alpha|\sqrt{X_\phi},\ |\alpha|\sqrt{X_{\phi,\perp}}.
\]

Secondary (not a detector, not a GO input):

- library Gram \(G_{ij}=\langle\phi_i,\phi_j\rangle\) on the probe window;
- \(\mathrm{cond}(G)\);
- correlation of `abs_v_v` with `signed_v2` and with the two tangent columns.

If contact raises collinearity of \(|v|v\) with \(v^2\) or with
\(J_\theta\), that supports “contact changes identifiability geometry, not
merely amplitude.” Do not promote this to a new statistic.

## GO

\[
\boxed{
RS2A\_GO = H2 \land H3a \land H3b
}
\]

H1 is reported. H1 failure on the adapter pair means exposure matching
itself did not hold under C1 dynamics; that is still diagnosis, not a
license to change \(s^\star\).

`RS2A_GO=true` **does not** set `RS2_GO=true` and **does not** authorize
threshold, adapter, or policy edits. It only supports the visibility
mechanism as the explanation of C1 detectability non-transfer.

## Failure interpretation

| Pattern | Read as |
|---------|---------|
| H1 ∧ H2 ∧ H3 | visibility geometry explains C1 miss |
| H1 ∧ ¬H2 | exposure matched but \(\kappa_\perp\) also matched; look elsewhere (noise, accounting leakage, windowing) |
| ¬H1 on adapter pair | C1 contact dynamics broke even raw \(X_\phi\) matching |
| H2 ∧ ¬H3 | \(\kappa_\perp\) drops but does not calibrate \(D_0\); incomplete mechanism |
| Gram cond / `signed_v2` collinearity high | secondary identifiability geometry; not a new detector |

None of these outcomes authorizes RS2 retuning or opening RS2B.

## Non-goals

- contact-specific \(D_0\) threshold search;
- changing \(s^\star\) or OSC scripts;
- VoI / \(C\) / passivity / H32;
- C2 consequence transport (RS2B);
- learned contact force, RGB, new operators.
