# R1-RS3A Report — Transport-Calibrated Structural Evidence

Date: 2026-08-16  
Prereg: `REPORT/REG/R1/R1_RS3A_PREREG.md`  
Depends on: `REPORT/REP/R1/R1_RS2_STAGE_FREEZE.md`  
Artifacts: `runs/r1_rs3a/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `f6fe5aeed260f9f4828727360a07049259e1694f4500a4f5f754825c5270e26e`

## Decision

\[
\boxed{RS3A\_GO=\mathrm{false}}
\]

\[
\boxed{
\text{local H0-whitened surprisal }S_\perp\text{ does not transport
evidential meaning across Mode-A and contact}
}
\]

This does **not** set `RS2_GO=true`. It does **not** rewrite `RS2A_GO` or
`RS2B_GO`. It does **not** unlock RS3B, RS3C, or a domain-specific
\(\tau_S\). \(X_{\phi,\perp}\) remains oracle-only.

Smoke (`runs/r1_rs3a/smoke/`) is plumbing only. The 90-episode matrix is
the first run allowed to set `RS3A_GO`.

## Question (unchanged)

\[
\boxed{
\text{Can we construct a learner-visible structural-evidence variable
whose evidential meaning is stable across intervention mechanisms?}
}
\]

The success notion is **calibration / transport of meaning**, not
single-domain AUROC or contact recall. The statistic is forbidden from
seeing domain ID or the true missing operator.

## Statistic (learner-visible)

Frozen \(D_0\) pipeline unchanged. Same-episode H0 window
(Mode-A: passive context; contact: `phase<2`) gives

\[
\hat\sigma_\perp^2=\mathrm{mean}(r_{\perp,\mathrm{H0}}^2),\qquad
T_\perp=\frac{\|r_{\perp,\mathrm{probe}}\|^2}{\hat\sigma_\perp^2+\epsilon}.
\]

\[
p_\perp=P(\chi^2_\nu\ge T_\perp\mid H_0),\qquad
S_\perp=-\log p_\perp,\qquad
\nu=\max(n_{\mathrm{probe}}-2,1).
\]

A single \(\tau_S\) is the Mode-A C0 sample maximum of \(S_\perp\).

## Matrix

New seeds \(\{12101,12111,12121,12131,12141\}\).
Regimes \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\). **C2 excluded.**
Mode-A \(A/A_0\in\{1.0,1.5\}\); contact four frozen jobs including
`pull_release@\(s^\star=2.0\)`. \(N=90\). No domain ID in the score.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 null calibration transport | **PASS** | Mode-A C0 FPR \(0/10\le0.01\); contact C0 FPR \(0/20\le0.05\) at the **same** \(\tau_S=690.78\) |
| H2 evidence operating-point gap | **FAIL** | adapter C1: \(g(S_\perp)=2.00\not< g(D_0)=0.699\) |

\[
\boxed{RS3A\_GO=H1\land H2=\mathrm{false}}
\]

H1 passing does **not** mean the two C0 clouds sit on the same scale.
\(\tau_S\) is the Mode-A C0 maximum (operational 99% quantile at \(n=10\)).
Contact C0 \(S_\perp\) never reaches that ceiling.

## What the numbers actually say

Adapter pair, C1 only (Mode-A \(1.5A_0\) vs `pull_release@\(s^\star\)`):

| Domain | median \(D_0\) | median \(S_\perp\) | typical \(n_{\mathrm{H0}}\) | typical \(n_{\mathrm{probe}}\) |
|--------|---------------:|-------------------:|----------------------------:|-------------------------------:|
| Mode-A | \(2.28\times10^{-3}\) | \(554\) | \(8\) | \(5000\) |
| contact | \(1.10\times10^{-3}\) | \(0\) | \(800\) | \(250\)–\(350\) |

C0 (all Mode-A amps / all contact jobs):

| Domain | median \(D_0\) | median \(S_\perp\) | \(S_\perp\) range |
|--------|---------------:|-------------------:|------------------:|
| Mode-A | \(2.01\times10^{-3}\) | \(414\) | \(0\)–\(690.78\) |
| contact | \(5.06\times10^{-4}\) | \(0.020\) | \(0\)–\(1.38\) |

Relative gap \(g(u)=|m_A-m_c|/(0.5(|m_A|+|m_c|)+\epsilon)\).
Whitening **widened** the Mode-A/contact gap relative to raw \(D_0\).

This matches the preregistered reading

\[
\boxed{H1\land\lnot H2:\ \text{C0 threshold transport is not enough;
}S_\perp\text{ on C1 is still domain-dominated}}
\]

## Scientific freeze (what was actually refuted)

The GO is a negative result and stays frozen. It refutes a **specific**
hypothesis, not the whole transport-calibrated-evidence programme:

\[
\boxed{
\text{intervention-local H0 whitening of }r_\perp
\not\Rightarrow
\text{an intervention-stable evidence ruler}
}
\]

\[
\boxed{\text{local normalization}\neq\text{semantic normalization}}
\]

H1 must not be over-read. The shared \(\tau_S=690.78\) yields C0 FPR
\(0/10\) and \(0/20\), but the C0 clouds are not on one scale
(\(\mathrm{median}\,S_\perp^{\mathrm{ModeA}}=414\) vs
\(\mathrm{median}\,S_\perp^{\mathrm{contact}}=0.020\)):

\[
\boxed{
\text{same decision at one threshold}
\not\Rightarrow
\text{same evidential scale}
}
\]

This is the same logic as RS2B: occasional agreement of a decision bit
does not mean the numerical quantity transports.

**Forbidden RS3A.1-as-whitening:** do not continue with matched H0 length,
Welch/\(F\)/\(\chi^2\) edits, robust variance, or a quieter contact phase.
Those would fit a common ruler to two interventions. The open follow-up,
if any, must change the **evidence semantics**, not the local null.

## Locked next stages

RS3B / RS3C remain locked. The only authorized follow-up is
**R1-RS3A.1** (`REPORT/REG/R1/R1_RS3A1_PREREG.md`): cross-fitted structural
excess-risk \(E_{\mathrm{XR}}\). If that also fails to transport, stop
inventing intervention-free scalars.

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
RS1C\_GO &= true\\
RS2\text{-C0\_PASS} &= true\\
RS2\_GO &= false\\
RS2A\_GO &= true\\
RS2B\_GO &= true\\
RS3A\_GO &= false
\end{aligned}
}
\]
