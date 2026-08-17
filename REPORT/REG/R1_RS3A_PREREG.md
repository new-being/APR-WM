# R1-RS3A Preregistration — Transport-Calibrated Structural Evidence

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R1_RS2_STAGE_FREEZE.md`  
Does not change: `RS2_GO=false`, `RS1C_GO`, `RS2A_GO`, `RS2B_GO`, C0, \(s^\star\),
\(D_0\) definition, \(C_{\mathrm{tol}}\), RS1C policy  
Locks: **RS3B** (typed consequence) and **RS3C** (policy integration)

## Placement

\[
\boxed{
RS2\text{ frozen}
\rightarrow
R1\text{-RS3A (this stage)}
\qquad
RS3B\text{/}C\text{ locked}
}
\]

Stage name:

\[
\boxed{
R1\text{-RS3A — Transport-Calibrated Structural Evidence}
}
\]

中文名：**跨干预语义校准的结构证据**。

This constructs a **learner-visible** evidence variable. It is not a contact
threshold fit, not an RS2 rescue, and not a new RS1C.

## Why not \(X_{\phi,\perp}\) as the detector

RS2A’s \(X_{\phi,\perp}\) and \(\kappa_\perp\) require the true missing
operator \(\phi(v)=|v|v\). They stay **oracle diagnostics only**. They may
explain visibility; they must not enter the learner statistic.

## Scientific question

\[
\boxed{
\text{Can we construct a learner-visible structural-evidence variable
whose evidential meaning is stable across intervention mechanisms?}
}
\]

中文：能否构造一个完全由 learner-visible 数据得到的结构错误证据量，使
“这个数值代表多强的模型不充分证据”在 direct torque 与 robot contact 下
具有相同含义？

The target is **calibration / transport of meaning**, not single-domain
discrimination. RS1A already showed \(D_0\) is strong inside Mode-A.

\[
\boxed{
\text{same }S_\perp
\Rightarrow
\text{approximately same evidential meaning across }\mathcal I
}
\]

is the success notion, not “contact recall went up.”

## Learner-visible statistic (no domain ID)

Frozen \(D_0\) pipeline (passive/H0 window \(\to\) \(\hat\theta\), probe
window \(\to r_\perp\)) is unchanged and still logged.

H0 (adequate physics) predicts that after removing \(\mathrm{col}(J_\theta)\),
probe residual is observation/numerical noise. Estimate a scalar

\[
\hat\sigma_\perp^2
=
\mathrm{mean}(r_{\perp,\mathrm{H0}}^2)
\]

from the **same episode’s** H0 window (Mode-A: passive context; contact:
`phase<2`). No \(\mathcal I\) flag.

\[
T_\perp
=
r_\perp^\top
(\hat\sigma_\perp^2 I+\epsilon I)^{-1}
r_\perp
=
\frac{\|r_\perp\|^2}{\hat\sigma_\perp^2+\epsilon}.
\]

Under H0, \(T_\perp\) is treated as \(\chi^2_{\nu}\) with
\(\nu=\max(n_{\mathrm{probe}}-2,1)\):

\[
p_\perp
=
P(T\ge T_\perp\mid H_0),\qquad
S_\perp=-\log p_\perp.
\]

Inputs: residual, tangent, ridge, H0/probe split. **Forbidden inputs:**
domain ID, true \(\alpha\), true \(\phi\), \(X_{\phi,\perp}\), \(C\), VoI.

A single threshold \(\tau_S\) is frozen from **Mode-A C0 only** at the
RS1A false-positive standard \(\le 1\%\). Contact C0 is evaluated at that
same \(\tau_S\); it must not get its own threshold.

## Oracle diagnostics (not gates)

Log RS2A’s \(X_\phi\), \(X_{\phi,\perp}\), \(\kappa_\perp\) on the probe
window. They must not enter \(S_\perp\) or \(\tau_S\).

## Matrix

New seeds, held out from RS2 Formal / C0 / RS2A:

\[
\{12101,12111,12121,12131,12141\}.
\]

Regimes: \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\). **C2 excluded.**

Mode-A: frozen sine \(f=0.20\,\mathrm{Hz}\), \(A/A_0\in\{1.0,1.5\}\).

Contact: frozen C0 scripts at scale \(1.0\), plus `pull_release` at
\(s^\star=2.0\) (adapter pair).

\[
N_{\mathrm{ModeA}}=30,\quad N_{\mathrm{contact}}=60,\quad N=90.
\]

Smoke: seed `12101` × C0 × {Mode-A \(1.5A_0\), contact `fast_pull`}. No GO.

## Hypotheses / GO

Let \(\tau_S\) be the empirical \(99\%\) quantile of Mode-A C0 \(S_\perp\)
(with \(n=10\), this is operationally the sample maximum).

### H1 — null calibration transport

\[
\boxed{
\begin{aligned}
\mathrm{FPR}(S_\perp>\tau_S\mid C0,\mathrm{ModeA}) &\le 0.01\\
\mathrm{FPR}(S_\perp>\tau_S\mid C0,\mathrm{contact}) &\le 0.05
\end{aligned}
}
\]

Contact allowance is \(1/20\) because \(n_{\mathrm{C0,contact}}=20\); it is
**not** a second fitted threshold. Report the raw counts.

### H2 — evidence operating-point transport (adapter pair, C1)

On C1, Mode-A \(1.5A_0\) vs contact `pull_release@\(s^\star\)`:

\[
g(u)
=
\frac{
\lvert\mathrm{median}\,u^{\mathrm{ModeA}}
-
\mathrm{median}\,u^{\mathrm{contact}}\rvert
}{
\tfrac12(
\lvert\mathrm{median}\,u^{\mathrm{ModeA}}\rvert
+
\lvert\mathrm{median}\,u^{\mathrm{contact}}\rvert
)+\epsilon
}.
\]

\[
\boxed{H2:\quad g(S_\perp)<g(D_0)}
\]

This asks whether whitened surprisal shrinks the Mode-A/contact gap relative
to raw \(D_0\). It is **not** an AUROC/recall gate.

\[
\boxed{RS3A\_GO = H1 \land H2}
\]

`RS3A_GO=true` does not set `RS2_GO=true`, does not unlock RS3B, and does
not authorize domain-specific \(\tau_S\).

## Failure interpretation

| Pattern | Read as |
|---------|---------|
| H1 ∧ H2 | local H0-whitened surprisal transports null and shrinks the \(D_0\) gap |
| ¬H1 | local \(\hat\sigma_\perp\) is not a transported H0 scale; stop |
| H1 ∧ ¬H2 | calibration on C0 is not enough; visibility geometry still dominates C1 \(S_\perp\) |
| AUROC high, H1/H2 fail | discrimination without transport; do not promote |

None of these outcomes authorizes contact \(\tau_{D_0}\) or \(C_{\mathrm{tol}}\)
search, or promoting \(X_{\phi,\perp}\) to a detector.

## Non-goals

- RS3B typed consequence / \(R_{\mathrm{abs}}\);
- RS3C tolerate/probe/revise/unknown-consequential policy;
- C2 / latch;
- domain ID features;
- using true \(\phi\) in the score.
