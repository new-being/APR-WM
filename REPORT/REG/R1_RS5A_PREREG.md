# R1-RS5A Preregistration — Intervention-Indexed Calibration and Abstention

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R1_RS4A_REPORT.md`  
Does not change: `RS4A_GO=false`, `RS3A_GO`, `RS3A.1_GO`, `RS2_GO`,
`RS2A_GO`, `RS2B_GO`, `RS1C_GO`, C0, \(s^\star\), RS1C policy  
Locks: **RS4B/C**, **RS3B/C**, **RS5B/C**, VoI, revision, H32, consequence

## Placement

RS4A closed zero-shot transport of evidential meaning. This is a **new
main stage**, not RS4A.1.

\[
\boxed{
RS4A\_GO=false
\rightarrow
R1\text{-RS5A (this stage)}
\qquad
RS5B\text{/}C\text{ locked}
}
\]

Stage name:

\[
\boxed{
R1\text{-RS5A — Intervention-Indexed Calibration and Abstention}
}
\]

中文名：**干预索引的校准与弃权**。

## Abandoned requirement

\[
\boxed{
\text{zero-shot transport of evidential meaning across unknown }\mathcal I
}
\]

is no longer a success criterion. Geometry may fingerprint families; it
does not transport calibrated \(p(Y\mid D_0)\).

## Scientific question

\[
\boxed{
\text{If epistemic semantics are intervention-dependent,
can a world model remain well-calibrated by explicitly tracking
which intervention calibration is valid?}
}
\]

中文：认识决策量允许依赖**已知**干预机制；若当前机制没有被校准过，
系统必须承认标尺未知并 **abstain**，而不是套用旧标尺。

This is not one-hot \(\tau_A,\tau_C\) over a closed world. The state is

\[
(\mathcal I,\mathcal C_{\mathcal I},\mathrm{valid}(\mathcal C_{\mathcal I})).
\]

\[
\pi(E\mid\mathcal I,\mathrm{valid}(\mathcal C_{\mathcal I})).
\]

## States (evidence layer only)

No revision / VoI / H32 / \(C\). Binary evidence decision is

\[
\hat Y\in\{\text{adequate},\text{inadequate}\}
\quad\text{or}\quad
\text{abstain}.
\]

`intervention-uncalibrated` \(\equiv\) abstain: \(D_0\) is computed but
has no licensed decision semantics.

## Calibrated vs unseen

Development may fit calibrations **only** for

\[
\mathcal I_{\mathrm{calibrated}}=\{\mathrm{ModeA},\mathrm{contact\text{-}pull}\}.
\]

`pull_push` is unseen: never used to fit \(\mathcal C\) or the support
threshold.

Coarse index (learner-visible, not a family string):

\[
\hat{\mathcal I}
=
\begin{cases}
\mathrm{ModeA} & \text{if }\mathrm{contact\_frac}<0.5\\
\mathrm{contact} & \text{otherwise.}
\end{cases}
\]

Support of \(\mathcal C_{\mathrm{contact}}\) is estimated from
**contact-pull development only**, using frozen coordinates

\[
z_{\mathrm{supp}}=(\mathrm{sign\_coverage},\log n,q_{\mathrm{span}}).
\]

Mahalanobis distance to that cloud; \(\tau_{\mathrm{supp}}=\) 95th
percentile of development distances. Mode-A has its own cloud.

`pull_push` is still coarse-`contact`, so indexed-without-abstain will
silently apply \(\mathcal C_{\mathrm{pull}}\). Abstain must refuse when
\(z_{\mathrm{supp}}\) is outside that cloud.

## Calibration objects

On the development split, ridge logistic

\[
\mathcal C_{\mathcal I}:\quad \hat p=p(Y=1\mid \log D_0,\mathcal I)
\]

fit separately for Mode-A and contact-pull. Global baseline: one logistic
on pooled Mode-A + contact-pull development.

No requirement that \(D_0\) or \(\hat p\) match across \(\mathcal I\).
Success on a known \(\mathcal I\) is **conditional calibration**:

\[
p(Y=1\mid\hat p,\mathcal I)\approx\hat p.
\]

## Policies

| Policy | Known \(\mathcal I\) in support | Unseen / OOD support |
|--------|--------------------------------|----------------------|
| \(\pi_{\mathrm{global}}\) | pooled \(\mathcal C\) | still pooled \(\mathcal C\) (always confident) |
| \(\pi_{\mathrm{indexed}}\) | \(\mathcal C_{\hat{\mathcal I}}\) | reuse \(\mathcal C_{\mathrm{contact}}\) if coarse-contact (always confident) |
| \(\pi_{\mathrm{indexed+abstain}}\) | \(\mathcal C_{\hat{\mathcal I}}\) | **abstain** |

A **confident epistemic decision** is any non-abstain \(\hat Y\) with
threshold \(\hat p>0.5\).

## Splits / matrix

New seeds \(\{15101,15111,15121,15131,15141\}\).
Development: first three. Held-out: last two.
Regimes \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\).
Jobs as RS4A (Mode-A two amps; pull `fast_pull`+`pull_release@\(s^\star\)`;
push two scales). \(N=90\). C2 excluded.

Smoke: `15101` × C0 × {Mode-A \(1.5A_0\), `fast_pull`, `pull_push@1`}.

## Hypotheses / GO

Evaluate **held-out seeds only**.

### H1 — indexed calibration beats global on known \(\mathcal I\)

On held-out Mode-A ∪ contact-pull, Brier of \(\pi_{\mathrm{indexed}}\)
**strictly below** Brier of \(\pi_{\mathrm{global}}\).

### H2 — known-\(\mathcal I\) C0 specificity

I-specific \(\tau=\) 95th percentile of that \(\mathcal C_{\mathcal I}\)
on **development C0**. Held-out C0 FPR of \(\pi_{\mathrm{indexed}}\)
\(\le 0.20\).

### H3 — unseen false-confidence (the new core gate)

On all `pull_push` episodes (dev+held-out; none entered \(\mathcal C\)):

\[
\boxed{
\begin{aligned}
\mathrm{FCR}(\pi_{\mathrm{indexed+abstain}}) &\le 0.10\\
\mathrm{FCR}(\pi_{\mathrm{global}}) &= 1\\
\mathrm{FCR}(\pi_{\mathrm{indexed}}) &= 1
\end{aligned}
}
\]

FCR = fraction of confident (non-abstain) decisions.

### H4 — abstain does not refuse the calibrated cloud

Held-out contact-pull abstain rate of \(\pi_{\mathrm{indexed+abstain}}\)
\(\le 0.20\).

\[
\boxed{RS5A\_GO = H1 \land H2 \land H3 \land H4}
\]

`RS5A_GO=true` does not unlock RS5B, RS3B, revision, or H32.

## Failure interpretation

| Pattern | Read as |
|---------|---------|
| all pass | indexed calibration + abstention is a viable evidence layer |
| ¬H1 | even in-family, \(D_0\) logistic does not beat a global ruler |
| ¬H2 | I-specific \(\hat p\) still over-fires C0 |
| ¬H3 | support check cannot mark `pull_push` as uncalibrated |
| ¬H4 | support check abstains the calibrated contact cloud too |
| H3 trivial one-hot family ID | forbidden; coarse index must not see `pull_push` label |

## Non-goals

- RS4A.1 extra \(z\) for transport;
- RS5B typed consequence;
- VoI / install / passivity / H32;
- requiring \(p_A(D_0)=p_B(D_0)\).
