# R7 Preregistration — Switch Certification

Date: 2026-08-17  
Status: **P0 PASS**; A-series **FROZEN**; **R7-P1 `PASS=true`**; **R7-B0 `GO=true`** (WM frozen); **R7-B1 `shift_valid=false`**; **R8 same-horizon STOP**; **R8-R0 `PASS=false`**; **R9 toy-family STOP**; **R9-B1-P0 `PASS=false`**; **R10-C0 LOCKED** (no real residual log)  
Depends on: `REPORT/REP/R6/R6_REALIZATION_FREEZE.md`,
`REPORT/REP/R7/R7_P0_REPORT.md`  
Does not reopen: R5 self-stress \(\lambda\); R6-C0; \(z_{\min}\) sweep  
Does not train: a classifier whose labels are the current D0 failure
set \(\{\lambda=0,2,12\}\)

## Role

New main stage after B0. Not a better uncertainty estimator on the
same seven \(\lambda\). Question:

> What evidence is sufficient to support switching from the current
> robust/default action to a **specific** alternative?

\[
\boxed{
\text{proposal}
\rightarrow
\text{alternative-specific certificate}
\rightarrow
\text{commit / abstain}
}
\]

B0 showed veto power without positive evidence that cons beats mid.

## Certificate (directional)

Do not use symmetric \(|\hat m|/\sigma\) as the commit rule. Require
an alternative-specific statement, for example

\[
\Pr\!\bigl(J(a_{\mathrm{alt}})<J(a_{\mathrm{default}})-\delta
\mid \mathcal E\bigr)
\]

or

\[
\mathrm{LCB}\bigl(J(a_{\mathrm{default}})-J(a_{\mathrm{alt}})\bigr)>\delta.
\]

Only then authorize \(a_{\mathrm{default}}\to a_{\mathrm{alt}}\);
otherwise keep default. The certificate must answer both: **is the
sign correct, and is the improvement large enough?**

## Fresh regimes only

R7 **must** be tested on a new decision family or fully fresh
hidden-state / action conditions. Using the STOP self-stress grid as
supervision would turn the known failure set into the training label.

Contact family stays paused. Self-stress I0/D0/B0 stay frozen.

## Two independent capabilities (not one accuracy)

1. **Negative certification:** \(P(\text{harmful switch accepted})\)
   low. (B0 is a prototype.)
2. **Positive certification:** \(P(\text{useful switch certified})>0\).
   Otherwise the system is the conservative baseline.

Gates must be split:

- harmful-switch **precision**
- useful-switch **recall**

then executed regret / realized \(\mathrm{VoI}_\Pi\). Eternal abstain
must not pass as a single accuracy.

## A0 (executed)

R7-A0 (`REPORT/REP/R7/R7_A0_REPORT.md`) froze \(\delta=10^{-3}\),
split-conformal \(L=\hat A-q\), and the four gates plus shuffled-\(X\)
control. `GO=true` on a fresh \(\alpha\) grid. **R7-P1**
(`REPORT/REP/R7/R7_P1_REPORT.md`) admits a saturated plant with
frozen G-mis. **R7-A1** (`REPORT/REP/R7/R7_A1_REPORT.md`) kept the
frozen degree-2 OLS and \(L=\hat A-q\) on a new 14/14/28 split:
\(q\uparrow\), precision 1, recall 0.273, VoI \(>0\), shuffle never
commits. Pattern: robust positive certification (conservative tail).
**R7-A2** (`REPORT/REP/R7/R7_A2_REPORT.md`) kept \(L=\hat A-q\) and
the A1 split; only \(f\) became a frozen cubic spline. Recall 0.273
\(\to\) 0.909, VoI \(\uparrow\), precision 1, coverage 0.964. Pattern:
recall recovery without safety loss.
**A-series frozen.** **R7-B0** (`REPORT/REP/R7/R7_B0_REPORT.md`)
replaces \(X\to\hat A\) by \(Y\)-only WM then \(J(\hat Y)\). Same
certificate; `GO=true`; pattern WM-mediated positive certification.
Mediation cost is diagnostic, not a gate.
**R7-B1** (`REPORT/REP/R7/R7_B1_REPORT.md`): frozen \(f_{\mathrm{WM}},q\)
under \(F_{\max}^{\mathrm{test}}=1.5\). Pattern: certificate not
shift-valid (coverage \(0.357\), 3 harmful commits). ID calibration
does not certify shifted dynamics. Do not retune B0.
