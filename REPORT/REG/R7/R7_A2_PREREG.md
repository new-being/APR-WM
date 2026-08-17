# R7-A2 Preregistration — Recall Recovery Under Frozen Certification

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R7/R7_A1_REPORT.md` (`GO=true`)  
Does not change: \(L=\hat A-q\); \(L>\delta\Rightarrow\mathrm{commit}\);
\(\delta=10^{-3}\); \(\varepsilon=0.10\); plant \(F_{\max}=2.0\);
A1 fit/cal/held \(\alpha\) splits  
Does not: architecture search; degree 3/4/5 sweep; neural probe;
new family; reuse P1 \(\alpha\) grid

## Question

Can a more adequate advantage predictor recover useful-switch recall
**without** sacrificing one-sided validity or switch precision?

\[
\boxed{
f\text{ is the only free object.}
}
\]

This is not an A1 rescue. A1 already answered misspecification
behavior for frozen degree-2 OLS. A2 asks whether **predictor
adequacy** controls opportunity while the **certificate** still
controls permission.

## Predictor (one model, locked here)

Cubic truncated-power spline, OLS, fixed knots in \(X=0.25\alpha\):

\[
\hat A(X)=\beta_0+\beta_1 X+\beta_2 X^2+\beta_3 X^3
+\sum_{k=1}^{3}\theta_k(X-t_k)_+^3.
\]

Interior knots, specified from the **plant**, not from A1 held
residuals:

\[
t=(0.30,0.50,0.70)
\quad\text{i.e.}\quad
\alpha\in\{1.2,2.0,2.8\}.
\]

The middle knot is default-action saturation
\(F_{\max}/u_{\mathrm{default}}=2.0\). The other two are the fixed
offsets \(\pm 0.8\) in \(\alpha\). Degrees of freedom \(=7\). Fit
\(n=14\). No knot or degree tuning after seeing A2 regret.

## Splits

Identical to A1 (`REPORT/REG/R7/R7_A1_PREREG.md`) so the comparison
isolates \(f\), not a new sample. Shuffle-\(X\) seed 0 inside each
split.

## Gates

A1 held numbers in `runs/r7_a1/formal/summary.json` are the official
baselines (rounded in the A1 report: recall \(0.273\), VoI
\(1.87\times10^{-4}\), precision \(1\), coverage \(1\)).

Non-tradeable (must both hold, else `GO=false`):

- G-cert: coverage \(\ge 0.90\)
- G-safe: non-vacuous precision \(=1\)

Recovery (strict vs A1 held, else not recovery):

- useful recall \(>\) A1 useful recall
- \(\mathrm{VoI}>\) A1 VoI

Control: \(\mathrm{VoI}_{\mathrm{align}}>\mathrm{VoI}_{\mathrm{shuf}}\).

Primary pattern (do not collapse into one accuracy):

| coverage+safety | recall/VoI vs A1 | meaning |
|---|---|---|
| \(\checkmark\) | both \(\uparrow\) | recall recovery without safety loss |
| \(\checkmark\) | not both \(\uparrow\) | better \(f\) need not yield more certificates |
| \(\times\) | \(*\) | certificate invalid |

Expected chain if recovery is real:

\[
f\text{ improves}\rightarrow q\downarrow\rightarrow
\text{more useful certificates}\rightarrow\text{recall/VoI}\uparrow.
\]

Forbidden chain:

\[
f\text{ improves}\rightarrow\text{aggressive commits}\rightarrow
\text{coverage/safety}\downarrow.
\]

## Diagnostic (offline, not a gate)

On every held useful point report

\[
A,\ \hat A,\ q,\ L,\ A-\delta,\ \hat A-\delta
\]

and partition misses:

- **model bias**: useful and \(\hat A\le\delta\)
- **certificate width**: useful and \(\hat A>\delta\) but \(L\le\delta\)
