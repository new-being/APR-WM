# R7-A2 Report — Recall Recovery Under Frozen Certification

Date: 2026-08-17  
Status: **`R7_A2_GO=true`**; pattern **recall recovery without safety loss**;
**A-series FROZEN** (next: B0, not A3)  
Prereg: `REPORT/REG/R7/R7_A2_PREREG.md`  
Artifacts: `runs/r7_a2/formal/summary.json`  
Plant / splits: same as A1  
Does not change: \(L=\hat A-q\); \(\delta=10^{-3}\); \(\varepsilon=0.10\)  
Does change: \(f\) only — cubic truncated-power spline, knots
\(X\in\{0.30,0.50,0.70\}\) (\(\alpha\in\{1.2,2.0,2.8\}\))

## Question

Can a more adequate \(\hat A\) recover useful-switch recall without
trading away one-sided coverage or precision?

## Held comparison (same 28 \(\alpha\) as A1)

| | A1 degree-2 | A2 spline |
|---|---|---|
| cal RMSE | \(7.77\times10^{-4}\) | **\(8.87\times10^{-5}\)** |
| \(q\) | \(1.30\times10^{-3}\) | **\(1.59\times10^{-4}\)** |
| coverage | 1.00 | **0.964** (\(\ge 0.90\)) |
| precision | 1.0 | **1.0** |
| useful recall | 0.273 (3/11) | **0.909 (10/11)** |
| VoI | \(1.87\times10^{-4}\) | **\(5.78\times10^{-4}\)** |
| \(n_{\mathrm{commit}}\) | 3 | **10** (all useful) |
| shuffle commits | 0 | **0** |

\[
\boxed{\texttt{R7\_A2\_GO}=\text{true}}
\]

\[
\boxed{\text{pattern}=\text{recall recovery without safety loss}}
\]

Coverage is 27/28. The single conformal miss is harmful
\(\alpha=1.6\), \(A-L\approx -4\times10^{-6}\); that point still
abstains. No harmful commit. Gray points still default.

## Chain

\[
\boxed{
f\text{ improves}\rightarrow q\downarrow\rightarrow
\text{more useful certificates}\rightarrow\text{recall/VoI}\uparrow
}
\]

Not the forbidden chain: commits rose from 3 to 10, but precision
stayed 1 and coverage stayed above \(1-\varepsilon\).

## Bias vs width (11 useful points)

| miss locus | A2 count |
|---|---|
| committed | 10 |
| model bias (\(\hat A\le\delta\)) | **0** |
| certificate width (\(\hat A>\delta\), \(L\le\delta\)) | **1** (\(\alpha=2.55\)) |

The leftover miss is knife-edge useful (\(A-\delta\approx 2\times10^{-6}\)):
\(\hat A>\delta\) but \(q\) still pushes \(L\) under \(\delta\). A1’s
eight misses were mostly bias-plus-width under a too-stiff quadratic;
A2 removed the bias term on this grid.

## Design principle (supported here)

\[
\boxed{
\text{predictor capacity controls opportunity;}
\quad
\text{certificate controls permission.}
}
\]

A0: near-realizable \(f\) \(\Rightarrow\) high recall + safe cert.  
A1: misspecified \(f\) \(\Rightarrow\) wider cert + safe abstention.  
A2: better \(f\) \(\Rightarrow\) recall recovery without losing safety.

## Not claimed

Not claimed: any spline, or a neural \(\hat A\), would recover recall.
The knots were frozen from the plant (default saturation at
\(\alpha=2\)), not searched on held regret. Not claimed: coverage can
be driven back to 1.00 without changing \(\varepsilon\). Not a new
family. **A-series frozen** after A2; world-model-mediated \(\hat A\)
is R7-B0 (`REPORT/REP/R7/R7_B0_REPORT.md`), not A3.
