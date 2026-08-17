# R7-P1 Report — Misspecification Preflight

Date: 2026-08-17  
Status: **`R7_P1_PASS=true`** (R7-A1 **run**, `GO=true`)  
Prereg: `REPORT/REG/R7/R7_P1_PREREG.md`  
Artifacts: `runs/r7_p1/formal/summary.json`  
Does not run: conformal \(L\), commit, model upgrade

## Family

\[
\ddot y=\operatorname{sat}(\alpha u;2)-c\dot y.
\]

Hold still has \(D(h^{S})=0\). \(X=0.25\alpha\). Mid clips for
\(\alpha\ge 2\) (\(F=2\)); cons does not clip on this grid.

## Gates

| Gate | Value | Pass |
|---|---|---|
| G0 \(\max D_h\) | **0** | true |
| G1 \(\rho(\alpha,X)\) | **1** | true |
| G2 both signs of \(A\) | \(A<0\) for \(\alpha\le 2.0\); \(A>0\) for \(\alpha\ge 2.3\) | true |
| G3 \(\lvert A\rvert>10^{-3}\) | 5 harmful + 5 useful | true |
| G4 coverage | 5 / 5 | true |
| G-mis LOO degree-2 | RMSE **\(1.35\times10^{-3}\)** \(>10^{-4}\); \(R^2_{\mathrm{LOO}}\) **0.682** \(<0.99\) | true |

\[
\boxed{\texttt{R7\_P1\_PASS}=\text{true}}
\]

G-mis was frozen before the curve. A0’s unsaturated plant would fail it
(\(R^2=1\), residual \(\sim 10^{-16}\)).

\(\alpha=2.3\) is gray: \(A=5.6\times10^{-4}<\delta\). Mid \(y_T\) is
flat at \(0.142\) once saturated; cons still climbs toward \(y^\star\).

## What this unlocks

Switchability is still clean **and** degree-2 \(A(X)\) is genuinely
imperfect. **R7-A1** (`REPORT/REP/R7/R7_A1_REPORT.md`) kept A0’s \(f\)
and \(L=\hat A-q\) on a new 14/14/28 split: \(q\) rose to
\(1.30\times10^{-3}\), precision stayed 1, recall fell to 0.273, VoI
stayed positive. No bigger predictor.
