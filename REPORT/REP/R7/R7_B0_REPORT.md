# R7-B0 Report — World-Model-Mediated Switch Certification

Date: 2026-08-17  
Status: **`R7_B0_GO=true`**; pattern **WM-mediated positive certification**  
Prereg: `REPORT/REG/R7/R7_B0_PREREG.md`  
Artifacts: `runs/r7_b0/formal/summary.json`  
A-series: **FROZEN** at A2  
Does not change: \(L=\hat A-q\); \(\delta=10^{-3}\); \(\varepsilon=0.10\);
saturated plant \(F_{\max}=2.0\)  
Does change: \(\hat A\) comes from \(J(\hat Y)\) after a \(Y\)-only WM

## Question

Does task-independent future prediction support the same one-sided
certificate with positive realized \(\mathrm{VoI}\)?

\[
\boxed{
X\rightarrow\hat Y^{future}\rightarrow\hat A_{\mathrm{WM}}
\rightarrow L_{\mathrm{WM}}>\delta
\rightarrow\text{useful switch}
\rightarrow\mathrm{VoI}>0.
}
\]

WM-fit never saw \(A\) or \(J\). \(h^S\) absmax \(=0\) (hold at rest).

## Held test (\(n=28\), fresh \(\alpha\))

| | WM certificate | Direct diagnostic (A2 spline on same split) |
|---|---|---|
| \(L_Y\) RMSE (test) | \(5.42\times10^{-3}\) | — |
| \(q\) | **\(8.71\times10^{-4}\)** | \(1.79\times10^{-4}\) |
| coverage | **0.929** | 1.00 |
| precision | **1.0** | 1.0 |
| useful recall | **0.70** (7/10) | 1.00 (10/10) |
| VoI | **\(4.26\times10^{-4}\)** | \(5.72\times10^{-4}\) |
| shuffle commits | **0** | — |

\[
\boxed{\texttt{R7\_B0\_GO}=\text{true}}
\]

\[
\boxed{\text{pattern}=\text{WM-mediated positive certification}}
\]

Commits: \(\alpha\in\{3.02,3.18,3.22,3.42,3.46,3.66,3.74\}\). Useful
misses: \(\{2.74,2.78,2.94\}\) (near-threshold; abstain). No harmful
commit.

## Mediation cost (not a gate)

Same cal/test, direct \(X\to A\) is narrower and recalls all useful
points. WM pays extra width (\(\Delta q\approx 6.9\times10^{-4}\)) and
drops recall \(1.00\to 0.70\). Historical A2 (\(q=1.59\times10^{-4}\),
recall \(0.909\)) is a **different sample**, cited only as the frozen
direct-certificate block.

The cost is real and **safe**: trajectory error is absorbed as
abstention, not as invalid commits. B0 was not required to match A2.

## Not claimed

Not claimed: this \(\phi(X,u)\) OLS is a general world model. Not
claimed: \(L_Y\) RMSE is the scientific object. **R7-B1**
(`REPORT/REP/R7/R7_B1_REPORT.md`) is the frozen-pipeline dynamics-shift
audit. No A3. No ID-WM retune.
