# R7-B1 Report — Frozen Certificate Under Dynamics Shift

Date: 2026-08-17  
Status: **`r7_b1_shift_valid=false`**; pattern **certificate not shift-valid**  
Prereg: `REPORT/REG/R7/R7_B1_PREREG.md`  
Artifacts: `runs/r7_b1/formal/summary.json`  
Does not adapt: \(f_{\mathrm{WM}}\), \(q\), \(\delta\), \(\varepsilon\), commit rule  
Shift: \(F_{\max}^{\mathrm{test}}=1.5\) (train \(2.0\)); same B0 held \(\alpha\)

## Question

Does an ID-calibrated one-sided certificate stay safe under a single
dynamics shift, or does it emit unsupported commits?

## Result

ID replay on \(F_{\max}=2.0\) reproduces B0 (coverage \(0.929\), recall
\(0.70\), 7 useful commits). Freeze is intact.

| | B0 / ID replay | \(F_{\max}=1.5\) |
|---|---|---|
| \(q\) (frozen) | \(8.71\times10^{-4}\) | same |
| coverage | 0.929 | **0.357** |
| precision | 1.0 | **0.571** (3/7 harmful) |
| useful recall | 0.70 (7/10) | **n/a** (\(n_{\mathrm{useful}}=0\)) |
| VoI | \(4.26\times10^{-4}\) | **\(-3.17\times10^{-6}\)** |
| \(n_{\mathrm{commit}}\) | 7 | **7** (same \(X\)-driven set) |

\[
\boxed{\texttt{r7\_b1\_shift\_valid}=\text{false}}
\]

\[
\boxed{\text{pattern}=\text{certificate not shift-valid}}
\]

The seven commits are exactly B0’s
\(\alpha\in\{3.02,3.18,3.22,3.42,3.46,3.66,3.74\}\). Under tighter
saturation those points are no longer useful: three are harmful
(\(A<0\)), four are gray (\(0<A<\delta\)). \(\hat A_{\mathrm{WM}}(X)\)
cannot see \(F_{\max}\), so \(L\) and the commit set do not move.

## What this is not

Not claimed: split-conformal as a method is broken. Exchangeability
was false here by construction.

\[
\boxed{\text{ID calibration does not certify shifted dynamics.}}
\]

Do not retune B0 \(q\) or enlarge \(f_{\mathrm{WM}}\) on this family.
**R8-P0** (`REPORT/REP/R8/R8_P0_REPORT.md`) tested a frozen diagnostic
probe: \(S_{\mathrm{epi}}\) aligns with certificate applicability, but
same-episode probe cost fails G3. R8-A0 stays locked.
