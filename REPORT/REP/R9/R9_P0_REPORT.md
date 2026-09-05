# R9-P0 Report — Amortized Validity-Acquisition Feasibility

Date: 2026-08-17  
Status: **`R9_P0_PASS=true`**; pattern **amortized_feasible**;
persistent \(b(\mathcal V)\) **unlocked, not implemented**  
Prereg: `REPORT/REG/R9/R9_P0_PREREG.md`  
Artifacts: `runs/r9_p0/formal/summary.json`  
R8 same-horizon family: `REPORT/REP/R8/R8_SAME_HORIZON_STOP.md`

## Question

With an independent calibration clock, does the frozen P0 probe plus
frozen R0 PD recover, and is \(C_{\mathrm{cal}}\) amortizable over a
block-stationary \(F_{\max}\) of length \(K\le 20\)?

No detector. No \(\omega_n\) retune.

## Recovery time (commit-set)

| class | mean \(T_{\mathrm{recover}}\) | max |
|---|---|---|
| ID \(F_{\max}=2.0\) | \(0.667\,\mathrm{s}\) | \(0.668\,\mathrm{s}\) |
| benign \(2.2\) | \(0.680\,\mathrm{s}\) | \(0.682\,\mathrm{s}\) |
| invalid \(1.5\) | \(0.626\,\mathrm{s}\) | \(0.626\,\mathrm{s}\) |

All recover inside \(5\,\mathrm{s}\). Conservative
\(T_{\mathrm{cal}}=0.10+0.682=0.782\,\mathrm{s}\).
Same-horizon R0 failed because only \(0.40\,\mathrm{s}\) remained after
the probe; the controller itself reaches \(\varepsilon\) in
\(\approx 0.68\,\mathrm{s}\) of reset.

## Break-even

\(V_{\mathrm{dec}}=\overline{|A|}_{\mathrm{commit,ID}}=1.70\times10^{-3}\).
\(C_{\mathrm{time}}=2.66\times10^{-3}\), \(C_{\mathrm{effort}}=3.6\times10^{-6}\)
(effort is negligible). \(C_{\mathrm{cal}}=2.67\times10^{-3}\).

\[
K_{\min}=2,\qquad
\tau_{\mathrm{validity}}\ge K_{\min}T_{\mathrm{task}}=1.0\,\mathrm{s}.
\]

G1 of the P0 probe is unchanged: gap \(1.82\times10^{-3}\).

## Gates

| Gate | result |
|---|---|
| G-recover \(T_{\mathrm{recover}}\le 5\,\mathrm{s}\) | **✓** |
| G-amortize \(K_{\min}\le 20\) | **✓ \(K_{\min}=2\)** |

\[
\boxed{\texttt{R9\_P0\_PASS}=\text{true}}
\]

Oracle answer: **if** \(F_{\max}\) lasts two subsequent \(0.50\,\mathrm{s}\)
tasks, one billed calibration is worth more than its amortized tax.
That is a persistence assumption, not a detector.

## Not done

No \(b_t(\mathcal V)\). No change detection. No re-probe policy.
R8-A0 remains locked; R8 same-horizon family remains STOP.
The unlocked next mechanism is persistent validity belief under
block-stationarity, then decay / change evidence.
**R9-A0 `GO=true`** (`REPORT/REP/R9/R9_A0_REPORT.md`): two-layer
\(b(\mathcal V)\) then \(L_{\mathrm{WM}}\); \(\Delta K_{\mathrm{realization}}=0\).
