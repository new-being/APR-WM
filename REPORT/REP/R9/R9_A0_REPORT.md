# R9-A0 Report — Persistent Certificate-Validity Belief

Date: 2026-08-17  
Status: **`R9_A0_GO=true`**; pattern **persistent_license**;
R9-B0 **unlocked, not run**  
Prereg: `REPORT/REG/R9/R9_A0_PREREG.md`  
Artifacts: `runs/r9_a0/formal/summary.json`

## Question

Can one P0 calibration establish a persistent \(b(\mathcal V)\) that
licenses the frozen R7 certificate for later fresh tasks, while
\(F_{\max}\) is block-stationary?

Two layers, not one score: \(b\ge\tau_{\mathcal V}=0.5\), then
\(L_{\mathrm{WM}}>\delta\). \(b_{k+1}=b_k\). No decay, no change
detection.

Logistic on \(\tilde S=S_{\mathrm{epi}}/10^{-3}\), fit only where
\(L>\delta\) on B0 fit∪cal. Held gate on the commit-set \(\times\)
three classes.

## Gates

| Gate | result |
|---|---|
| G-validity AUROC / Brier | **✓ \(1.0\) / \(\approx 0\)** |
| G-safe \(P(A>0\mid\mathrm{commit})\) | **✓ \(1.0\)** (\(n=14\)) |
| G-use useful recall, valid blocks | **✓ \(0.583\)** |
| G-persist \(k=2\) mask \(=k=1\) | **✓** |
| G-NetVoI mean \(\mathrm{NetVoI}(K=2)>0\) | **✓ \(6.14\times10^{-4}\)** |

\[
\boxed{\texttt{R9\_A0\_GO}=\text{true}}
\]

Invalid commit-set blocks: \(b=0\), none licensed. Valid: \(b=1\),
all 14 licensed. The two-layer policy matches the oracle-validity
ceiling on this held commit-set.

## Amortization

| \(K\) | persist | oracle | B0 (no \(C_{\mathrm{cal}}\)) | shuffle \(S\) |
|---|---|---|---|---|
| 1 | \(-9.85\times10^{-4}\) | same | \(+1.59\times10^{-3}\) | \(-1.69\times10^{-3}\) |
| **2** | \(\mathbf{+6.14\times10^{-4}}\) | same | \(+3.19\times10^{-3}\) | \(-7.90\times10^{-4}\) |
| 4 | \(+3.81\times10^{-3}\) | same | \(+6.38\times10^{-3}\) | \(+1.00\times10^{-3}\) |
| 8 | \(+1.02\times10^{-2}\) | same | \(+1.28\times10^{-2}\) | \(+4.59\times10^{-3}\) |

\[
K_{\min}^{\mathrm{oracle}}=2,\qquad
K_{\min}^{\mathrm{realized}}=2,\qquad
\Delta K_{\mathrm{realization}}=0.
\]

\(K=1\) is negative once \(C_{\mathrm{cal}}\) is billed: same-horizon
R8 was right that one decision cannot pay. \(K=2\) is the first
positive NetVoI, matching P0’s oracle break-even. Shuffle at \(K=2\)
stays negative: persistence is not a free lunch from unpaired \(S\).

B0’s higher NetVoI omits calibration cost **and** licenses invalid
shifts (B1’s three harmful commits). That is not a competitor for G-safe.

## Not done

No decay. No mid-block \(F_{\max}\) cut. No re-probe. R7 \(f_{\mathrm{WM}}\)
and \(q\) unchanged. **R9-B0 `GO=false`**
(`REPORT/REP/R9/R9_B0_REPORT.md`): licensed cons does not excite
saturation, so passive \(r_k\) cannot revoke \(1.5\).
