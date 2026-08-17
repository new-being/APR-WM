# R7-A1 Report — Certificate Under Misspecification

Date: 2026-08-17  
Status: **`R7_A1_GO=true`**; pattern **robust positive certification**  
Prereg: `REPORT/REG/R7/R7_A1_PREREG.md`  
Artifacts: `runs/r7_a1/formal/summary.json`  
Plant: P1 saturation \(F_{\max}=2.0\)  
Does not change: degree-2 OLS; \(L=\hat A-q\); \(\delta=10^{-3}\); \(\varepsilon=0.10\)

## Question

Does the frozen A0 certificate, on a plant where degree-2 \(A(X)\) is
imperfect, raise \(q\) and abstain more — or emit unsafe commits?

## Held test (\(n=28\), new \(\alpha\), not P1)

| | A0 (linear) | A1 (saturated) |
|---|---|---|
| \(q\) | \(\sim 6\times10^{-17}\) | **\(1.30\times10^{-3}\)** |
| coverage | 0.90 (\(n=10\)) | **1.00** |
| \(n_{\mathrm{commit}}\) | 5 / 5 useful | **3 / 11** useful |
| precision | 1.0 | **1.0** |
| useful recall | 1.0 | **0.273** (\(\ge 0.25\)) |
| VoI | \(5.43\times10^{-3}\) | **\(1.87\times10^{-4}\)** |
| shuffle commits | 0 | **0** |

\[
\boxed{\texttt{R7\_A1\_GO}=\text{true}}
\]

\[
\boxed{\text{pattern}=\text{robust positive certification}}
\]

(safety \(\checkmark\), recall \(\checkmark\) at the frozen 0.25 floor).
Shuffle still never commits.

## Degeneration (the scientific result)

Prediction worsens \(\rightarrow\) \(q\) rises by many orders of
magnitude \(\rightarrow\) far more abstention. The three commits are
the strongest useful tail (\(\alpha\in\{3.55,3.70,3.75\}\)). Gray and
borderline useful points (\(A\) just above \(\delta\)) stay default.
**No harmful commit.**

That is the intended misspecification behavior, not A0’s near-oracle
recall. VoI stays positive but is much smaller: the certificate
harvests only the well-separated tail.

## Not claimed

Not claimed: recall would stay near 1 under saturation. The 0.273
figure is a **floor pass**, not A0-level positive certification.
Raising \(f\)’s capacity is still forbidden as a rescue. **R7-A2**
(`REPORT/REP/R7/R7_A2_REPORT.md`) is the separate prereg that changes
only \(f\) and recovers recall without dropping coverage/precision.
