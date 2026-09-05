# R8-P0 Report — Active Certificate-Validity Feasibility

Date: 2026-08-17  
Status: **`R8_P0_PASS=false`**; locus **probe_cost**; R8-A0 **locked**  
Prereg: `REPORT/REG/R8/R8_P0_PREREG.md`  
Artifacts: `runs/r8_p0/formal/summary.json`  
Does not: validity model; \(\tau\); \(\pi\); WM update; recalibration

## Question

Can a frozen short probe, before mid/cons, supply evidence of
**certificate applicability** (not of \(F_{\max}\))?

## Oracle classes (labels not used in \(S_{\mathrm{epi}}\))

| class | \(F_{\max}\) | coverage | harmful commits | \(\mathcal V\) |
|---|---|---|---|---|
| ID | 2.0 | 0.929 | 0 | valid |
| benign shift | 2.2 | 1.00 | 0 | valid |
| invalidating | 1.5 | 0.357 | 3 | invalid |

## Gates

| Gate | Result |
|---|---|
| G0 passive \(D(h^S)=0\) | **true** |
| G1 commit-set \(S_{\mathrm{epi}}\) gap \(\ge 10^{-3}\) | **true** (\(1.82\times10^{-3}\)) |
| G2 probe before commit | **true** |
| G3 \(\overline{C}_{\mathrm{down}}/\overline{\lvert A\rvert}\le 1\) | **false** (ratio **3.35**) |

\[
\boxed{\texttt{R8\_P0\_PASS}=\text{false},\quad\text{locus}=\text{probe cost}}
\]

## What worked

On the frozen commit-set
\(\alpha\in\{3.02,3.18,3.22,3.42,3.46,3.66,3.74\}\):

| class | \(S_{\mathrm{epi}}\) range |
|---|---|
| ID | \([3.4,7.1]\times10^{-4}\) |
| benign | \([1.7,5.4]\times10^{-4}\) |
| invalid | \([2.53,2.91]\times10^{-3}\) |

Intervals are disjoint. Benign shift is **not** a large residual.
\(S_{\mathrm{epi}}\) tracks applicability, not “\(F_{\max}\neq 2\)”.
Hold remains \(h^S\equiv 0\). No \(A\) or \(J\) in \(S_{\mathrm{epi}}\).

## What failed

Same-episode tax: after the \(0.10\,\mathrm{s}\) \(u=1\) pulse, running
default \(0.50\,\mathrm{s}\) from the displaced state costs
\(\overline{C}_{\mathrm{down}}=5.70\times10^{-3}\) vs
\(\overline{\lvert A\rvert}=1.70\times10^{-3}\) on that commit-set.
Prefix-only cost is small (\(8.7\times10^{-5}\)); the damage is
**downstream task contamination**, not the pulse energy.

## Not claimed / not next

Not claimed: detectors cannot work. G1 passed. Not claimed: conformal
should be retuned. Do **not** open R8-A0 (\(\tau\), \(\pi\), NetVoI
policy) with this pulse. Do **not** train a validity classifier.

If R8 continues, the object is a **cheaper same-episode probe** (or an
explicit reset protocol whose time is billed), not more capacity on
\(S_{\mathrm{epi}}\). **R8-P1** (`REPORT/REP/R8/R8_P1_REPORT.md`)
replaced the pulse by a frozen four-phase waveform: G3 passed
(ratio \(0.088\)), G1 and G-return did not. **R8-P2**
(`REPORT/REP/R8/R8_P2_REPORT.md`) used damped-return \([+,-,+]\):
G-return and G3 passed, G1 did not. Open-loop probe family frozen.
A0 remains locked.
