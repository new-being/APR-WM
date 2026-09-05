# R8-P1 Report — State-Neutral Validity Probe

Date: 2026-08-17  
Status: **`R8_P1_PASS=false`**; locus **validity_alignment**; R8-A0 **locked**  
Prereg: `REPORT/REG/R8/R8_P1_PREREG.md`  
Artifacts: `runs/r8_p1/formal/summary.json`  
One change vs P0: waveform \([+,-,-,+]\) with \(n=(13,12,12,13)\).  
Same \(u_0=1\), \(T=0.10\,\mathrm{s}\). No sweep. No detector.

## Question

Does a fixed zero-moment probe keep P0’s validity evidence while
returning \((y,\dot y)\) and making same-episode \(C_{\mathrm{down}}\)
affordable?

## Gates

| Gate | P0 | P1 |
|---|---|---|
| G0 \(D(h^S)=0\) | true | **true** |
| G1 commit-set \(S\) gap \(\ge 10^{-3}\) | true (\(1.82\times10^{-3}\)) | **false** (\(1.14\times10^{-4}\)) |
| G2 timing | true | **true** |
| G-return mean \(D_{\mathrm{terminal}}\le 2.2\times10^{-3}\) | (no gate) | **false** (\(7.40\times10^{-3}\)) |
| G3 \(C_{\mathrm{down}}/\lvert A\rvert\le 1\) | false (3.35) | **true (0.088)** |

\[
\boxed{\texttt{R8\_P1\_PASS}=\text{false}}
\]

Primary locus is **G1**. G-return also fails. **G3 passes**: P0’s
`probe_cost` locus is repaired by the waveform alone.

## Commit-set \(S_{\mathrm{epi}}\) (RMSE vs ID-plant rollout)

| class | \(S_{\mathrm{epi}}\) |
|---|---|
| ID | \(0\) |
| benign | \(7.59\times10^{-5}\) |
| invalid | \(1.90\times10^{-4}\) |

Order is still ID \(<\) benign \(<\) invalid, but the gap is an order of
magnitude below \(\varepsilon_S=10^{-3}\). Self-cancellation removed
most saturation-identifying transient. This is the
**neutrality vs observability** tension, even though G-return is not
cleanly passed.

## Terminal state (ID commit-set)

\(y_T\approx 1.5\times10^{-4}\) (much smaller than P0’s
\(8.8\times10^{-3}\)). \(D_{\mathrm{terminal}}\) is dominated by
**leftover** \(\dot y_T\approx 7.4\times10^{-3}\). The plant is damped,
not a double integrator, so \(\int a=0\) does not give \(\dot y(T)=0\).

## Cost

\(\overline{C}_{\mathrm{down}}=1.50\times10^{-4}\) vs
\(\overline{\lvert A\rvert}=1.70\times10^{-3}\) (ratio 0.088). Same-episode
task tax is now small.

## Not next

Not R8-A0. Not a detector. Not a \(T\)/\(u_0\) sweep. A billed
probe\(\to\)reset\(\to\)decision protocol is still not opened: P1 did
not produce a returning **and** informative probe. The damped leftover
velocity is a plant fact, not a reason to enlarge \(S_{\mathrm{epi}}\).
**R8-P2** (`REPORT/REP/R8/R8_P2_REPORT.md`) is the damped-kernel
correction, not another undamped-null waveform.
