# R8-P2 Report — Damping-Aware State-Returning Probe

Date: 2026-08-17  
Status: **`R8_P2_PASS=false`**; pattern **returning but uninformative**;
open-loop same-episode probe family **FROZEN**; R8-A0 **locked**  
Prereg: `REPORT/REG/R8/R8_P2_PREREG.md`  
Artifacts: `runs/r8_p2/formal/summary.json`

## Question

Does replacing P1’s undamped moments by the correct damped kernels
restore return **and** keep enough validity evidence?

Waveform frozen by \(d_{\mathrm{initial}}\) only: \([+,-,+]\),
\(n=(13,25,12)\). Not chosen to maximize \(S_{\mathrm{epi}}\).

## Gates

| Gate | P0 | P1 | P2 |
|---|---|---|---|
| G0 | ✓ | ✓ | **✓** |
| G1 gap \(\ge 10^{-3}\) | ✓ \(1.82\times10^{-3}\) | × \(1.14\times10^{-4}\) | **× \(1.13\times10^{-4}\)** |
| G2 | ✓ | ✓ | **✓** |
| G-return \(\le 2.2\times10^{-3}\) | — | × \(7.40\times10^{-3}\) | **✓ \(1.68\times10^{-4}\)** |
| G3 ratio \(\le 1\) | × 3.35 | ✓ 0.088 | **✓ \(\approx 0\)** |

\[
\boxed{\texttt{R8\_P2\_PASS}=\text{false},\quad
\text{pattern}=\text{returning but uninformative}}
\]

Damped return geometry works: leftover \(D_{\mathrm{terminal}}\) drops
two orders from P1. Same-episode cost stays cheap (P1’s G3 repair
holds). **G1 does not come back.** Commit-set \(S_{\mathrm{epi}}\):
ID \(0\), benign \(7.55\times10^{-5}\), invalid \(1.89\times10^{-4}\).

The first-phase \(d_1=0.026\,\mathrm{s}\) is the longest initial
excitation compatible with damped return at this \(T\); it is still
far shorter than P0’s \(0.10\,\mathrm{s}\) one-way pulse. Force-scale
differences that are \(F_{\max}s(t)\) are almost cancelled by the same
kernel that zeroes the terminal state.

## Family table (now complete)

| | evidence | cheap | return |
|---|---|---|---|
| P0 one-way | ✓ | × | × |
| P1 undamped-null | × | ✓ | × |
| P2 damped-return | × | ✓ | ✓ |

\[
\boxed{
\text{certificate applicability is actively observable (P0),}
\quad
\text{but the tested state-returning open-loop probes
cannot acquire enough evidence cheaply.}
}
\]

## Not next

Not a fifth waveform. Not a detector. Not R8-A0. This open-loop
same-episode probe family is frozen. **R8-R0**
(`REPORT/REP/R8/R8_R0_REPORT.md`) billed P0 probe plus a nominal PD
reset: G1 held, G-reset did not.
