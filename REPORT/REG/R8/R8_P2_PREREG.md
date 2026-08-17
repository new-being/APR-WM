# R8-P2 Preregistration — Damping-Aware State-Returning Probe

Date: 2026-08-17  
Status: **RAN**; **`R8_P2_PASS=false`**; family frozen; A0 locked  
Depends on: `REPORT/REP/R8/R8_P1_REPORT.md`  
Last open-loop same-episode probe mechanism test.  
Does not: detector; \(\tau\); \(\pi\); NetVoI; \(T\)/\(u_0\) sweep;
undamped moments; maximize \(S_{\mathrm{epi}}\)

## Question

If the return constraint uses the **damped** kernels instead of
undamped moments, can one frozen bang-bang waveform keep P0-style
evidence and P1-style cheap \(C_{\mathrm{down}}\)?

\[
v(T)=\int_0^T e^{-c(T-t)}a(t)\,dt,\qquad
y(T)=\frac1c\int_0^T\bigl(1-e^{-c(T-t)}\bigr)a(t)\,dt.
\]

Return: both integrals \(=0\). \(c=4\), \(u_0=1\), \(T=0.10\,\mathrm{s}\).

## Waveform (solved before seeing P2 gates)

Two analytic families, both with \(\sum d_j=T\) and the two damped
constraints. Selection uses **only** \(d_{\mathrm{initial}}\), not
\(S_{\mathrm{epi}}\) or validity labels:

| template | continuous \(d\) (s) | \(d_1\) |
|---|---|---|
| \([+,-,+]\) unique | \((0.02625,0.05000,0.02375)\) | \(0.02625\) |
| \([+,-,+,-]\) max \(d_1\) | \((0.02524,0.04823,0.02476,0.00177)\) | \(0.02524\) |

Chosen: three-phase \([+,-,+]\). Integer snap to \(\mathrm{d}t=0.002\),
\(n=50\):

\[
n=(13,25,12).
\]

This is a long informative prefix relative to the other feasible
damped-return solution, plus a recovery tail. Not a G1-tuned probe.

Predictor of \(a^{\mathrm{epi}}\): frozen ID plant \(F_{\max}=2.0\).
\(S_{\mathrm{epi}}=\mathrm{RMSE}_t(y-\hat y_{\mathrm{ID}})\).

## Gates (unchanged)

G0, G1 (\(\varepsilon_S=10^{-3}\)), G2, G-return (\(2.2\times10^{-3}\)),
G3 (ratio \(\le 1\)). Same ID / benign / invalid \(F_{\max}\) and B0
commit-set.

If all five pass: unlock R8-A0.  
If G-return and G3 pass but G1 fails: freeze this probe family;
applicability is observable (P0) but tested returning open-loop probes
cannot acquire enough evidence cheaply. Then a billed reset protocol
may be considered — not a fifth waveform.
