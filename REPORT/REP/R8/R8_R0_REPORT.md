# R8-R0 Report — Billed Probe→Reset Feasibility

Date: 2026-08-17  
Status: **`R8_R0_PASS=false`**; pattern **recovery_failure**;
R8-A0 **locked**; billed probe→reset family **STOP**  
Prereg: `REPORT/REG/R8/R8_R0_PREREG.md`  
Artifacts: `runs/r8_r0/formal/summary.json`

## Question

Can the frozen P0 one-way probe keep its validity evidence if a
nominal, state-only PD reset restores \((y,\dot y)\to(0,0)\) inside
the original \(0.50\,\mathrm{s}\) task horizon?

Probe unchanged: \(u=1\), \(T_{\mathrm{probe}}=0.10\,\mathrm{s}\).
Reset: \(k_p=50\), \(k_d=8\) from \(\alpha_{\mathrm{des}}=2\),
\(\omega_n=10\), \(\zeta=1\). Not fit on \(F_{\max}=1.5\) or \(S_{\mathrm{epi}}\).

## Gates

| Gate | result |
|---|---|
| G1 pre-reset gap \(\ge 10^{-3}\) | **✓ \(1.82\times10^{-3}\)** (P0 reproduced) |
| G-reset all classes \(D\le 2.2\times10^{-3}\) | **× max \(D=1.61\times10^{-2}\)** |
| G-cost \(\overline{C}_{\mathrm{epi}}/\overline{|A|}\le 1\) | × \(4.60\) (not the primary locus) |
| G-indep \(\pi_{\mathrm{reset}}(y,\dot y)\) only | **✓** |

\[
\boxed{\texttt{R8\_R0\_PASS}=\text{false},\quad
\text{pattern}=\text{recovery\_failure}}
\]

Commit-set \(S_{\mathrm{epi}}\) before reset: ID
\([3.4,7.1]\times10^{-4}\), benign \([1.7,5.4]\times10^{-4}\),
invalid \([2.53,2.91]\times10^{-3}\). Sensing is still P0.

Reset used the **entire leftover** \(T_{\mathrm{reset}}=0.40\,\mathrm{s}\)
on every commit-set class and still missed \(\varepsilon\):

| class | max \(D_{\mathrm{reset}}\) |
|---|---|
| ID | \(1.46\times10^{-2}\) |
| benign \(F_{\max}=2.2\) | \(1.61\times10^{-2}\) |
| invalid \(F_{\max}=1.5\) | \(1.10\times10^{-2}\) |

The frozen critically damped PD is slow relative to
\(\varepsilon=2.2\times10^{-3}\) once the \(0.10\,\mathrm{s}\) probe has
taken its share of \(T_{\mathrm{total}}\). This is not a reason to sweep
\(\omega_n\).

## Reading

\[
\boxed{
\text{certificate applicability is observable (P0),}
\quad
\text{but a billed same-horizon probe}\to\text{nominal reset
does not recover the decision state in time.}
}
\]

Outcome cell 3 of the prereg: evidence ✓, reset ×. Do not train a
validity detector. Do not enumerate another open-loop waveform.
R8-A0 stays locked.

Next is **R9-P0** (`REPORT/REP/R9/R9_P0_REPORT.md`): amortized
calibration on a separate clock. Same-horizon family STOP:
`REPORT/REP/R8/R8_SAME_HORIZON_STOP.md`.
