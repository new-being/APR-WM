# R7-P0 Report — Fresh Switchability Preflight

Date: 2026-08-17  
Status: **`R7_P0_PASS=true`** (R7-A0 unlocked, **not implemented**)  
Prereg: `REPORT/REG/R7/R7_P0_PREREG.md`  
Artifacts: `runs/r7_p0/formal/summary.json`  
Does not train: \(\hat A\), LCB, commit rule, encoder

## Family

\[
\ddot y=\alpha u-c\dot y,\qquad
h^{S}=(y,\dot y,\ddot y,u)\ \text{on hold }u=0,
\qquad
X=0.25\alpha.
\]

Default \(u=1.0\) (mid), alternative \(u=0.4\) (cons).
\(A=J_{\mathrm{default}}-J_{\mathrm{alt}}\). Self-stress \(\lambda\) unused.

## Gates

| Gate | Value | Pass |
|---|---|---|
| G0 \(\max D_h\) | **0** | true |
| G1 \(\rho(\alpha,X)\) | **1** | true |
| G2 both signs of \(A\) | \(A<0\) on \(\alpha\le 1.8\); \(A>0\) on \(\alpha\ge 2.2\) | true |
| G3 \(\lvert A\rvert>10^{-3}\) | 4 harmful + 4 useful | true |
| G4 coverage | 4 / 4 (not 1 vs 20) | true |

\[
\boxed{\texttt{R7\_P0\_PASS}=\text{true}}
\]

G1 is **by construction** (\(X\propto\alpha\)). That is intended: P0 tests
switchability of \(A\), not modality acquisition.

## \(A(\alpha)\) (oracle)

| \(\alpha\) | \(X\) | \(y_T\) mid | \(y_T\) cons | \(A\) | switch useful |
|---|---|---|---|---|---|
| 0.6 | 0.15 | 0.043 | 0.017 | \(-3.58\times10^{-3}\) | no |
| 1.0 | 0.25 | 0.071 | 0.028 | \(-4.28\times10^{-3}\) | no |
| 1.4 | 0.35 | 0.099 | 0.040 | \(-3.62\times10^{-3}\) | no |
| 1.8 | 0.45 | 0.128 | 0.051 | \(-1.62\times10^{-3}\) | no |
| 2.2 | 0.55 | 0.156 | 0.062 | \(+1.75\times10^{-3}\) | yes |
| 2.6 | 0.65 | 0.184 | 0.074 | \(+6.46\times10^{-3}\) | yes |
| 3.0 | 0.75 | 0.213 | 0.085 | \(+1.25\times10^{-2}\) | yes |
| 3.4 | 0.85 | 0.241 | 0.097 | \(+2.00\times10^{-2}\) | yes |

Crossover sits between 1.8 and 2.2. Weak \(\alpha\): cons undershoots,
keep default. Strong \(\alpha\): default overshoots \(y^\star=0.10\),
cons is closer. Margins are \(O(10^{-3})\)–\(O(10^{-2})\), not R6’s
\(10^{-8}\) knife edge. Both signs have four G3-qualified regimes.

## What this does not do

No \(\hat A(X,h^{S})\). No one-sided \(L(X)\le A\). No
\(L>\delta\Rightarrow\mathrm{commit}\). R6 realization stays frozen.

R7-A0 is **run** (`REPORT/REP/R7/R7_A0_REPORT.md`): `GO=true`.
One-sided \(L=\hat A-q\), \(\delta=10^{-3}\). Shuffle-\(X\) VoI is 0.
