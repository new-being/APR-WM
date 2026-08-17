# R4-I0 v2 Report — Stiffness-Swap Counterexample

Date: 2026-08-17  
Prereg: `REPORT/REG/R4_I0_PREREG.md` (protocol v2)  
v1 (retained): `REPORT/REP/R4_I0_REPORT.md`, `runs/r4_i0/closure/`  
Artifacts: `runs/r4_i0/v2/`  
Scientific result: **none** (infrastructure)

## Verdict

\[
\boxed{R4\_I0\_PASS=\mathrm{true}\ \text{(protocol v2)}}
\]

This unlocks **R4-I1** only. It does **not** set R4-C0 GO, does not
train a probe, and does not reopen Door.

v1 remains a useful no-go: scalar \(\mu\) leaked through \(F_n\).

## Protocol

\[
\mu^{A}=\mu^{B}=0.80,\qquad
\bar{\texttt{solref}}^{A}=\bar{\texttt{solref}}^{B}=0.022.
\]

Hidden variable: left/right contact-compliance swap (`solref` 0.004 vs
0.040). Same squeeze, same \(F_t(t)\) command.

## Gates (no classifier)

| Gate | Result | Detail |
|---|---|---|
| maps non-zero | **PASS** | both patterns |
| \(\mu\) / mean \(k\) matched | **PASS** | identical |
| Gate 0 macro matched | **PASS** | \(N=475\) |
| Gate 1 local split | **PASS** | \(N_{\mathrm{counterexample}}=371\); mean \(D_{\mathrm{taxel}}=1.53\) |

On a representative matched frame, global mechanics agree to numerical
noise:

\[
q,\;\dot q,\;F_n,\;F_t,\;u_{\mathrm{finger}},\;F_t^{\mathrm{cmd}}
\]

while local fields are spatial flips:

\[
\int p^{A}=\int p^{B},\qquad
p^{A}(x,y)\approx p^{B}(-x,y),
\]

and likewise \(\tau_y\). \(\|\tau^A-\tau^B\|_2>0\) but
\(\|\tau^A_{\mathrm{flipped}}-\tau^B\|_2\approx 0\).

\[
\boxed{
D(h^{S,A},h^{S,B})\approx 0
\qquad\text{and}\qquad
D(X^{A},X^{B})\gg 0
}
\]

This is the many-to-one aggregation the wrist wrench performs.

## What this is not

- not stick→incipient→gross closure (not required this round)
- not I1 AUROC
- not \(I(X;e_t\mid h^{S})\)
- shear centroid of \(|\tau|\) can stay near 0 because the pattern is a
  flip, not a one-sided blob

## Next

R4-I1 is done (`REPORT/REP/R4_I1_REPORT.md`). I0 v2 physics stay frozen.
