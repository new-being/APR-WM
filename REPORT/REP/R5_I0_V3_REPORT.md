# R5-I0 v3 Report — Integral-Constrained \(k_t(x;\lambda)\)

Date: 2026-08-17  
Status: **`R5_I0_V3_PASS=false`** (G_cons fail)  
Prereg: `REPORT/REG/R5_I0_PREREG.md`  
Artifacts: `runs/r5_i0/kt/summary.json`  
v1 / v2 retained: `REPORT/REP/R5_I0_REPORT.md`, `REPORT/REP/R5_I0_V2_REPORT.md`  
Does not change: R4 STOP  
Does not train: any probe  
Does not pass by: deleting \(f_t\); deleting endpoints; moving \(t_\star\);
raising \(F_t^{\mathrm{probe}}\)

## Question

Does a quadrupole rotation of per-taxel `solref`

\[
k_t(\lambda)=k_0+\rho(\cos\lambda\,\phi_1+\sin\lambda\,\phi_2)
\]

with \(\phi_1=x^{2}-z^{2}\), \(\phi_2=2xz\) (mean-zero, dipole-zero), a
**fixed** hold load \(F_t=0.48\), and the **frozen** probe \(F_t=4.8\),
give on \(\lambda\in[0,\pi/2]\)

\[
D(h^{S})\approx 0,\qquad
X_{\mathrm{stat}}\text{ ordered},\qquad
C\text{ ordered}?
\]

No learner.

## Result

| Gate | Value | Pass |
|---|---|---|
| G_macro \(\max D(h^{S}_\lambda,h^{S}_0)\) | **\(4.5\times 10^{-4}\)** | true |
| G_local Spearman\((\lambda,X_{\mathrm{stat}})\) | **0.857** | true |
| G_cons Spearman\((\lambda,C)\) | **\(-0.214\)** | false |

\[
\boxed{\texttt{R5\_I0\_V3\_PASS}=\text{false}}
\]

| \(\lambda\) | \(D_h\) | \(X_{\mathrm{stat}}=\langle\tau,\phi_2\rangle\) | \(C\) | \(\sum\|\tau\|\) |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.480 |
| 0.26 | \(1.8\times10^{-4}\) | 0.030 | 0.625 | 0.480 |
| 0.52 | \(2.2\times10^{-4}\) | 0.062 | 0.663 | 0.480 |
| 0.79 | \(8.0\times10^{-5}\) | 0.097 | 0.624 | 0.480 |
| 1.05 | \(2.3\times10^{-4}\) | 0.108 | 0.610 | 0.480 |
| 1.31 | \(4.0\times10^{-4}\) | 0.105 | 0.609 | 0.480 |
| \(\pi/2\) | \(4.5\times10^{-4}\) | 0.103 | 0.609 | 0.480 |

Mean `solref` is identically \(0.022\). Signed \(F_x\) stays at the
commanded hold load. \(\sum|\tau|\) is constant (same-sign orthant).
Official \(h^{S}\) therefore really is approximately invariant:
this is the nullspace construction working, including v2's missing
\(\sum|\tau|\) constraint.

## Interpretation

v3 **does** what v1 and v2 each missed, at once:

- current tactile sees a continuous spatial coordinate (\(X\) ordered);
- learner-visible \(h^{S}=A(x)\) does not move.

It still fails admission, for the same independent reason as v2's
second failure:

\[
X(\lambda)\text{ varies}
\not\Rightarrow
C(\lambda)\text{ is ordered}.
\]

\(C\) jumps once off \(\lambda=0\) into a narrow band \(\approx 0.61\)–\(0.66\)
and then slightly **decreases**. The frozen diagnostic action is sensitive
to “not being pure \(\phi_1\)”, not to the continuous mixing angle that
\(X_{\mathrm{stat}}\) tracks.

That is R4's lesson in continuous form: a currently visible spatial
pattern is not automatically a consequence-varying coordinate.

## Stopping

Triangle (this family):

\[
\begin{array}{c|ccc}
& h^{S}\text{ null} & X\text{ ordered} & C\text{ ordered}\\
\hline
v1 & \checkmark & \times & \checkmark\\
v2 & \times & \text{partial} & \times\\
v3 & \checkmark & \checkmark & \times
\end{array}
\]

v3 excludes “nullspace of official \(h^{S}\) is sufficient”. The
remaining failure is

\[
I(X;\lambda\mid h^{S})>0
\not\Rightarrow
I(X;C\mid h^{S})>0.
\]

Do not enumerate a fourth contact pattern. Do not upgrade this to an
impossibility theorem for all contact systems.

No neural probe. R4 stays stopped. Canonical pause:
`REPORT/REP/R5_CONTACT_FAMILY_PAUSE.md`.
