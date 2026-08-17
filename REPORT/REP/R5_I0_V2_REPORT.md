# R5-I0 v2 Report — Self-Equilibrated Local Prestress

Date: 2026-08-17  
Status: **`R5_I0_V2_PASS=false`**  
Prereg: `REPORT/REG/R5_I0_PREREG.md`  
Artifacts: `runs/r5_i0/prestress/summary.json`  
v1 retained: `REPORT/REP/R5_I0_REPORT.md`, `runs/r5_i0/formal/`  
Does not change: R4 STOP  
Does not train: any probe  
Does not move: \(t_\star\) into the probe

## Question

At hold-end \(t_\star\) (\(F_t^{\mathrm{cmd}}=0\)), does an intra-pad
opposing shear

\[
\tau_{x<0}(\lambda)=+\lambda,\qquad \tau_{x>0}(\lambda)=-\lambda
\]

give, on one \(\lambda\) interval,

\[
\partial h^{S}/\partial\lambda\approx 0,\quad
\partial X/\partial\lambda\neq 0,\quad
\partial C/\partial\lambda\neq 0
\]

with net signed wrench targeted at zero? No learner.

## Construction

Each pad is split into \(x<0\) / \(x>0\) sliding halves. Hold uses
feedforward \(\pm\lambda\) plus PD to zero half-displacement. Pad–pad
collisions are excluded (`contype`/`conaffinity`); the block \(y\)-face
covers the pad. Frozen probe remains \(F_t^{\mathrm{cmd}}=4.8\) for
\(0.40\,\mathrm{s}\).

\(X_{\mathrm{stat}}\) is the signed \(\tau\) difference of \(x\)-halves.
On this \(y\)-normal contact MuJoCo stores world-\(x\) shear as `tau_y`.

## Result

\(\lambda\in\{0,0.15,\ldots,0.90\}\).

| Gate | Value | Pass |
|---|---|---|
| G_macro \(\max D(h^{S}_\lambda,h^{S}_0)\) | **0.392** | false |
| G_local Spearman\((\lambda,X_{\mathrm{stat}})\) | **0.464** | false |
| G_cons Spearman\((\lambda,C)\) | **0.679** | false |

\[
\boxed{\texttt{R5\_I0\_V2\_PASS}=\text{false}}
\]

Curves (hold \(t_\star\)):

| \(\lambda\) | \(D_h\) | \(X_{\mathrm{stat}}\) | \(C\) | \(F_x\) | \(M_z\) | \(F_t=\sum\|\tau\|\) |
|---|---|---|---|---|---|---|
| 0.00 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.15 | 0.173 | 0.256 | 0.076 | \(\sim 0\) | \(\sim 0\) | 0.256 |
| 0.30 | 0.291 | 0.430 | 0.075 | \(\sim 0\) | \(\sim 0\) | 0.430 |
| 0.45 | 0.368 | 0.543 | 0.072 | \(\sim 0\) | \(\sim 0\) | 0.543 |
| 0.60 | 0.383 | 0.565 | 0.075 | \(\sim 0\) | \(\sim 0\) | 0.565 |
| 0.75 | 0.392 | 0.579 | 0.077 | \(\sim 0\) | \(\sim 0\) | 0.579 |
| 0.90 | 0.062 | 0.072 | 0.078 | \(\sim 0\) | \(\sim 0\) | 0.072 |

Signed net \(F_x\) and \(M_z\) cancel to numerical noise. Unsigned
shear and \(X_{\mathrm{stat}}\) are the same monotone rise on
\([0,0.75]\), then both collapse at \(\lambda=0.90\) (prestress itself
slips). \(C(\lambda)\) stays in a narrow band \(\approx 0.07\)–\(0.08\)
and is not ordered. \(\max D\) on signed \((F,M)\) is \(0.054\)
(fy drift); official \(D(h^{S})\) is large because \(h^{S}\) includes
\(f_t=\sum|\tau|\).

## Interpretation

v1 failed because the future-consequential DOF was not in the current
tactile snapshot. v2 **does** put a higher-order couple into \(X\) at
\(t_\star\), and that couple is **macro-null in signed wrench**.

It still fails admission, for two separate reasons:

1. **This \(h^{S}\) is not a pure signed spatial integral.**
   \(I1\) keys include \(f_t=\sum|\tau|\), which tracks the couple.
   So \(\partial h^{S}/\partial\lambda\neq 0\) even while
   \(\partial F_x/\partial\lambda\approx\partial M_z/\partial\lambda\approx 0\).
2. **The frozen co-directional probe does not turn that couple into
   ordered \(C(\lambda)\).** Local \(X\) varies; future macros barely
   do, and not monotonically. At large \(\lambda\) the prestress
   saturates and \(X\) collapses, which also kills Spearman.

Do not drop the last \(\lambda\), drop \(f_t\) from \(h^{S}\), or move
\(t_\star\) into the probe in order to pass. Those would be a new I0,
not this gate.

No neural probe. R4 stays stopped. v3 quadrupole \(k_t\) is in
`REPORT/REP/R5_I0_V3_REPORT.md` (`PASS=false`; family pause).
