# R9-P0 Preregistration — Amortized Validity-Acquisition Feasibility

Date: 2026-08-17  
Status: **RAN**; **`R9_P0_PASS=true`**; belief unlocked, not implemented  
Depends on: `REPORT/REP/R8/R8_R0_REPORT.md`;
`REPORT/REP/R8/R8_SAME_HORIZON_STOP.md`  
Does not: validity detector; \(\tau_{\mathrm{valid}}\); \(\pi\); NetVoI;
\(\omega_n\) retune; \(T_{\mathrm{probe}}/u_0\) change; persistent \(b(\mathcal V)\)
implementation; change detection

## Question

If applicability evidence is acquired **once** in a billed calibration
phase (not inside the \(0.50\,\mathrm{s}\) task horizon), can that cost
be amortized over \(K\) subsequent decisions while \(F_{\max}\) is
block-stationary?

\[
\boxed{
\text{calibrate once}
\to
\text{oracle validity bookkeeping}
\to
\text{reuse across }K\text{ tasks}
}
\]

R8 same-horizon family is STOP. This is a new operational setting:
calibration is allowed its own clock.

## Frozen objects

- P0 probe: \(u=1\), \(T_{\mathrm{probe}}=0.10\,\mathrm{s}\)
- R0 PD: \(k_p=50\), \(k_d=8\) (nominal \(\alpha=2\), \(\omega_n=10\),
  \(\zeta=1\)). Not retuned.
- \(\varepsilon_{\mathrm{reset}}=2.2\times10^{-3}\)
- \(T_{\mathrm{task}}=0.50\,\mathrm{s}\) (one R7 decision episode)
- B0 \(f_{\mathrm{WM}}\), \(q\), commit-set
- Classes: \(F_{\max}\in\{2.0,2.2,1.5\}\)

## Persistence assumption (oracle)

\[
\boxed{F_{\max}\text{ is constant over }K\text{ consecutive tasks.}}
\]

No learned \(b_t(\mathcal V)\). Bookkeeping only: if the validity
state lasted \(K\) tasks, would calibration be worth it?

## Recovery time

After the frozen probe, run the frozen PD until
\(D\le\varepsilon_{\mathrm{reset}}\) or \(T_{\mathrm{recover}}^{\max}\).

\[
T_{\mathrm{recover}}
=
\inf\{t:D((y_t,\dot y_t),(0,0))\le\varepsilon\}.
\]

\[
T_{\mathrm{recover}}^{\max}=5.0\,\mathrm{s}
\]

is a calibration-session cap, not a controller sweep. Report mean/max
on the commit-set for each class. If any commit-set trajectory hits the
cap without \(D\le\varepsilon\): **stop this PD-calibration mechanism**.
Do not retune \(\omega_n\).

## Costs and break-even

\(T_{\mathrm{cal}}=T_{\mathrm{probe}}+T_{\mathrm{recover}}\).
Conservative \(T_{\mathrm{recover}}\) is the **max** over commit-set
and all three classes.

\[
V_{\mathrm{dec}}
=
\overline{|A|}_{\mathrm{commit,ID}}
\]

(same per-decision advantage as R8 G3).

\[
C_{\mathrm{time}}=\frac{T_{\mathrm{cal}}}{T_{\mathrm{task}}}V_{\mathrm{dec}},
\qquad
C_{\mathrm{effort}}
=
\beta_u\,\overline{u^2}_{\mathrm{cal}}\,\frac{T_{\mathrm{cal}}}{T_{\mathrm{task}}},
\]

\[
C_{\mathrm{cal}}=C_{\mathrm{time}}+C_{\mathrm{effort}}.
\]

Effort uses the same \(\beta_u\) as \(J\). No extra \(\beta_T\).

\[
K_{\min}
=
\min\left\{K\in\mathbb N:\frac{C_{\mathrm{cal}}}{K}<V_{\mathrm{dec}}\right\}.
\]

\[
K_{\max}=20
\]

is the longest block-stationary horizon this preflight will treat as
operationally plausible. Implied lifetime lower bound:
\(\tau_{\mathrm{validity}}\ge K_{\min}T_{\mathrm{task}}\).

## Gates

- **G-recover**: every commit-set × {ID, benign, invalid} recovers
  with \(T_{\mathrm{recover}}\le 5\,\mathrm{s}\)
- **G-amortize**: \(K_{\min}\le 20\)

Pass means oracle amortization is feasible under block-stationarity,
**not** that a persistent belief has been implemented. Fail on
G-recover: stop PD-calibration, no detector. Fail on G-amortize:
recovery works but this task margin cannot pay even a 20-task block.

## After a pass (not this stage)

Then implement persistent \(b(\mathcal V)\), then change detection /
re-probe. Not now.
