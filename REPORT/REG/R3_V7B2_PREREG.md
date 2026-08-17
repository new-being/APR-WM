# R3-V7B.2 Preregistration — Fast/Slow Epistemic State Separation

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R3_V7B1_REPORT.md` (scientific matrix; GO may stay false)  
Does not change: `V7A_GO`, `V7B_GO`, `V7B.1_GO`, RS* GOs  
Locks: **V7C/D**, **RS5B**, revision, VoI, \(C,V\), RGB, tactile, Transformer  
Does **not** change the V7B learner-visible input or the BCE objective

## Placement

\[
\boxed{
V7B.1\text{ (IBS insufficient)}
\rightarrow
R3\text{-V7B.2 (this stage)}
\qquad
V7C\text{ locked}
}
\]

Refined hypothesis (not a silent H3 repair of V7B):

\[
\boxed{
\text{history is useful, but a single-timescale latent does not
separate transient interaction evidence from persistent inadequacy}
}
\]

Stage name:

\[
\boxed{
R3\text{-V7B.2 — Fast/Slow Epistemic State Separation}
}
\]

中文名：**快慢双时间尺度认识状态分离**。

## Question

\[
\boxed{
\text{Can explicitly separating transient surprise from persistent
inadequacy retain accumulation while reducing C0 false-belief occupancy?}
}
\]

All recurrent models use the **same time-weighted BCE** as V7B B2.
Only representation changes. Phase / family / script are **not** model
inputs (evaluator-only for H3 events).

## Representation (B4)

\[
u_t^{fast}=F_f(u_{t-1}^{fast},x_t)
\quad\text{(GRU)}
\]

\[
g_t=\sigma(W_g[u_t^{fast},u_{t-1}^{fast},u_{t-1}^{slow}])
\]

\[
\tilde u_t^{slow}=\tanh(W_c[u_{t-1}^{slow},u_t^{fast}])
\]

\[
u_t^{slow}=(1-g_t)\,u_{t-1}^{slow}+g_t\,\tilde u_t^{slow}
\]

\[
p_{\mathrm{struct},t}=H(u_t^{slow})
\quad\text{never }H(u^{fast}).
\]

\(g_t\) is a **scalar persistence gate**. Fast GRU hidden = slow dim =
**32**. \(x_t\) is the V7B 12-D step vector.

## Baselines (same data, same BCE)

**B2.** Single GRU hidden 32 (V7B architecture).  
**B2-wide.** Single GRU hidden **40**, parameter count matched to B4
within 5% (B4 \(\approx 6626\), B2-wide \(\approx 6521\)).  
**B4.** Fast/slow + persistence gate.

Primary comparison is **B4 vs B2**. B2-wide is the capacity control
(H4). Static V7A / \(D_0\) are not primary.

## Data

Seeds \(\{19101,19111\}\) train, \(19121\) val, \(\{19131,19141\}\)
held-out. Same 105-episode mixture as V7B including `switch_cycle`.

## Frozen constants

\[
\delta=0.02
\quad\text{(Brier non-inferiority vs B2)}
\]

\[
\tau_{\mathrm{dev}}
=
\text{95th percentile of B2 development C0 }p_T
\]

Frozen once from B2 on train+val C0 finals; applied to B4 (and logged
for B2-wide). Do not refit on held-out or on B4.

**Event rise** on C0 `switch_cycle` (evaluator phase only):

\[
\Delta p_e
=
\max\!\left(0,\;
\overline{p}_{[t_e,t_e+2]}
-
\overline{p}_{[t_e-4,t_e)}\
\right)
\]

for \(e\in\{\)first contact \(2\), release \(3\), re-contact \(2.5\}\).
Quiet is diagnostic, not in the max. Per episode: \(\Delta p=\max_e\Delta p_e\).

\[
\eta
=
\operatorname{median}_{i\in\mathrm{dev,C0,switch}}
\Delta p_i^{B2}
\]

Frozen from **B2 on development** after B2 is trained, before scoring
held-out B4. Not taken from V7B.1 formal numbers.

## Hypotheses (GO)

**H1 — keep accumulation vs B2.**

\[
\mathrm{Brier}^{B4}_{mid}\le\mathrm{Brier}^{B2}_{mid}+\delta,
\quad
\mathrm{Brier}^{B4}_{final}\le\mathrm{Brier}^{B2}_{final}+\delta,
\]

and **at least one** of mid / final is strictly better than B2.

**H2 — C0 occupancy vs B2.**

\[
B_{\mathrm{C0}}^{B4}<B_{\mathrm{C0}}^{B2},
\qquad
P(p_t>\tau_{\mathrm{dev}}\mid\text{held-out C0})\le 0.20.
\]

\(B_{\mathrm{C0}}\) is episode-balanced \(\mathbb E_i[\frac1{T_i}\sum_t p_t^2]\)
on C0.

**H3 — slow belief bounded at C0 interaction events.**

Held-out C0 `switch_cycle` median \(\Delta p^{B4} < \eta\).

Fast-state L2 change at first contact is diagnostic only (must be
allowed to rise).

**H4 — not just extra recurrent capacity.**

\[
B_{\mathrm{C0}}^{B4}<B_{\mathrm{C0}}^{B2\text{-wide}}.
\]

\[
\boxed{V7B.2\_GO=H1\land H2\land H3\land H4}
\]

Smoke must not set the GO.

## Diagnostics (not GO)

- \(M_{\mathrm{C0}}\); FP dwell; recovery.
- Event-aligned mean \(p\) and mean \(\|u^{fast}\|\) at contact / release /
  recontact / quiet.
- \(\mathrm{Brier}(t/T)\) at \(\{0.1,0.25,0.5,0.75,1.0\}\).
- Parameter counts of B2 / B2-wide / B4.

## What success does not authorize

- V7C tactile or V7D RGB without a new prereg;
- feeding phase into the model;
- claiming Transformer / dual-GRU-for-capacity is the result (H4 exists
  to block that);
- revision / VoI / H32.

## What failure would mean

Explicit fast/slow + persistence gating, under matched BCE and a
capacity control, does not buy C0 occupancy or event-bounded slow
belief. Then the Door oracle mixture may not identify time-scale
separation, or the gate may need a different inductive bias — **new
prereg**, not occupancy penalties and not V7C.
