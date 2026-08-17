# R3-V7B.1 Preregistration — Transient-Robust Temporal Epistemic Calibration

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R3/R3_V7B_REPORT.md` (scientific matrix run; `V7B_GO` may stay false)  
Does not change: `V7A_GO`, `V7B_GO`, RS* GOs, C0, \(s^\star\), RS1C policy  
Locks: **V7C/D**, **RS5B/C**, revision, VoI, \(C,V\) heads, RGB, tactile  
Does **not** change GRU architecture, hidden size, or inputs

## Placement

\[
\boxed{
V7B\text{ (memory helps; C0 occupancy too high)}
\rightarrow
R3\text{-V7B.1 (this stage)}
\qquad
V7C\text{/D locked}
}
\]

Refined V7B reading (not a GO rewrite):

\[
\boxed{
\text{history helps calibration, but adequate-physics transients
too easily raise structural belief}
}
\]

This is **entry / transient robustness**, not long hysteresis
(C0 `switch_cycle` median recovery was 0; mean FP dwell \(\approx 5.9\)).

Stage name:

\[
\boxed{
R3\text{-V7B.1 — Transient-Robust Temporal Epistemic Calibration}
}
\]

中文名：**瞬态鲁棒的时序认识校准**。

## Question

\[
\boxed{
\text{Can we retain the calibration benefit of recurrent evidence
accumulation while suppressing false structural belief caused by
adequate-physics transients?}
}
\]

Hypothesis under test:

\[
\boxed{
\text{representation works; the temporal training objective is misaligned}
}
\]

Not: larger memory. Not: forgetting penalty \(\lambda\sum_{C0}p_t\).
Not: fast/slow latents (only if this stage fails).

## What stays identical to V7B B2

- GRU hidden **32**, one layer, SGD;
- 20 Hz oracle state / proprioception / \(r_\perp\) / local geometry;
- no family / script / **phase** in the model;
- same mixture including `switch_cycle`;
- \(p_{\mathrm{struct},t}\) only.

**B2** and **B3** share this architecture. The only structural change is
the training objective. Both are **retrained on this stage's data**
(fresh seeds). B2 is the V7B time-weighted BCE objective, not frozen
171xx weights.

## Objective change

**B2 (incumbent).** Time-weighted BCE:
\(w_t=0.3+0.7\,t/(T-1)\), early-stop on val **final** Brier.

**B3 (challenger).** Episode-balanced integrated Brier:

\[
\mathcal L_{\mathrm{IBS}}
=
\frac1N\sum_{i=1}^{N}
\frac1{T_i}\sum_{t=1}^{T_i}(p_{i,t}-y_i)^2.
\]

Early-stop on val **IBS**. No C0-specific occupancy penalty.

## Data

Seeds \(\{18101,18111\}\) train, \(18121\) val, \(\{18131,18141\}\)
held-out. Same job grid as V7B (\(N=105\)).

## Frozen constants

\[
\delta = 0.02
\quad\text{(absolute Brier; non-inferiority slack)}
\]

\[
\tau_{\mathrm{dev}}
=
\text{95th percentile of B2 development C0 }p_T
\]

Frozen **once** from B2 on the development split (train+val C0 finals).
The same \(\tau_{\mathrm{dev}}\) is applied to B3 for the FPR cap.
Do not refit \(\tau\) on held-out or on B3.

`t_{\mathrm{det}}\) is **abandoned** (uninformative in V7B).

## Hypotheses (GO)

**H1.** Held-out episode-balanced IBS: \(IBS_{B3}<IBS_{B2}\).

**H2.** Threshold-free C0 occupancy falls, and the V7B safety line holds
for B3 against frozen \(\tau_{\mathrm{dev}}\):

\[
B_{\mathrm{C0}}^{B3}<B_{\mathrm{C0}}^{B2},
\qquad
B_{\mathrm{C0}}=\mathbb E_{i\in C0}\!\left[\tfrac1{T_i}\sum_t p_{i,t}^2\right],
\]

\[
P(p_t>\tau_{\mathrm{dev}}\mid\text{held-out C0, all }t)\le 0.20.
\]

**H3.** Non-inferior accumulation vs B2:

\[
\mathrm{Brier}_{mid}^{B3}\le\mathrm{Brier}_{mid}^{B2}+\delta,
\qquad
\mathrm{Brier}_{final}^{B3}\le\mathrm{Brier}_{final}^{B2}+\delta.
\]

Midpoint is \(t=\lfloor(T-1)/2\rfloor\).

\[
\boxed{V7B.1\_GO = H1 \land H2 \land H3}
\]

Smoke must not set the GO.

## Diagnostics (not GO)

- \(M_{\mathrm{C0}}=\mathbb E_{C0,i}[\frac1{T_i}\sum_t p_{i,t}]\) (false-belief mass).
- \(\mathrm{Brier}(t/T)\) at \(\{0.10,0.25,0.50,0.75,1.00\}\).
- Event-aligned mean \(p_t\) on held-out C0 `switch_cycle` at first
  contact (`phase==2`), release (`3`), re-contact (`2.5`), quiet (`5`).
  Phase is **evaluator-only**.
- Mean FP dwell; recovery time (not GO).

## What success does not authorize

- V7C tactile / V7D RGB;
- forgetting penalties as the “real” method;
- Transformer / dual-GRU;
- revision / VoI / H32.

## What failure would mean

Direct temporal proper scoring is not enough to separate transient
surprise from persistent inadequacy. **Then** a new prereg may introduce
fast/slow \(u^{fast}\to u^{slow}\), \(p=H(u^{slow})\). Do not jump to
V7C on that failure.
