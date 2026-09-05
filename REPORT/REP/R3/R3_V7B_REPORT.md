# R3-V7B Report — Persistent Contextual Epistemic Belief

Date: 2026-08-16  
Prereg: `REPORT/REG/R3/R3_V7B_PREREG.md`  
Depends on: `REPORT/REP/R3/R3_V7A_REPORT.md`  
Artifacts: `runs/r3_v7b/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `5b8e84fed232d237c9bd5652e7a40a53ed1c4baf5750b421e3d611b39f3dae99`

## Decision

\[
\boxed{V7B\_GO=\mathrm{false}}
\]

\[
\boxed{
\text{history-aware }p_t\text{ improves calibration over V7A static }\tilde h,
\text{ but adequate-physics transients too easily raise structural belief}
}
\]

H1 and H2 pass. H3 fails. This does **not** rewrite `V7A_GO=true`.
It does **not** open V7C/D, RS5B, revision, VoI, or H32.

Smoke (`runs/r3_v7b/smoke/`): `switch_cycle` phases
\(\{0,1,2,2.5,3,3.5,5\}\); plumbing only.

## Question (unchanged)

Does maintaining history improve epistemic calibration beyond a static
episode summary? Primary baseline is **B1** (V7A-style contextual
logistic retrained on this mixture), not \(D_0\).

## Design (frozen)

- Seeds \(\{17101,17111\}\) train, \(17121\) val, \(\{17131,17141\}\) held-out.
- Mixture = V7A grid + `switch_cycle` (free → pull → release → re-grasp
  → push → quiet). \(N=105\).
- B2: GRU hidden 32, SGD, time-weighted BCE, \(p_t\) every 20 Hz step.
- No family / script / phase channels. Oracle state + oracle contact only.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 final Brier B2 \(<\) B1 | **PASS** | \(0.128<0.205\) (B0 \(0.222\)) |
| H2 midpoint Brier B2 \(<\) B1 | **PASS** | \(0.129<0.303\) |
| H3 C0 step FPR \(\le0.20\) | **FAIL** | \(0.281\) at \(\tau\approx0.606\) |

\[
\boxed{V7B\_GO=H1\land H2\land H3=\mathrm{false}}
\]

## What the numbers say

Recurrence is doing the job V7A could not: **same mixture, better
calibration**, including at a causal midpoint when the static prefix
summary is still poorly calibrated (Brier \(0.303\)).

That is evidence for

\[
\text{evidence accumulation}
\]

relative to a single \(\tilde h\).

H3 says the same memory is **too willing to raise \(p_t\) on C0
transients**. Mean false-positive dwell on held-out C0 is \(5.9\) steps;
C0 `switch_cycle` median recovery is \(0\). The failure is **entry /
transient robustness** (contact transients too easily push structural
belief up), not long hysteresis (“once suspicious, always suspicious”).

Median \(t_{\mathrm{det}}/T=0\) for both B1 and B2 on C1 (diagnostic,
not GO): both series often already exceed \(\tau\) at \(t=0\). Detection
time is uninformative here, which is why H2 used midpoint Brier.

## What this does not authorize

- enlarging to Transformer because H1/H2 passed;
- adding \(C,V\) heads or RGB/tactile;
- opening V7C;
- retuning \(\tau\) or the 0.20 cap after seeing H3;
- returning to invariant scalars or RS5 indexing.

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
V7A\_GO &= true\\
V7B\_GO &= false\\
RS5A\_GO &= true \quad\text{(side)}\\
RS5B,\ R3\text{-V7C/D} &= locked\\
\text{transport mainline} &= closed
\end{aligned}
}
\]

If continuing: **R3-V7B.1** tests the same GRU with an integrated-Brier
objective (transient-robust temporal calibration). Not V7C, not a
forgetting penalty, not a silent H3 repair of this report.
