# R3-V7B.3 Preregistration — Causal Evidence-Warranted Epistemic Belief

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R3/R3_V7B2_REPORT.md` (scientific matrix; GO may stay false)  
Does not change: V7A/B/B.1/B.2 GOs  
Locks: **V7C/D**, RS5B, revision, VoI, \(C,V\), RGB, tactile, Transformer,
fast/slow architecture changes

## Placement

\[
\boxed{
V7B.2\text{ (fast/slow occupancy not solved)}
\rightarrow
R3\text{-V7B.3 (this stage)}
\qquad
V7C\text{ locked}
}
\]

New hypothesis (not another GRU gate):

\[
\boxed{
\text{persistent epistemic belief needs supervision about when evidence
becomes sufficient, not merely whether the hidden model class is wrong}
}
\]

Stage name:

\[
\boxed{
R3\text{-V7B.3 — Causal Evidence-Warranted Epistemic Belief}
}
\]

中文名：**因果证据充分性约束的认识信念**。

## Question

\[
\boxed{
\text{Does explicitly supervising when structural evidence becomes
causally available reduce C0 interaction baseline without destroying
history-based calibration?}
}
\]

Architecture is **B2 single GRU** for both arms. Only supervision changes.

## Matched counterfactual pairs

Same seed, initial interior state, action tape, script/amp. Only hidden
dynamics change. Contact: record C0 OSC tape, `replay_rollout` on C1-L
and C1-H. Mode-A: same torque series.

Runtime model sees only the live \(h_{1:t}\). It does **not** see the
C0 counterfactual, \(w_t\), phase, or family.

## Evidence weight (train-only)

\[
A_t=\sum_{k\le t}(r_k^{C1}-r_k^{C0})^2,
\qquad
w_t^{\mathrm{evid}}=A_t/(A_T+\varepsilon).
\]

C0 rows: target \(=0\). C1 rows: target \(=w_t\) (warranted belief:
\(z^{\mathrm{struct}}=1\) and evidence available). Residuals for \(A_t\)
are noiseless simulator residuals after 20 Hz downsample.

## Models

**B2.** Time-weighted BCE vs episode \(y\in\{0,1\}\) (V7B).  
**B5.** Same GRU, BCE vs \(y_t^{\mathrm{war}}=y\cdot w_t\) (C0: \(0\);
C1: \(w_t\)), episode-mean over time. Early-stop: B2 on val final Brier
vs \(y\); B5 on val mean BCE vs \(y_t^{\mathrm{war}}\).

## Data

Seeds \(\{20101,20111\}\) train, \(20121\) val, \(\{20131,20141\}\)
held-out. Same 7 jobs × 5 seeds × 3 regimes \(=105\), now **paired**.

## Frozen constants

\[
\delta=0.02,\quad
\gamma=0.05,\quad
w_{\mathrm{low}}=0.25,\quad
w_{\mathrm{high}}=0.75.
\]

\(\tau_{\mathrm{dev}}=\) 95th percentile of **B2** development C0 \(p_T\),
frozen once.

## Hypotheses (GO)

**H1.** Mid/final Brier vs episode \(y\) on held-out:
\(\mathrm{Brier}^{B5}\le\mathrm{Brier}^{B2}+\delta\) at mid and final.

**H2.** \(B_{\mathrm{C0}}^{B5}<B_{\mathrm{C0}}^{B2}\) and
\(P(p_t>\tau_{\mathrm{dev}}\mid C0)\le 0.20\).

**H3.** Held-out C1: mean episode Spearman\((p_t,w_t)>0\), and
\(\mathbb E[p_t\mid w_t>w_{\mathrm{high}}]
>
\mathbb E[p_t\mid w_t<w_{\mathrm{low}}]\).

**H4.** Matched pairs, \(w\) from the C1 member:
\(\mathbb E[|p^{C1}-p^{C0}|\mid w<w_{\mathrm{low}}]\le 0.20\), and
\(\mathbb E[p^{C1}-p^{C0}\mid w>w_{\mathrm{high}}]
>
\mathbb E[p^{C1}-p^{C0}\mid w<w_{\mathrm{low}}]+\gamma\).

\[
\boxed{V7B.3\_GO=H1\land H2\land H3\land H4}
\]

Smoke must not set the GO.

## Stopping rule (predeclared)

If this stage fails H2, **stop** further recurrent-architecture search
on this oracle Door mixture. Accept:

\[
\text{persistent structural belief remains incompletely calibrated
on this sandbox.}
\]

Do not open V7B.4 gates, occupancy penalties, Transformer, or V7C as
the default next step. Reassess data coverage / observational
uncertainty / returning to V7A static contextual belief.

## What success does not authorize

V7C, phase inputs, using \(w_t\) at runtime, revision/VoI.
