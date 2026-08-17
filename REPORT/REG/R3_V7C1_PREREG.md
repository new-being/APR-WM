# R3-V7C.1 Preregistration — Learned Tactile Epistemic Representation

Date: 2026-08-16  
Status: **FROZEN, LOCKED** (not opened; V7C success does **not** unlock)  
Depends on: `REPORT/REP/R3_V7C_REPORT.md` (`V7C_GO=true`),
`REPORT/REP/R3_V7_STAGE_FREEZE.md`  
Does not change: V7A–V7C GOs, RS* GOs, C0, \(s^\star\), RS1C policy  
Locks: **V7D**, RGB, object slots, revision, VoI, \(C,V\), occupancy
penalties, GRU-gate search, \(\tau\) refit, V7C contact-proxy changes

## Placement

\[
\boxed{
V7C\text{ (noisy proprioceptive contact)}\ \checkmark
\rightarrow
R3\text{-V7C.1 (this candidate)}
\qquad
V7D\text{ locked}
}
\]

This is **not** V7D. It is one step on the sensor ladder:

\[
\boxed{
\text{low-dimensional proprioceptive contact}
\rightarrow
\text{tactile representation}
\rightarrow
\text{RGB/object belief}
}
\]

Stage name:

\[
\boxed{
R3\text{-V7C.1 — Learned Tactile Epistemic Representation}
}
\]

中文名：**学习触觉表示下的证据正当化认识信念**。

## Question

\[
\boxed{
\text{Does a learned tactile representation preserve evidence-warranted
epistemic calibration?}
}
\]

Hard question is **not** “can the encoder reconstruct contact?”

\[
\boxed{
\text{does tactile representation preserve epistemic semantics?}
}
\]

## Single manipulated variable

Keep frozen:

- B5 single GRU32;
- \(y\cdot w_t\) supervision (train-only \(w_t\));
- Door mixture; matched C0/C1 pairs;
- oracle \(q,\dot q\) / world residual for \(\theta\) and for \(w_t\);
- no phase/family input;
- no revision; no \(C,V\); no RGB; no object slots.

Replace **only**:

\[
\text{hand-crafted low-dimensional contact proxy (B5-S)}
\]

by

\[
z_t^{\mathrm{tactile}}=f_\phi(x_t^{\mathrm{tactile}})
\]

then

\[
u_{t+1}^{\mathrm{epi}}
=
F\!\left(u_t^{\mathrm{epi}},q_t,\dot q_t,a_t,z_t^{\mathrm{tactile}},\ldots\right).
\]

\(x_t^{\mathrm{tactile}}\) is a learner-visible tactile field (taxel /
pressure map at the gripper), **not** hinge \(J^\top f\) and **not**
the contact-pair list used as a model input. Encoder \(f_\phi\) is
trained for the epistemic objective, not as a contact reconstructor.

Primary baseline: **B5-S** from V7C (frozen weights or retrained on
the new seed split under the same B5-S recipe). Ablation: B5-N
(no tactile / no contact channel). Oracle contact remains a ceiling,
not the GO target.

## Hypotheses (GO, if this stage is later unlocked)

Reuse V7C semantics vs **B5-S**, \(\delta=0.02\), C0 FPR \(\le 0.20\),
\(\tau_{\mathrm{dev}}\) from development C0 of the tactile arm.

**H1.** Mid/final Brier vs \(y\):
\(\mathrm{Brier}^{tactile}\le\mathrm{Brier}^{S}+\delta\).

**H2.** \(P(p_t>\tau_{\mathrm{dev}}\mid C0)\le 0.20\).

**H3.** Held-out C1: mean Spearman\((p_t,w_t)>0\) and
\(\mathbb E[p\mid w_{\mathrm{high}}]>\mathbb E[p\mid w_{\mathrm{low}}]\).

**H4.** Tactile is not an empty channel vs B5-N, by the same
metric-dependent rule as V7C (pooled Brier **or**
\(B_{\mathrm{C0}}\) + C1-final).

Reconstruction error of contact / force is **diagnostic only**. If
reconstruction is good but H3 fails, the stage **fails**.

\[
\boxed{V7C.1\_GO=H1\land H2\land H3\land H4}
\]

Smoke must not set the GO. This file does **not** authorize a run.

## What this file does not authorize

RGB, V7D, mixing tactile with object slots, changing \(w_t\)
semantics, or treating contact reconstruction as success.
