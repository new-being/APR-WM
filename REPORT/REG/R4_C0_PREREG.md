# R4-C0 Preregistration — Observation Necessity of Tactile given \(h^{S}\)

Date: 2026-08-17  
Status: **FROZEN** (run; `R4_C0_GO=false`; Door / R3-V7 is closed)  
Depends on: `REPORT/REP/R3_V7_STAGE_FREEZE.md`, `REPORT/REP/R3_V7D_REPORT.md`  
Does not change: any R3-V7 GO, Door C.5/V7D negatives, B5-S *recipe*,
\(e_t=y\,w_t\), GRU32 hidden, \(w_t\), \(\tau\) recipe, RS* GO, C0, \(s^\star\)  
Locks: **R3-V7E**, Door CNN/RGB/fusion reopening, DINO/CLIP, higher RGB
resolution, visuo-tactile attention, modality router, larger GRU,
feature-normalization search, encoder training in this stage

## Infrastructure lock (does not change the question)

R4-C0 formal stays locked until:

- `R4_I0_PASS` (`REPORT/REG/R4_I0_PREREG.md`) — many-to-one
  local/global counterexample (I0 v2 stiffness swap);
- `R4_I1_PASS` (`REPORT/REG/R4_I1_PREREG.md`) — over-complete \(h^{S}\)
  does not identify A vs B; tactile gain survives trajectory split.

Those stages do not train an encoder and do not replace this \(\Delta_X\)
question.

## Why this is R4, not V7E

Door mixture saturation (V7C.5 + V7D):

\[
\boxed{
h^{S}\ \text{already contains the information needed for current
Door structural-error judgment}
}
\]

Adding \(Z_{\mathrm{NSG}}\) or \(Z_{\mathrm{RGB}}\) to \(h^{S}\) was
**redundant or harmful**. Capacity shopping on Door cannot create a
conditional increment that the task does not require.

\[
\boxed{
R4\text{ — Contact-Rich Epistemic Domain}
}
\]

中文名：**接触丰富认识域**（滑动 / 摩擦 / 分布柔顺接触）。

First stage:

\[
\boxed{
R4\text{-C0 — Observation Necessity of Tactile given }h^{S}
}
\]

中文名：**在已有本体感觉接触历史上，触觉观测是否必要**。

## Domain requirement (must hold before any encoder)

Construct (or select) a contact-rich setting where two states can share
nearly the same proprioceptive summary

\[
(q,\dot q,\text{wrist wrench})
\]

but differ in local contact:

\[
\text{stable sticking}
\qquad\text{vs}\qquad
\text{incipient slip}.
\]

Then local tactile \(X\) is *in principle* not recoverable from Door-style
\(h^{S}\). Exact asset / controller / friction schedule is a later
implementation lock; this prereg freezes the **identifiability
requirement**, not a backbone.

Do **not** reuse Door episodes or Door GRU weights. Re-collect \(h^{S}\)
on the new mixture. Architecture and \(e_t\) recipe transfer; parameters
do not.

## Question

\[
\boxed{
I(X_{\mathrm{tactile}};e_t\mid h^{S})>0\ ?
}
\]

Baseline is **\(h^{S}\)**, not C.3’s \(h^{-c}\). C.5/V7D showed the live
scientific object is conditional-on-the-existing-sensor-set.

Not “train a tactile CNN.” Not “fuse RGB.” Not “which encoder.”

## Step 1 only (this stage)

Compare

\[
P_0:\ h^{S}\to e_t
\qquad\text{vs}\qquad
P_X:\ (h^{S},X_{\mathrm{tactile}})\to e_t.
\]

\(X\) is a **raw / linearly mapped** tactile observation (no learned
encoder). Same GRU32, same warranted BCE vs \(e_t=y\,w_t\).

\[
\Delta_X=L(h^{S})-L(h^{S},X_{\mathrm{tactile}}),\qquad\varepsilon=0.005.
\]

**GO** iff \(\Delta_X>\varepsilon\) on held-out seeds of the new domain.

Optional diagnostic (not a substitute for \(\Delta_X\)): time-shuffle
\(X\) inside each episode; if shuffle retains \(\Delta\), the increment
is not time-locked local contact.

## Stop / next

If \(\Delta_X\le\varepsilon\): **do not** train an encoder. **Do not**
reconstruct I0 to make B.3 \(w_t\) fire (that would undo the
macro-equivalent construction). C0 is closed.

A later stage, if any, must be a **new** prereg on a different
epistemic target (`REPORT/REG/R4_C1_PREREG.md`), not a C0 patch.

If \(\Delta_X>\varepsilon\): this stage still does **not** train a CNN.
Next would be a **new** prereg on representation sufficiency of that \(X\)
only.

## What this does not authorize

- R3-V7E or any further Door modality stuffing
- DINO / CLIP / larger vision or tactile backbones
- RGB+tactile attention, routers, bigger GRU
- treating Door C.3 \(\Delta_{\mathrm{NSG}}\) as a pass for R4
- revision, \(C,V\), object slots
