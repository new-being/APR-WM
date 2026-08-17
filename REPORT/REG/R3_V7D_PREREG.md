# R3-V7D Preregistration — Conditional Modality Value of RGB

Date: 2026-08-16  
Status: **FROZEN** (run complete; `V7D_GO=false`)  
Depends on: `REPORT/REP/R3_V7C5_REPORT.md` (scientific result; `V7C.5_GO=false` is **not** a lock),
`REPORT/REP/R3_V7_EVIDENCE_CHAIN.md`  
Does not change: V7A–V7C.5 results, B5-S, \(e_t=y\,w_t\), GRU32 hidden, \(w_t\), \(\tau\) recipe, C0, \(s^\star\)  
Locks: V7C.6, Door tactile CNN/fusion, visuo-tactile fusion, RGB encoder search,
object slots, revision, \(C,V\), domain/family ID

## Placement

\[
\boxed{
\begin{array}{c}
\text{context}\\
\downarrow\\
\text{history}\\
\downarrow\\
\text{evidence-warranted supervision}\\
\downarrow\\
\text{imperfect proprioceptive contact}\\
\downarrow\\
\text{conditional modality value}\\
\downarrow\\
R3\text{-V7D RGB (this stage)}
\end{array}
}
\]

Door tactile C.3–C.5 is a **negative baseline**, not a reason to reopen
CNN search:

\[
I(X_{\mathrm{NSG}};e_t\mid h^{-c})>0
\qquad\text{but}\qquad
I(Z_{\mathrm{NSG}};e_t\mid h^{S})\le 0.
\]

There is **no V7C.6**.

Stage name:

\[
\boxed{
R3\text{-V7D — Conditional Modality Value of RGB}
}
\]

中文名：**视觉相对已有本体感觉接触历史的条件模态价值**。

## Question

\[
\boxed{
I(Z_{\mathrm{RGB}};e_t\mid h^{S})>0\ ?
}
\]

Not “did pooled Brier drop after stuffing RGB.” Not “which vision
encoder is stronger.”

In words: given frozen B5-S history (proprioception, action, residual,
noisy contact), does vision still supply **extra evidence that is
relevant to warranted structural belief**?

## Tactile negative control (frozen)

\[
\boxed{
\text{a modality can have information}
\neq
\text{that modality deserves weight in the current belief}
}
\]

V7D tests whether RGB is in the second class given \(h^{S}\), using
the same allocation test as C.5.

## Frozen (do not change)

- B5-S 12-D recipe and GRU32;
- target \(e_t=y\,w_t\) (runtime never sees \(w_t\));
- \(\delta=0.02\), C0 FPR \(\le 0.20\), \(\varepsilon=0.005\),
  \(\varepsilon_{\mathrm{temporal}}=0.005\);
- matched C0/C1 Door mixture; no domain/family/phase input;
- no occupancy penalty; no GRU-gate search; no \(\tau\)/\(w_t\) refit.

## Single manipulated variable

\[
h_t^{S}
\;\longrightarrow\;
(h_t^{S},\,Z_{\mathrm{RGB},t})
\]

**RGB observation (frozen).** Learner-visible offscreen `agentview`,
\(32\times 32\times 3\), same \(20\,\mathrm{Hz}\) as B5 steps. No
segmentation, no object-pose GT, no privileged door inertia in the
image pipeline.

**\(Z_{\mathrm{RGB}}\) encoder (frozen architecture, not a search).**
One tiny per-frame CNN, capacity-matched to V7C.4’s tactile encoder:
`Conv(3→8) → Tanh → Conv(8→8) → Tanh → AdaptiveAvgPool(2) → Flatten
→ Linear(32→2) → Tanh`. Concatenate \(z_t\in\mathbb{R}^{2}\) to B5-S.
Trained only with the warranted BCE. No reconstruction loss.

## Arms

1. **B5-S**
2. **B5-S + real \(Z_{\mathrm{RGB}}\)**
3. **B5-S + temporally shuffled \(Z_{\mathrm{RGB}}\)**
   (shuffle RGB/\(z\) **inside each episode**; B5-S unshuffled)

## Metrics (C.5 inheritance)

\(L\) = held-out BCE vs \(e_t\).

\[
\Delta_{\mathrm{real}}=L(S)-L(S{+}Z_{\mathrm{RGB}}),\qquad
\Delta_{\mathrm{shuf}}=L(S)-L(S{+}Z_{\mathrm{shuf}}).
\]

Report also \(L(S{+}Z)-L(S)\) (sign-flipped convenience). Primary
scientific objects are \(\Delta_{\mathrm{real}}\) and
\(\Delta_{\mathrm{real}}-\Delta_{\mathrm{shuf}}\), not pooled Brier.

Fresh seeds \(\{27101,27111\}\) train, \(27121\) val,
\(\{27131,27141\}\) held-out.

## Gates (GO)

Same five-gate logic as V7C.5, with \(Z=Z_{\mathrm{RGB}}\).

**H1.** Mid/final Brier vs \(y\):
\(\mathrm{Brier}^{S{+}Z}\le\mathrm{Brier}^{S}+\delta\).

**H2.** \(P(p_t>\tau_{\mathrm{dev}}\mid C0)\le 0.20\) for \(S{+}Z\),
\(\tau_{\mathrm{dev}}\) from development C0 finals of \(S{+}Z\).

**H3.** Held-out C1: mean Spearman\((p_t,w_t)>0\) and
\(\mathbb E[p\mid w_{\mathrm{high}}]>\mathbb E[p\mid w_{\mathrm{low}}]\).

**H4.** \(S{+}Z\) useful vs B5-S by `h4_sensor_useful`.

**H5 (hard temporal gate).**

\[
\boxed{
\Delta_{\mathrm{real}}-\Delta_{\mathrm{shuf}}
>\varepsilon_{\mathrm{temporal}}
}
\]

\[
\texttt{V7D\_GO}=H1\land H2\land H3\land H4\land H5.
\]

H4 without H5: possible **static scene/object identity**, not
time-locked visual evidence. GO false.

H5 without H4: time-locked \(e_t\) increment did not become V7C belief
metrics vs B5-S. GO false.

If \(\Delta_{\mathrm{real}}\le\varepsilon\): RGB is **conditionally
redundant (or harmful) given \(h^{S}\)** on this Door mixture — the
same scientific class as C.5 tactile. Do **not** search vision
encoders on this distribution.

## What GO does and does not unlock

**Unlocks (new prereg required):** a later visual-representation or
visuo-tactile / multimodal belief-integration stage, only if
`V7D_GO=true`.

**Does not unlock:** encoder architecture search, RGB resolution
sweeps, fusion layers, revision, \(C,V\), object slots, Door tactile
reopening, slip/friction tactile distribution (that remains option A,
separate prereg).

## Unlock of this stage

V7C.5 has a scientific (non-smoke) matrix. `V7C.5_GO=false` **is
allowed**. This stage is option B from the evidence-chain freeze, not
a reward for tactile success.
