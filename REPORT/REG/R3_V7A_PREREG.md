# R3-V7A Preregistration — Mixture-Trained Contextual Epistemic Belief

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R1_TRANSPORT_STAGE_FREEZE.md`,
`REPORT/REP/R1_RS5A_REPORT.md`  
Does not change: any frozen RS* GO, C0, \(s^\star\), RS1C policy  
Locks: **RS5B/C**, **RS4B/C**, **RS3B/C**, revision, VoI, H32  
Does **not** add RGB or learned tactile this stage

## Placement

The RS2–RS4 transport mainline is closed. RS5A is a side architecture
result (indexed calibration + abstention), not the next experiment.

\[
\boxed{
\text{transport branch frozen}
\rightarrow
R3\text{-V7A (this stage)}
\qquad
RS5B\text{ locked}
}
\]

This is **not** the old 2D V7 routing experiment in `aprwm_v0/v07.py`.

Stage name:

\[
\boxed{
R3\text{-V7A — Mixture-trained contextual epistemic belief}
}
\]

中文名：**混合具身域上的情境化认识信念（第一步）**。

## Question

\[
\boxed{
\text{Can one contextual model trained on the union of embodied
interaction families produce a calibrated structural-inadequacy
belief on held-out seeds from the same mixture?}
}
\]

Not: Mode-A \(\rightarrow\) contact zero-shot transport.  
Not: a globally invariant scalar \(E\in\mathbb R\).  
Not: LOIO as a GO (that was RS4A; it failed).

Contrast with RS4A:

| | RS4A | R3-V7A |
|---|---|---|
| Train | two families | **union** of all families |
| Test | held-out **family** (domain shift) | held-out **seeds** (within mixture) |
| Target | transport \(p\) | within-domain contextual \(p_{\mathrm{struct}}\) |

## What is being learned (honest scope)

Full target later:

\[
z_t^{\mathrm{epi}}=f_\psi(h_t),\qquad
b_t=(s_t,\theta_t,M_t,u_t^{\mathrm{epi}}).
\]

This stage is **Step 0**: a frozen episode summary of \(h_t\)

\[
\tilde h =
(\log D_0,\; z_{\mathrm{geom}})
\]

the same learner-visible geometry as RS4A (`FEATURE_NAMES`), **without**
family / script / domain-ID labels. A single ridge logistic

\[
\hat p_{\mathrm{struct}} = \sigma(w^\top \tilde h)
\]

is trained on the development **mixture**

\[
\mathcal D =
\mathcal D_{\mathrm{free}}\cup\mathcal D_{\mathrm{pull}}\cup\mathcal D_{\mathrm{push}}.
\]

This is not yet a recurrent epistemic state. Sequential \(b_{t+1}=F(b_t,a_t,x_{t+1})\)
is **R3-V7B**, locked until this stage reports.

Sensor staging (not this run):

1. **this stage**: oracle state + proprioception + oracle contact geometry;
2. later: tactile / contact sensors;
3. later: RGB / object slots.

## Design

- Seeds \(\{16101,16111,16121\}\) development, \(\{16131,16141\}\) held-out.
- Same 90-episode job grid as RS4A/RS5A: Mode-A amps \(\{1.0,1.5\}\);
  pull `(fast_pull,1.0)+(pull_release,2.0)`; push `(pull_push,1.0)+(pull_push,1.5)`;
  regimes C0 / C1-L / C1-H.
- No family string in the predictor.
- \(D_0\)-only logistic on the same mixture is the baseline.
- Leave-one-family-out on development, scored on held-out push, is
  **diagnostic only** (documents transport vs mixture; not a GO).

## Hypotheses (GO)

**H1.** On held-out mixture, Brier of contextual \(\hat p_{\mathrm{struct}}\)
is strictly less than Brier of mixture-trained \(D_0\)-only.

**H2.** Held-out C0 FPR at the 95th percentile of **mixture-development C0**
\(\hat p_{\mathrm{struct}}\) is \(\le 0.20\).

**H3.** On held-out data, AUROC of \(\hat p_{\mathrm{struct}}\) is
\(\ge 0.60\) **inside each family** (Mode-A, contact-pull, contact-push).

\[
\boxed{V7A\_GO = H1 \land H2 \land H3}
\]

Smoke (`--smoke`) must not set `V7A_GO`.

## Diagnostic (not GO)

Train on development mixture **excluding** `contact_push`; evaluate
held-out `contact_push`. Report Brier / AUROC vs the mixture-trained
model on the same test rows. Expected: mixture-in-support is better
than LOIO. Failure of this diagnostic does not flip H1–H3; it only
clarifies whether union coverage, not geometry, is doing the work.

## What success does **not** authorize

- claiming invariant scalars are solved;
- claiming LOIO transport is solved;
- opening RS5B / revision / VoI / H32;
- dumping RGB + tactile in one step;
- treating this logistic as a persistent belief \(b_t\).

## What failure would mean

Mixture training of a static \(\tilde h\) is not enough for even
within-domain calibrated \(p_{\mathrm{struct}}\). Then diagnose
(capacity vs features vs label noise) before R3-V7B. Do **not**
quietly return to hunting \(\tau\) on contact.

## True OOD (out of scope here)

Cloth, fluids, unseen mechanisms remain
\(\mathcal D_{\mathrm{deploy}}\not\subset\mathrm{support}(\mathcal D_{\mathrm{train}})\).
Abstention / \(p_{\mathrm{unknown}}\) is a later head, not this GO.
