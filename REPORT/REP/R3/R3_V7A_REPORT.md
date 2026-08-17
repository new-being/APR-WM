# R3-V7A Report — Mixture-Trained Contextual Epistemic Belief

Date: 2026-08-16  
Prereg: `REPORT/REG/R3/R3_V7A_PREREG.md`  
Depends on: `REPORT/REP/R1/R1_TRANSPORT_STAGE_FREEZE.md`  
Artifacts: `runs/r3_v7a/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `1df494b4eef33a150e2293def69e4065830bf4dcf537b86c47912d6966173bd4`

## Decision

\[
\boxed{V7A\_GO=\mathrm{true}}
\]

\[
\boxed{
\text{a single contextual }p_{\mathrm{struct}}(\tilde h)
\text{ trained on the embodied mixture is calibrated
on held-out seeds of the same mixture}
}
\]

This does **not** set `RS4A_GO`, does **not** rescue zero-shot transport,
does **not** open RS5B / revision / VoI / H32, and does **not** open
R3-V7B (sequential \(b_t\)).

Smoke (`runs/r3_v7a/smoke/`) is plumbing only.

## Question (unchanged)

Within-domain generalization of contextual structural-inadequacy belief,
not Mode-A \(\rightarrow\) contact transport.

## Design (frozen)

- Development seeds \(\{16101,16111,16121\}\); held-out \(\{16131,16141\}\).
- Union \(\mathcal D_{\mathrm{free}}\cup\mathcal D_{\mathrm{pull}}\cup\mathcal D_{\mathrm{push}}\).
- Predictor: ridge logistic on RS4A geometry \(\tilde h\) (**no** family label).
- Baseline: mixture-trained \(D_0\)-only logistic.
- LOIO excluding push: diagnostic only.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 contextual Brier \(<\) \(D_0\)-only on held-out mixture | **PASS** | \(0.164<0.209\) |
| H2 held-out C0 FPR \(\le0.20\) | **PASS** | \(0.083\) at mix-dev C0 95th \(\tau\approx0.716\) |
| H3 within-family AUROC \(\ge0.60\) | **PASS** | Mode-A \(0.75\); pull \(1.00\); push \(0.75\) |

\[
\boxed{V7A\_GO=H1\land H2\land H3=\mathrm{true}}
\]

## Diagnostic (not GO)

On held-out `contact_push`, mixture-trained Brier \(0.167\) vs LOIO
(train without push) Brier \(0.199\); AUROC both \(0.75\).

Union coverage improves **calibration** of the same discrimination.
That is the intended contrast with RS4A: geometry as context inside a
shared generative process, not as a transported probability from two
families onto a third.

## What this does and does not show

Shows: if Mode-A / pull / push are treated as one embodied mixture,
a single contextual \(\hat p_{\mathrm{struct}}\) can be held-out
calibrated without per-\(\mathcal I\) routers or a universal scalar
threshold.

Does **not** show:

- a persistent recurrent epistemic state \(u_t^{\mathrm{epi}}\);
- new objects, masses, cameras, or tasks (still Door oracle);
- RGB / tactile;
- true OOD abstention (cloth, fluids, unseen mechanisms);
- that RS4A LOIO would now pass (it remains false).

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
RS1C\_GO &= true\\
RS2\_GO &= false\\
RS3A/A.1/RS4A\_GO &= false\\
RS5A\_GO &= true \quad\text{(side architecture)}\\
V7A\_GO &= true\\
V7B\_GO &= false\\
RS5B/C,\ R3\text{-V7C/D} &= locked\\
\text{transport mainline} &= closed
\end{aligned}
}
\]

Next, if continuing the mainline: **R3-V7B** was run (`V7B_GO=false`).
See `REPORT/REP/R3/R3_V7B_REPORT.md`. Not RS5B.
