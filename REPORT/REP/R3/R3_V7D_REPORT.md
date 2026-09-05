# R3-V7D Report — Conditional Modality Value of RGB

Date: 2026-08-16  
Prereg: `REPORT/REG/R3/R3_V7D_PREREG.md`  
Depends on: `REPORT/REP/R3/R3_V7C5_REPORT.md` (scientific matrix; `V7C.5_GO=false` is **not** a lock)  
Artifacts: `runs/r3_v7d/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `e42a6fa227439f9c87aa313648852903930f71c879d1ab5a5145b04391fc7562`

## Decision

\[
\boxed{V7D\_GO=\mathrm{false}}
\]

\[
\boxed{\text{conditional usefulness of }Z_{\mathrm{RGB}}\text{ given }h^{S}:\ \mathrm{FAIL}}
\]

Door tactile C.5 remains a negative baseline. This stage does **not**
search a larger vision encoder, does **not** reopen Door tactile
CNN/fusion, and does **not** open visuo-tactile fusion.

Smoke (`runs/r3_v7d/smoke/`) confirmed RGB plumbing only; GO was not
evaluated there.

## Question (unchanged)

\[
I(Z_{\mathrm{RGB}};e_t\mid h^{S})>0\ ?
\]

Given frozen B5-S history, does agentview RGB still supply extra
evidence relevant to warranted structural belief?

## Design (frozen)

- Seeds \(\{27101,27111\}\) train, \(27121\) val, \(\{27131,27141\}\)
  held-out (\(n=42/21/42\)).
- RGB: offscreen `agentview`, \(32\times32\times3\), \(20\,\mathrm{Hz}\).
- Encoder frozen:
  `Conv(3→8)→Tanh→Conv(8→8)→Tanh→AdaptiveAvgPool(2)→Flatten→Linear(32→2)→Tanh`.
- Three arms: **B5-S**, **B5-S + real \(Z_{\mathrm{RGB}}\)**,
  **B5-S + temporally shuffled \(Z_{\mathrm{RGB}}\)**.
- \(L\) = held-out BCE vs \(e_t=y\,w_t\).
- \(\varepsilon_{\mathrm{temporal}}=0.005\), \(\delta=0.02\),
  C0 FPR \(\le 0.20\).
- Manual SGD (no Adam). Headless GL: `MUJOCO_GL=egl`.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 \(S{+}Z\) Brier \(\le S+\delta\) | **FAIL** | mid \(0.192\not\le 0.165+0.02\); final \(0.195\not\le 0.158+0.02\) |
| H2 C0 step-FPR \(\le 0.20\) | **PASS** | FPR \(0.009\) |
| H3 Spearman\((p,w)>0\) and high \(w>\) low \(w\) | **PASS** | \(\rho=0.196\); \(0.632>0.310\) |
| H4 \(S{+}Z\) useful vs B5-S | **FAIL** | \(B_{\mathrm{C0}}^{SZ}=0.217>0.189\); C1-final \(0.167>0.143\) |
| H5 \(\Delta_{\mathrm{real}}-\Delta_{\mathrm{shuf}}>\varepsilon_{\mathrm{temporal}}\) | **FAIL** | \(\Delta_{\mathrm{real}}=-0.048\); \(\Delta_{\mathrm{shuf}}=-0.029\); gap \(-0.019\not>0.005\) |

\[
\boxed{V7D\_GO=H1\land H2\land H3\land H4\land H5=\mathrm{false}}
\]

## How to read the failure

Adding frozen \(Z_{\mathrm{RGB}}\) **hurt** the warranted-evidence
objective relative to B5-S alone:

\[
L(S)=0.441,\qquad L(S{+}Z_{\mathrm{RGB}})=0.488,\qquad L(S{+}Z_{\mathrm{shuf}})=0.470.
\]

Real RGB is **worse** than time-shuffled RGB. That is the opposite of
a time-locked visual increment: the aligned camera stream is not
paying for itself given proprioceptive contact history, and shuffling
does not make it worse.

H2 occupancy stays legal. H3 tracks \(w_t\) at a modest Spearman.
Those do not rescue H1/H4/H5.

This is the same scientific class as V7C.5 tactile:

\[
\boxed{
I(Z_{\mathrm{RGB}};e_t\mid h^{S})\le 0
\quad\text{on this Door mixture (frozen encoder)}
}
\]

Do **not** interpret the failure as “the CNN was too small.” The
prereg forbade encoder search. A larger vision net on the same
mixture would not be a test of conditional modality value; it would
be capacity shopping.

## What this does not authorize

- RGB resolution / architecture / fusion search
- visuo-tactile fusion
- reopening Door tactile CNN / V7C.6
- occupancy penalties or GRU-gate edits
- treating C.4 SUFFICIENT or C.5 GO=false as incomplete

## What follows

\[
\boxed{
\begin{aligned}
\text{Door tactile conditional usefulness} &\quad\text{FAIL (C.5)}\\
\text{Door RGB conditional usefulness} &\quad\text{FAIL (V7D)}\\
\text{visuo-tactile fusion} &\quad\text{not opened}
\end{aligned}
}
\]

A later tactile stage requires a **new distribution** (R4), not Door
capacity. See `REPORT/REG/R4/R4_C0_PREREG.md`. There is **no V7E**.
Belief update remains evidence accumulation plus **conditional
information allocation**, not \(p_t=f(\text{all modalities})\).
