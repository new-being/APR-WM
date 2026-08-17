# R3-V7C.2 Preregistration — Tactile Epistemic Information Decomposition

Date: 2026-08-16  
Status: **FROZEN** (diagnostic classification; no method GO)  
Depends on: `REPORT/REP/R3_V7C1_REPORT.md` (scientific matrix; GO may stay false)  
Does not change: V7A–V7C.1 GOs, RS* GOs, C0, \(s^\star\), B5-S proxy, \(w_t\)  
Locks: **V7D**, RGB, encoder-architecture search, GRU-gate search, revision,
\(C,V\)

## Placement

\[
\boxed{
V7C.1\text{ (8×8 tactile channel adds no net epistemic value)}
\rightarrow
R3\text{-V7C.2 (this diagnostic)}
\qquad
V7D\text{ locked}
}
\]

Stage name:

\[
\boxed{
R3\text{-V7C.2 — Tactile Epistemic Information Decomposition}
}
\]

中文名：**触觉认识信息分解诊断**。

This stage **classifies a failure locus**. It does not authorize a
stronger encoder, a new sensor, or V7D.

## Question

\[
\boxed{
\text{Where is the epistemically useful contact information lost?}
}
\]

along

\[
x_t^{\mathrm{tactile}}\rightarrow z_t^{\mathrm{tactile}}\rightarrow u_t^{\mathrm{epi}}\rightarrow p_t.
\]

Primary diagnostic target (offline, not a runtime input):

\[
e_t=y\,w_t.
\]

Oracle contact flags are **not** the primary target.

## Incremental probes (same GRU32, 12-D, \(e_t\) BCE)

\(h_{1:t}^{-c}\): V7B steps with contact slots zeroed.

**P0.** \(h^{-c}\rightarrow e_t\) (B5-N recipe).  
**PX.** \(h^{-c}\) plus a **linear readout of flattened raw taxel**
(\(64\to 2\), tanh) into the contact slots, trained for \(e_t\).  
**PZ.** V7C.1 CNN encoder \(\to 2\) fused the same way, trained for \(e_t\).

Held-out loss \(L=\) mean BCE vs \(e_t\).

\[
\Delta_X=L(P_0)-L(P_X),\qquad
\Delta_Z=L(P_0)-L(P_Z).
\]

Frozen: \(\varepsilon=0.005\). \(\Delta\le\varepsilon\) counts as
“no incremental information.”

## Auxiliary (not the locus rule)

- Matched C0/C1 taxel \(L_2\) early (\(w<0.25\)) vs late (\(w>0.75\)).
- Time-shuffle taxel on PZ at eval; \(\Delta_{\mathrm{sh}}=L(\mathrm{shuffle})-L(P_Z)\).
- H4-style T vs N on this split (occupancy / C1 Brier), diagnostic only.

## Locus (output, not a GO)

\[
\begin{aligned}
\Delta_X\le\varepsilon &\Rightarrow \textbf{FIELD}\\
\Delta_X>\varepsilon,\ \Delta_Z\le\varepsilon &\Rightarrow \textbf{ENCODER}\\
\Delta_Z>\varepsilon\text{ and T not useful vs N} &\Rightarrow \textbf{FUSION}\\
\text{otherwise} &\Rightarrow \textbf{UNRESOLVED}
\end{aligned}
\]

Smoke must not set the locus. Success of this stage is a **valid
classification**, not FIELD vs ENCODER vs FUSION themselves.

## What this does not authorize

Redesigning the taxel geometry, swapping the CNN, fusion studies, or
V7D, until the locus is read and a **new** prereg is written for that
branch only.
