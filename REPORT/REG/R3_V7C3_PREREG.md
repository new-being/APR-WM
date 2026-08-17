# R3-V7C.3 Preregistration — Epistemically Informative Tactile Observation

Date: 2026-08-16  
Status: **FROZEN** (observation sufficiency; no encoder, no B5 integration)  
Depends on: `REPORT/REP/R3_V7C2_REPORT.md` (locus FIELD)  
Does not change: V7A–V7C.2 results, B5-S, \(w_t\), GRU32  
Locks: **V7D**, CNN/encoder search, fusion studies, RGB, revision, \(C,V\)

## Placement

\[
\boxed{
V7C.2\ (\mathrm{locus}=\mathrm{FIELD})
\rightarrow
R3\text{-V7C.3 (this stage)}
\qquad
V7D\text{ locked}
}
\]

Stage name:

\[
\boxed{
R3\text{-V7C.3 — Epistemically Informative Tactile Observation}
}
\]

中文名：**面向认识信息的触觉观测设计**。

## Question

\[
\boxed{
\text{Can a tactile observation expose incremental warranted-evidence
information that is absent from }h^{-c}?
}
\]

Not “which encoder is stronger.” Not full B5 gates.

## Ablation (linear readout only)

Same GRU32, 12-D, \(e_t=y\,w_t\) BCE as V7C.2 P0/PX.
Each \(X^{(j)}\) is linearly mapped \(d_j\to 2\) into contact slots.

- \(X_N\): 8×8 force-magnitude field (V7C.2 failed baseline).
- \(X_{NS}\): \(X_N\) + 8×8 shear-magnitude field.
- \(X_{NSG}\): \(X_{NS}\) + patch geometry
  (centroid, mass/area proxy, second moments, eccentricity).
- \(X_{NSGM}\): \(X_{NSG}\) + multi-surface (finger1 / finger2 / palm)
  normal, shear, local moment \(z\).

\[
\Delta_j=L(P_0)-L(P_j),\qquad \varepsilon=0.005.
\]

A candidate is sufficient iff \(\Delta_j>\varepsilon\) on held-out seeds.

Matched C0/C1 \(L_2\) is **mechanism-only**, not a substitute for
\(\Delta_j\).

## Stop rule

If \(\max_j\Delta_j\le\varepsilon\), freeze:

\[
\text{tactile is largely conditionally redundant for warranted
structural belief on this Door mixture given }h^{-c}.
\]

Do not open encoder or V7D. Expanding the task distribution
(slip / friction / distributed contact) would need a new prereg.

If some \(\Delta_j>\varepsilon\), this stage **does not** train a CNN
or re-run B5. Next would be representation sufficiency on that
observation only, under a new prereg.

## What this does not authorize

Bigger taxel grids as the only change, encoder architecture search,
fusion, RGB, V7D.
