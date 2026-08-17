# R3-V7C.4 Preregistration — Representation Sufficiency of \(X_{NSG}\)

Date: 2026-08-16  
Status: **FROZEN** (representation sufficiency; no B5 integration)  
Depends on: `REPORT/REP/R3/R3_V7C3_REPORT.md` (`stop_tactile_branch=false`, \(X_{NSG}\) max \(\Delta\))  
Does not change: V7A–V7C.3 results, V7C.2 FIELD locus, B5-S, \(w_t\), GRU32  
Locks: **V7D**, encoder architecture search, fusion, RGB, revision, \(C,V\)

## Placement

\[
\boxed{
V7C.3\ (X_{NSG}\ \Delta=0.057)
\rightarrow
R3\text{-V7C.4 (this stage)}
\qquad
V7D\text{ locked}
}
\]

Stage name:

\[
\boxed{
R3\text{-V7C.4 — Representation Sufficiency}
}
\]

中文名：**触觉表示是否保留 \(X_{NSG}\) 的条件增量认识信息**。

## Frozen from prior stages

- V7C.2 FIELD for the old 8×8 magnitude map is **retained**. Seed
  instability of \(\Delta_N\) does not rewrite that locus.
- V7C.3 showed \(\exists X: I(e_t;X\mid h^{-c})>0\) on seeds \(241xx\),
  with nested geometry—not shear—as the useful increment.
- This stage does **not** reopen observation search (\(N/NS/NSGM\)).
  The observation is frozen as \(X_{NSG}\).

## Question

\[
\boxed{
\text{Can a learned tactile representation preserve the incremental
epistemic information present in }X_{NSG}?
}
\]

Causal chain under test (only the middle arrow):

\[
X_{NSG}\rightarrow Z_{NSG}\rightarrow e_t
\]

Not \(Z\to p_t\). Not B5. Not V7D.

## Design

- Fresh seeds \(\{25101,25111\}\) train, \(25121\) val,
  \(\{25131,25141\}\) held-out. C.3 held-out is not reused.
- Same GRU32, 12-D \(h^{-c}\), target \(e_t=y\,w_t\), BCE, manual SGD.
- **P0:** \(h^{-c}\to e_t\).
- **PX:** linear readout of frozen \(X_{NSG}\) (normal + shear 8×8 +
  7-D patch geometry) into the two contact slots. Same class as V7C.3
  \(P_{NSG}\). This **re-estimates** \(\Delta_X\) on the new split; it
  is a replication check, not a new observation question.
- **PZ:** one frozen encoder, not an architecture search:
  2-channel CNN on stacked normal/shear maps, MLP on geometry,
  concatenate, map to \(z\in\mathbb{R}^{2}\) (tanh), write the contact
  slots, then the same GRU.
- \(\varepsilon=0.005\) (same information threshold as V7C.2/C.3).
- \(\varepsilon_{\mathrm{retain}}=0.02\): absolute slack for
  “preserves,” larger than C.3 nested jitter around \(\varepsilon\)
  and smaller than C.3 \(\Delta_{NSG}=0.057\).

\[
\Delta_X=L(P_0)-L(P_X),\qquad
\Delta_Z=L(P_0)-L(P_Z).
\]

The primary comparator is **same-run** \(\Delta_X\), not the frozen
C.3 number \(0.057\). That number is reported as a reference only.
Reason: V7C.2/C.3 already showed that the linear-\(\Delta\) estimator
is seed-noisy.

Time-shuffling \(X_{NSG}\) through PZ is **diagnostic only**.

## Predeclared trichotomy

If \(\Delta_X\le\varepsilon\):

\[
\boxed{\texttt{OBSERVATION\_NOT\_REPLICATED}}
\]

Do **not** classify the encoder. Do **not** open B5. The observation
claim failed to replicate on the new split; that is an estimator /
distribution issue, not a representation verdict.

Otherwise \(\Delta_X>\varepsilon\):

\[
\begin{aligned}
\Delta_Z\le\varepsilon
&\Rightarrow
\texttt{DESTROYS}\\
\varepsilon<\Delta_Z<\Delta_X-\varepsilon_{\mathrm{retain}}
&\Rightarrow
\texttt{LOSES}\\
\Delta_Z\ge\Delta_X-\varepsilon_{\mathrm{retain}}
&\Rightarrow
\texttt{SUFFICIENT}
\end{aligned}
\]

\(\Delta_Z>\Delta_X\) still counts as **SUFFICIENT**.

Unlocking a later belief-integration prereg (\(Z_{NSG}\to p_t\) on
frozen B5) requires **SUFFICIENT**. LOSES / DESTROYS / NOT_REPLICATED
do **not** open B5, encoder search, or V7D.

## What this does not authorize

B5 training, occupancy / Brier gates, CNN search, fusion, RGB, V7D,
reopening \(X_N\) or shear-only stories, rewriting V7C.2 FIELD.
