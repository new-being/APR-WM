# R3-V7C.4 Report — Representation Sufficiency of \(X_{NSG}\)

Date: 2026-08-16  
Prereg: `REPORT/REG/R3_V7C4_PREREG.md`  
Depends on: `REPORT/REP/R3_V7C3_REPORT.md`  
Artifacts: `runs/r3_v7c4/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `f96834eb9efbf2dcd9d2e613e98e63e4180f551e5de6c4d63755be49cb5e1dd0`

## Decision (representation, not B5 / not V7D)

\[
\boxed{\texttt{verdict}=\textbf{SUFFICIENT}}
\]

by the **frozen** trichotomy on **same-run** \(\Delta_X,\Delta_Z\).
This stage did **not** train B5 and does **not** open V7D.

\[
\boxed{
\Delta_Z=0.014\approx\Delta_X=0.015
}
\]

V7C.2 FIELD for the old 8×8 magnitude map remains frozen.
C.3 \(\Delta_{NSG}=0.057\) is a reference, not the comparator.

Smoke did not classify a verdict.

## Question (unchanged)

Can a learned \(Z_{NSG}\) preserve the incremental \(e_t\) information
in frozen \(X_{NSG}\), given \(h^{-c}\)?

## Design (frozen)

- Seeds \(\{25101,25111\}\) train, \(25121\) val, \(\{25131,25141\}\)
  held-out (C.3 held-out not reused).
- P0: \(h^{-c}\). PX: linear \(X_{NSG}\to 2\). PZ: 2-channel CNN +
  geometry MLP \(\to z\in\mathbb{R}^{2}\) into the same contact slots.
- \(\varepsilon=0.005\), \(\varepsilon_{\mathrm{retain}}=0.02\).
- Time-shuffle of \(X_{NSG}\) through PZ is **diagnostic only**.

## Results

| Quantity | Value |
|---|---|
| \(L(P_0)\) | \(0.471\) |
| \(L(P_X)\) | \(0.455\) |
| \(L(P_Z)\) | \(0.457\) |
| \(L(P_Z^{\mathrm{shuffle}})\) | \(0.457\) |
| \(\Delta_X\) | \(0.015>\varepsilon\) |
| \(\Delta_Z\) | \(0.014>\varepsilon\) |
| \(\Delta_Z^{\mathrm{shuffle}}\) | \(0.014\) |
| C.3 \(\Delta_{NSG}\) reference | \(0.057\) |
| frozen verdict | **SUFFICIENT** |

Observation **replicated** on this split (\(\Delta_X>\varepsilon\)), so
the encoder trichotomy is defined. \(\Delta_Z\ge\Delta_X-0.02\) holds
(\(0.014\ge -0.005\)).

## How to read SUFFICIENT here

Two facts must be kept together.

**1. Same-run preservation.** \(\Delta_Z\) matches \(\Delta_X\) to
about \(0.001\). The encoder did not destroy the *same-run* linear
increment.

**2. The increment itself is small and not time-locked.**
\(\Delta_X=0.015\) is far below the C.3 reference \(0.057\)
(seed instability continues). Because \(\Delta_X<\varepsilon_{\mathrm{retain}}\),
the predeclared **LOSES** band
\((\varepsilon,\Delta_X-\varepsilon_{\mathrm{retain}})\) is **empty**
on this split. SUFFICIENT therefore mostly means “not DESTROYS.”

Shuffle:

\[
\Delta_Z^{\mathrm{shuffle}}\approx\Delta_Z
\]

so PZ is **not** using episode-specific, temporally aligned \(X_{NSG}\)
maps—the same diagnostic pattern as V7C.2’s CNN path. That was not a
trichotomy input, and is **not** rewritten into a fail after seeing
the data. It **is** a reason not to treat this verdict as “the
representation found contact geometry.”

\[
\boxed{
\text{SUFFICIENT}\neq\text{the CNN is reading patch geometry over time}
}
\]

## What this does not authorize

- stuffing \(Z_{NSG}\) into frozen B5 in this stage
- V7D, RGB, encoder search
- rewriting V7C.2 FIELD
- claiming shear is useful
- treating C.3 \(0.057\) as replicated on \(251xx\)

## What follows

Prereg allows a **later** belief-integration prereg only because
the frozen verdict is SUFFICIENT. That later prereg, if written,
should gate on temporal use (shuffle \(\Delta\) drop), not only on
\(\Delta_Z\approx\Delta_X\). This stage still does not train
\(Z\to p_t\).

\[
\boxed{
\begin{aligned}
V7C.2 &: \text{old 8×8 FIELD retained}\\
V7C.3 &: X_{NSG}\text{ incremental on }241xx\\
V7C.4 &: \texttt{SUFFICIENT}\text{ on same-run }\Delta;\ \text{shuffle-null}\\
V7D &: \text{locked}
\end{aligned}
}
\]
