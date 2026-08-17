# R3-V7C.1 Report — Learned Tactile Epistemic Representation

Date: 2026-08-16  
Prereg: `REPORT/REG/R3/R3_V7C1_PREREG.md` (unchanged; SHA at run time)  
Depends on: `REPORT/REP/R3/R3_V7C_REPORT.md` (`V7C_GO=true`)  
Artifacts: `runs/r3_v7c1/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `d091583da998fa77f692ce26a8b6b37e498da7f79b7640ddb56730e8f94167a1`

## Decision

\[
\boxed{V7C.1\_GO=\mathrm{false}}
\]

\[
\boxed{
\text{learned tactile }z_t\text{ did not beat a no-contact ablation as an
epistemic channel, so tactile representation is the bottleneck}
}
\]

This does **not** reopen V7B.3 / V7C, GRU gates, \(w_t\), or B5-S.
**V7D stays locked.** Failure is not a reason to doubt warranted
supervision on proprioceptive contact.

Smoke (`runs/r3_v7c1/smoke/`): taxel maps nonzero on contact, zero on
Mode-A; GO not evaluated.

## Question (unchanged)

Does a learned tactile representation preserve evidence-warranted
epistemic semantics vs B5-S, and is it a nonempty channel vs B5-N?

Good tactile reconstruction \(\neq\) good epistemic representation.
Reconstruction is diagnostic only.

## Design (frozen)

- Seeds \(\{22101,22111\}\) train, \(22121\) val, \(\{22131,22141\}\) held-out.
- Same B5 GRU32 and \(y\cdot w_t\). Only contact slots (cols 5–6) come
  from \(z_t=f_\phi(x_t^{\mathrm{tactile}})\), an 8×8 gripper-local
  pressure map (not hinge \(J^\top f\), not pair-list input).
- **S:** V7C B5-S recipe retrained on this split.
- **N:** contact slots zeroed.
- \(\tau_{\mathrm{dev}}=0.528\) from tactile-arm development C0 \(p_T\).

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 tactile Brier \(\le\) S \(+\delta\) | **PASS** | mid \(0.160\le0.164+0.02\); final \(0.158\le0.156+0.02\) |
| H2 C0 FPR \(\le0.20\) | **PASS** | FPR \(0.122\) |
| H3 Spearman\((p,w)>0\) and high \(w\) \(>\) low \(w\) | **PASS** | \(\rho=0.131\); \(0.674>0.366\) |
| H4 nonempty vs N | **FAIL** | pooled Brier \(N<T\); \(B_{\mathrm{C0}}^{T}=0.202>0.191=B_{\mathrm{C0}}^{N}\) |

\[
\boxed{V7C.1\_GO=H1\land H2\land H3\land H4=\mathrm{false}}
\]

## What the numbers say

H1–H3 say the warranted GRU **can still run** when contact slots are
filled by a learned \(z_t\): calibration vs S is inside \(\delta\),
C0 FPR stays under 0.20, and \(p_t\) still co-moves with \(w_t\).

H4 is the scientific miss. Tactile is **not** a useful epistemic
channel relative to dropping contact entirely: occupancy is worse
than N, and pooled mid/final Brier is also worse. C1-final is
slightly better (\(0.145<0.149\)), which is not enough for the
predeclared occupancy+C1 clause.

Diagnostic (not GO): Spearman of taxel-mean vs oracle contact flag is
\(\approx 0\) (\(-0.013\)). The 8×8 field is a weak carrier of the
contact evidence this mixture actually uses. Linear-probe MSE of
\(z\to\) taxel-mean is \(0.0057\) — the encoder can track its own
field without that field being the right epistemic observation.

\[
\boxed{
\text{good tactile representation}
\neq
\text{good epistemic representation}
}
\]

Prefer the attribution:

\[
\boxed{\text{tactile representation bottleneck}}
\]

not a failure of evidence-warranted supervision (already passed on
B5-S).

## What this does not authorize

- revising V7B.3 / V7C GOs;
- GRU / \(w_t\) / B5-S proxy search;
- opening V7D / RGB;
- treating reconstruction MSE as a pass.

If continuing: a **tactile-representation diagnostic** (field vs
timing vs fusion), not another Door-oracle GRU tweak. New prereg
required. Not automatic.
