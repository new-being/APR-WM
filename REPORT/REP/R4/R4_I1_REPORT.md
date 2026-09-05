# R4-I1 Report — No-Leak Audit of Full \(h^{S}\)

Date: 2026-08-17  
Status: **PASS** (`R4_I1_PASS=true`)  
Prereg: `REPORT/REG/R4/R4_I1_PREREG.md`  
Artifacts: `runs/r4_i1/formal/summary.json`  
Depends on: frozen I0 v2 (`REPORT/REP/R4/R4_I0_V2_REPORT.md`)  
Does not change: I0 \(\mu\) / `solref` / grid / squeeze / \(F_t(t)\),
R4-C0 question  
Does not set: R4-C0 GO (C0 later ran `GO=false`)

## Question

When every learner-visible macro channel is included, can \(h^{S}\)
still identify stiffness-swap A vs B? Does adding the taxel field \(X\)
create a large gain?

\[
h^{S}_t=\bigl\{
q,\dot q,\ddot q,\;
q_{\mathrm{finger}},\dot q_{\mathrm{finger}},\ddot q_{\mathrm{finger}},\;
u_{\mathrm{finger}},\tau_{\mathrm{motor}},F_t^{\mathrm{cmd}},\;
F_n,F_t,F_{x,y,z},M_{x,y,z}
\bigr\}_{t-W+1:t},\quad W=8.
\]

No per-finger split forces. Linear probe only. Split by trajectory
\(q_0\), not by random frames.

## Result

| probe (held-out \(q_0\)) | AUROC |
|---|---|
| \(h^{S}\) | **0.556** |
| \(X\) | 0.993 |
| \(h^{S}+X\) | **0.996** |
| gain \(\mathrm{AUROC}(h^{S},X)-\mathrm{AUROC}(h^{S})\) | **0.440** |
| \(h^{S}+X\) canonicalize B onto A | **0.504** |
| bilateral \(x\)-flip of both classes | 0.993 (equivariant; not a drop) |

H1: \(\mathrm{AUROC}(h^{S})=0.556<0.90\), not \(\ge 0.95\).  
H2: gain \(0.440>0.10\).

\[
\boxed{\texttt{R4\_I1\_PASS}=\text{true}}
\]

Train: 5 trajectories, 2375 macro-matched pairs (4750 probe rows), 1861
local-split. Held: 2 trajectories, 950 pairs (1900 rows), 745 local-split.

## Interpretation

I0 v2's physical claim survives a **strong** proprioceptive baseline:

\[
D(h^{S,A},h^{S,B})\approx 0
\quad\text{to a linear learner on held-out ICs,}
\qquad
\mathrm{Perf}(h^{S},X)-\mathrm{Perf}(h^{S})\gg 0.
\]

The label is not a trajectory fingerprint: the probe is fit on
\(\{q_0\}\) train and scored on unseen \(q_0\).

Canonicalize (not bilateral flip) shows the information is **spatial
allocation**. Flipping both A and B is a relabeling the linear probe
can relearn; aligning B to A's \(x\) frame drops AUROC to chance.

v1 no-go is unchanged: hiding a scalar \(\mu\) leaked into \(F_n\).
v2+I1 pass by matching global integrals and changing spatial allocation.

## What this does not claim

- not \(I(X;e_t\mid h^{S})>0\) (that is R4-C0)
- not that a GRU/encoder is needed
- not a license to retune I0

## Next

R4-C0 is done (`REPORT/REP/R4/R4_C0_REPORT.md`): `GO=false`.
I0/I1 stay frozen. Do not train an encoder.
