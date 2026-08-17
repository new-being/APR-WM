# R3-V7C Report — Sensorized Contact Epistemic Belief

Date: 2026-08-16  
Prereg: `REPORT/REG/R3_V7C_PREREG.md`  
Depends on: `REPORT/REP/R3_V7B3_REPORT.md` (`V7B.3_GO=true`)  
Artifacts: `runs/r3_v7c/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `59dff6bda1bbc3ae3257ac40477a88e3bd79e8c87c13ca5f1577c20b6e78f707`

## Decision

\[
\boxed{V7C\_GO=\mathrm{true}}
\]

\[
\boxed{
\text{evidence-warranted temporal belief survives imperfect contact observation}
}
\]

Same B5 GRU and \(y\cdot w_t\) as V7B.3. Runtime **B5-S** does not see
contact pairs, hinge \(J^\top f\), \(w_t\), or phase. This does **not**
open V7C.1, V7D, RGB, or tactile images.

The pass is the conjunction, not pooled Brier vs N:

\[
\boxed{
\begin{aligned}
&\text{oracle contact}\rightarrow\text{noisy proprioceptive sensing}\\
&\text{C0 occupancy does not explode (FPR }0.013\text{)}\\
&\text{evidence timing retained}\\
&\text{sensor channel useful vs no-contact, metric-dependently}
\end{aligned}
}
\]

Smoke (`runs/r3_v7c/smoke/`): 3 matched triples; contact
`sensor_proxy_mean` finite and nonzero.

## Question (unchanged)

Does warranted persistent belief remain calibrated when inference-time
oracle contact is replaced by learner-visible noisy contact sensing?

Primary comparison: **B5-S vs B5-N**, not beating B5-O.

## Design (frozen)

- Seeds \(\{21101,21111\}\) train, \(21121\) val, \(\{21131,21141\}\) held-out.
- Matched C0 record / C1 replay; \(N=105\); B5 targets \(y\cdot w_t\).
- 12-D V7B steps; only columns 5–6 change.
- **O:** `contact_count` binary / window fraction.
- **S:** wrist wrench observer + arm torque residual + \(\log(1+\|\hat f\|)\),
  \(\sigma_f=0.5\), \(\sigma_\tau=0.05\), \(\sigma_r=0.1\).
- **N:** contact columns zeroed.
- \(\tau_{\mathrm{dev}}=0.599\) from B5-S development C0 \(p_T\) 95th.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 S Brier \(\le\) O \(+\delta\) | **PASS** | mid \(0.158\le0.170+0.02\); final \(0.161\le0.169+0.02\) |
| H2 C0 step-FPR \(\le0.20\) | **PASS** | FPR \(0.013\); \(B_{\mathrm{C0}}^{S}=0.166\) |
| H3 Spearman\((p,w)>0\) and high \(w\) \(>\) low \(w\) | **PASS** | \(\rho=0.244\); \(0.678>0.298\) |
| H4 S useful vs N | **PASS** | not via pooled Brier; via \(B_{\mathrm{C0}}\) \(0.166<0.186\) and C1-final \(0.151<0.157\) |

\[
\boxed{V7C\_GO=H1\land H2\land H3\land H4=\mathrm{true}}
\]

## What the numbers say

B5-S is **inside** the oracle non-inferiority band and, on this seed
split, slightly **better** than O on mid/final Brier vs \(y\). C0
step-FPR stays low (\(0.013\)). Evidence timing is intact: \(p_t\)
still co-moves with train-only \(w_t\).

H4 is the honest one. Do **not** write “sensorized contact always
improves prediction”: pooled mid/final Brier has \(N<S\)
(\(0.154/0.153\) vs \(0.158/0.161\)). The predeclared alternative
fires: \(B_{\mathrm{C0}}^{S}<B_{\mathrm{C0}}^{N}\) (\(0.166<0.186\))
and C1-final Brier \(^{S}<^{N}\) (\(0.151<0.157\)).

\[
\boxed{
\text{contact sensing provides task-relevant epistemic information,
but its value is metric- and regime-dependent}
}
\]

Architecture: V7B.3 split world-truth from warranted belief; V7C shows
that warranted belief also does **not** need perfect access to the
causal contact variable. \(w_t\) remains a training principle, not the
final epistemic-state definition.

Do not reopen GRU gates, \(\tau\), or the B5-S proxy after this report.

## What this does not authorize

- feeding \(w_t\) or the C0 tape into the deployed model;
- claiming a real wrist FT sensor was used (this is a proprioceptive
  observer plus frozen noise);
- tactile images / RGB / V7D;
- occupancy penalties, Transformer, or GRU-gate search.

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
V7A\_GO &= true\\
V7B/B.1/B.2\_GO &= false\\
V7B.3\_GO &= true\\
V7C\_GO &= true\\
V7C.1/V7D &= locked
\end{aligned}
}
\]

Stage freeze: `REPORT/REP/R3_V7_STAGE_FREEZE.md`.

Next candidate **if** later unlocked: **V7C.1** (learned tactile
representation vs B5-S; reconstruction is not the GO). Prereg
`REPORT/REG/R3_V7C1_PREREG.md`. Not automatic. Not RGB. Not V7D.
