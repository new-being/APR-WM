# R3-V7B Preregistration — Persistent Contextual Epistemic Belief

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R3_V7A_REPORT.md` (`V7A_GO=true`)  
Does not change: any frozen RS* GO, C0, \(s^\star\), RS1C policy, `V7A_GO`  
Locks: **RS5B/C**, revision, VoI, consequence, H32, RGB, tactile  
Does **not** add \(p_{\mathrm{param}},p_{\mathrm{unknown}},\hat C,\hat V\) heads

## Placement

\[
\boxed{
V7A\_GO=true
\rightarrow
R3\text{-V7B (this stage)}
\qquad
R3\text{-V7C/D locked}
}
\]

Not RS5B. Not the old 2D V7 routing experiment.

Stage name:

\[
\boxed{
R3\text{-V7B — Persistent Contextual Epistemic Belief}
}
\]

中文名：**持续更新的情境认识信念**。

## Question

\[
\boxed{
\text{Does maintaining history improve epistemic calibration
beyond a static episode summary?}
}
\]

Primary comparison is **V7B vs V7A-style static contextual**, not vs \(D_0\).

\(D_0\)-only is logged as B0 only.

## State and readout

\[
u_{t+1}^{\mathrm{epi}}
=
F_\psi\!\left(u_t^{\mathrm{epi}},\,o_t,\,a_t,\,r_{\perp,t},\,z_t^{\mathrm{geom}}\right)
\]

\[
\hat p_{\mathrm{struct},t}=H(u_t^{\mathrm{epi}})
\quad\text{at every downsampled time, not only at episode end.}
\]

\(F_\psi\): GRU, hidden size **32**, one layer. No Transformer.

Inputs are learner-visible: oracle hinge state / proprioception, action
features, parameter-tangent orthogonal residual, local interaction
geometry. **No** family / script / phase / domain-ID channels.

## Baselines (same new mixture)

**B0.** Episode \(D_0\)-only logistic (V7A's weaker baseline).  
**B1.** V7A-style static contextual logistic \(p(\tilde h)\) on episode
geometry `FEATURE_NAMES`, **retrained** on this stage's development
mixture (includes `switch_cycle`).  
**B2.** GRU \(p_t\).

Causal static-window for H2: apply the **same B1 weights** to the
prefix summary \(\tilde h_{1:t}\) at each \(t\).

## Data

Fresh seeds \(\{17101,17111,17121\}\) train/val, \(\{17131,17141\}\)
held-out. Val = last development seed (`17121`); do not train on
held-out.

Mixture = V7A grid **plus** in-episode interaction switch:

```text
free / approach → pull contact → release → re-grasp → push → quiet
```

script name `switch_cycle` (not added to frozen RS2 `SCRIPTS`).

Regimes C0 / C1-L / C1-H. Oracle state + oracle contact only.
Downsample all traces to **20 Hz** (contact: control rate; Mode-A:
stride from 2 ms).

\(\theta\) for \(r_\perp\) is fit on a **causal H0 prefix** only
(contact: `phase<2`; Mode-A: first 20% of the downsampled episode).

## Hypotheses (GO)

**H1 — Final calibration.** Held-out episode-end Brier:

\[
\mathrm{Brier}_{B2}(p_T) < \mathrm{Brier}_{B1}(\tilde h).
\]

**H2 — Early evidence.** Held-out Brier at the causal midpoint
\(t=\lfloor(T-1)/2\rfloor\):

\[
\mathrm{Brier}_{B2}(p_{t_{\mathrm{mid}}})
<
\mathrm{Brier}_{B1}(\tilde h_{1:t_{\mathrm{mid}}}).
\]

Detection time \(t_{\mathrm{det}}=\min\{t:p_t>\tau\}\) is reported,
not a GO (can be vacuous if neither model crosses \(\tau\)).

**H3 — C0 temporal stability.** With \(\tau=\) 95th percentile of
**development C0 final** \(p_T\) for B2,

\[
P(p_t>\tau\mid \text{held-out C0, all }t) \le 0.20.
\]

\[
\boxed{V7B\_GO = H1 \land H2 \land H3}
\]

Smoke must not set `V7B_GO`.

## Diagnostic (not GO)

- B0 vs B1 vs B2 final Brier (B2 need not beat B0 if B1 already does;
  H1 is vs B1).
- Median \(t_{\mathrm{det}}/T\) on held-out C1, B2 vs B1.
- Mean false-positive dwell (run length of \(p_t>\tau\)) on held-out C0.
- **Belief recovery** on held-out C0 `switch_cycle`: after quiet onset
  (`phase==5`), steps until \(p_t\le\tau\) if a contact-phase exceedance
  occurred; else 0. Report median. Hysteresis
  (“once suspicious, always suspicious”) is a failure *mode* to
  describe, not a silent rewrite of H1–H3.

## Loss

Time-weighted BCE on B2: \(w_t=0.3+0.7\,t/(T-1)\), episode label
\(Y\in\{0,1\}\) (regime \(\neq\) C0). Optimizer: **SGD** on GRU
parameters (no `torch.optim.Adam` after MuJoCo import stubs). Early-stop
on val **final** Brier.

## What success does not authorize

- RGB / tactile (those are V7C/D);
- revision / VoI / H32 / consequence heads;
- RS5B;
- claiming invariant scalars or LOIO transport;
- treating a 32-unit GRU as the full \(b_t=(s_t,\theta_t,M_t,u_t^{epi})\).

## What failure would mean

Persistent memory does not improve calibration over V7A static
summaries on this Door mixture. Then diagnose hysteresis vs capacity
before enlarging the temporal model. Do **not** return to invariant
scalars or RS5 indexing as the mainline.
