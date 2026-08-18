# CAP-X1 Preregistration — Matched-Family Capacity Replacement (\(\rho=0\))

Date: 2026-08-17  
Status: **FROZEN**; **formal PASS** (`REPORT/REP/CAPX/CAPX1_REPORT.md`);
width / planner / matched gates remain frozen — no post-hoc retune  
Depends on: `REPORT/REP/CAPX/CAPX0_REPORT.md` (`cap_x0_passed=true`),
`REPORT/REG/CAPX/CAPX_PREREG.md`, host `capx_arm3.v1`  
Does not: \(\rho>0\) (CAP-X2); Coulomb convention lock; scene few-shot
(CAP-X3); visual (CAP-X4); unlock R10; retune widths / planner / matched
gates after seeing capacity curves; treat X0 val NRMSE \(0.101\) as the
formal PureNN reference

## One question

\[
\boxed{
\text{Under matched physics }(\rho=0),\text{ how much trainable neural
capacity does explicit structure replace at matched prediction
\textit{and} planning performance?}
}
\]

Primary scalar:

\[
\boxed{R_P(0)}
\]

Interpretation ceiling if Physics-only already matches:

\[
\boxed{\textbf{matched-family replacement upper bound}}
\]

— **not** “physics replaces 100% of latent in the real world.”

## Plant / data (carry from CAP-X0)

| item | frozen |
|---|---|
| host | `capx_arm3.v1` |
| splits | 128 / 32 / 64 scenes (same seeds / fingerprints as X0 formal) |
| \(\rho\) | **0** (plant ∈ explicit family; no \(\tau_\perp\)) |
| Coulomb \(\mu\) | **schema slots remain; values frozen to 0 throughout X1** |

\[
\boxed{
\mu\text{ slot remains in schema, but is frozen to zero throughout X1.}
}
\]

Enabling Coulomb requires a **separate convention-lock cell**. It must
**not** be mixed into the X1 capacity sweep (X0 already showed MuJoCo
`frictionloss` ≉ current \(-\mu\mathrm{sign}(v)\) accounting).

Oracle \(\theta_e\) is visible to Physics / Hybrid / PureNN context
exactly as in X0 (fair structured context vs structured physics).

## Width grids (frozen before any run)

\[
\boxed{
H_{\mathrm{pure}}\in\{8,16,32,64,128,256\}
}
\]

\[
\boxed{
H_{\mathrm{hybrid}}\in\{0,8,16,32,64,128,256\}
}
\]

\[
H_{\mathrm{hybrid}}=0
\;\equiv\;
\textbf{Physics-only}
\quad(d_z=0,\;\text{no residual net}).
\]

Architecture family (same as X0 PureNN backbone, scaled by \(H\)):

- PureNN: MLP \([q,\dot q,u,\theta_e]\to\ddot q\), three hidden SiLU
  layers of width \(H\), linear head.
- Hybrid: explicit rigid-body \(\hat{\ddot q}_{\mathrm{phy}}(\theta_e)\)
  plus residual MLP on the **same input** producing \(\tau_{\mathrm{res}}\)
  (or \(\Delta\ddot q\); declare one convention in the implementation
  header and never switch mid-sweep). \(H=0\) skips the residual MLP.

## Training protocol (identical across all widths)

| item | frozen |
|---|---|
| optimizer | AdamW |
| lr | \(10^{-3}\) |
| weight decay | \(10^{-4}\) |
| batch size | 512 |
| schedule | cosine over full update budget |
| update budget | **40 epochs** over the X0 train pool (or equal #steps if pool subsampled — same subsample rule for all \(H\)) |
| early-stop | patience **8** epochs on val one-step NRMSE; restore best |
| train seeds | **5** fresh seeds: \(\{101,102,103,104,105\}\) |
| per-width tuning | **forbidden** |

X0’s single sanity run (val NRMSE \(0.101\)) is **not** the formal
reference. Reference metrics come only from X1 fresh PureNN \(H=256\)
runs under this protocol.

## Reference model

\[
\boxed{Q_{\mathrm{ref}}=\text{PureNN }H=256}
\]

Aggregate over the 5 seeds (mean on the **test** split unless noted):

\[
E_{1,\mathrm{ref}},\quad
E_{\mathrm{rollout,ref}},\quad
S_{\mathrm{plan,ref}}.
\]

## Matched performance (all three required)

One-step acceleration error on test:

\[
E_1=\mathrm{NRMSE}(\hat{\ddot q},\ddot q).
\]

Open-loop state rollout (model-rate \(\Delta t=0.01\,\mathrm{s}\)) at
horizons \(\{10,50,100\}\) steps; report

\[
E_{\mathrm{rollout}}
=
\mathrm{mean}_{H\in\{10,50,100\}}
\mathrm{RMSE}(\hat s_{t+H},s_{t+H})
\]

(with \(s=(q,\dot q)\)).

Planning success under the frozen CEM budget below:

\[
S_{\mathrm{plan}}
=
\mathbb{P}
\bigl(\|q_T-q^\star\|<\varepsilon_q\;\wedge\;\text{no limit violation}\bigr),
\]

with \(\varepsilon_q=0.15\,\mathrm{rad}\) (per-joint max-norm), \(T=\)
horizon end, random \(q^\star\) drawn in the observed train \(q\) box.

**Matched** iff all hold (vs reference means):

\[
\boxed{
E_1\le 1.05\,E_{1,\mathrm{ref}}
}
\]

\[
\boxed{
E_{\mathrm{rollout}}\le 1.10\,E_{\mathrm{rollout,ref}}
}
\]

\[
\boxed{
S_{\mathrm{plan}}\ge S_{\mathrm{plan,ref}}-0.05
}
\]

## Planner budget (identical for every world model)

```text
horizon        = 1.0 s
model_dt       = 0.01 s
action_knots   = 20
CEM candidates = 256
CEM iterations = 5
elite_frac     = 0.10
```

Cost (frozen):

\[
J=\sum_t
\bigl(
\|q_t-q^\star\|^2
+\lambda_v\|\dot q_t\|^2
+\lambda_u\|u_t\|^2
\bigr),
\qquad
\lambda_v=0.05,\;\lambda_u=0.01.
\]

No model may receive extra CEM samples, longer horizon, or different
\(\lambda\).

## Primary statistic \(R_P(0)\)

Let \(P(H)\) be trainable parameter count of the neural module
(PureNN full net; Hybrid residual net only — Physics-only has \(P=0\)).

\[
P_{\mathrm{pure}}^{\min}
=
\min_H P(H_{\mathrm{pure}})
\quad\text{s.t. matched performance (5-seed mean)},
\]

\[
P_{\mathrm{hybrid}}^{\min}
=
\min_H P(H_{\mathrm{hybrid}})
\quad\text{s.t. matched performance (5-seed mean)}.
\]

\[
\boxed{
R_P(0)
=
1-
\frac{P_{\mathrm{hybrid}}^{\min}}
{P_{\mathrm{pure}}^{\min}}
}
\]

If Physics-only (\(H=0\)) is already matched:

\[
P_{\mathrm{hybrid}}^{\min}=0
\;\Rightarrow\;
R_P(0)=1,
\]

reported only as **matched-family replacement upper bound**.

If no PureNN width matches its own \(H=256\) reference under the
protocol: **STOP** — optimizer/protocol bug, not a physics win.

## Compute reporting (not a sole \(R_F\) from MLP MACs)

Do **not** define primary \(R_F\) from neural-MACs alone (explicit
dynamics also cost). X1 **must report all of**:

| quantity | meaning |
|---|---|
| \(P\) | trainable parameter count |
| neural MACs / step | residual or PureNN only |
| full-model step latency | physics + neural wall-clock / model step |
| CEM planning wall-clock | end-to-end planning time, same budget |

Optional diagnostic ratios on latency are allowed in the report; they
are **not** the primary claim. The scientifically central curve remains
CAP-X2’s \(R_P(\rho)\) for \(\rho>0\).

## Gates

| id | requirement |
|---|---|
| **G0** | Carry X0 oracle accounting on a frozen smoke subset: \(\mathrm{NRMSE}<10^{-4}\) with \(\mu=0\) |
| **G-ref** | PureNN \(H=256\) 5-seed test \(E_1\) finite; protocol intact |
| **G-grid** | All listed \(H_{\mathrm{pure}}\) and \(H_{\mathrm{hybrid}}\) trained × 5 seeds |
| **G-match** | \(P_{\mathrm{pure}}^{\min}\) and \(P_{\mathrm{hybrid}}^{\min}\) exist under the triple matched rule (else report failure mode, no fabricated \(R_P\)) |
| **G-label** | Artifacts under `runs/cap_x1/`; `rho=0`; `mu=0`; `capacity_claim` only via \(R_P(0)\) with the upper-bound wording if \(H=0\) matches |

Suggested (non-binding until after formal, **not** for retuning):
strong replacement if \(R_P(0)\ge 0.75\); moderate if \(\ge 0.50\).
**Do not change grids or matched thresholds after seeing results.**

## Claims ceiling

**Allowed if PASS:** under \(\rho=0\) matched family, explicit structure
replaces fraction \(R_P(0)\) of trainable dynamics parameters at
matched one-step, rollout, and planning quality.

**Forbidden:** real-world 100% replacement; Coulomb solved; \(\rho>0\)
extrapolation; R10 unlock; latent-dim-only “\(4\times\) cheaper.”

## Unlock

\[
\text{CAP-X1 formal complete}
\;\Rightarrow\;
\text{preregister CAP-X2 }R_P(\rho)\text{ grid}
\]

## Ledger

```text
CAP-X0 = PASS

CAP-X1 = PASS  (R_P(0)=1.0 matched-family upper bound; Physics-only matched)
         rho = 0
         mu  = 0  (schema slots present; values frozen)
         report = REPORT/REP/CAPX/CAPX1_REPORT.md
         artifacts = runs/cap_x1/formal/

CAP-X2 = LOCKED (prereg frozen: REPORT/REG/CAPX/CAPX2_PREREG.md; no sweep)
R10    = LOCKED
```

Formal complete. CAP-X2 implementation / P0 / \(\rho\) sweep starts only
on explicit request after `CAPX2_PREREG.md` freeze.
