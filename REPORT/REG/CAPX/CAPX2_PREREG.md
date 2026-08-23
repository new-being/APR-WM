# CAP-X2 Preregistration — Structural-Mismatch Capacity Decay

Date: 2026-08-18  
Status: **FROZEN**; **formal complete** (`REPORT/REP/CAPX/CAPX2_REPORT.md`);
`cap_x2_passed=true`; pattern **`reference_failure`**; planning disabled;
no post-hoc retune of residual family / grids / competence gate / P0  
Depends on: `REPORT/REP/CAPX/CAPX1_REPORT.md` (`cap_x1_passed=true`,
\(R_P(0)=1.0\) matched-family upper bound),
`REPORT/REG/CAPX/CAPX_PREREG.md`, host `capx_arm3.v1`  
Does not: reopen / retune CAP-X1 grids or matched thresholds; Coulomb
convention lock; CAP-X3 few-shot; CAP-X4 visual; unlock R10; claim
parameter efficiency \(\equiv\) runtime efficiency; expand PureNN beyond
\(H=256\) to “rescue” incompetent \(\rho\) points

## Why X2 is not “just sweep \(\rho\)”

CAP-X1 fixed the left endpoint:

\[
\boxed{R_P(0)=1.0}
\]

as a **matched-family replacement upper bound** only. X1 also exposed two
facts that X2 must handle **in prereg**, not after peeking at curves:

1. **Planning gate non-discriminative:** \(S_{\mathrm{plan}}=0\) for all
   models under the X1 CEM budget (including Physics-only and PureNN
   \(H=256\)).
2. **Parameter \(\neq\) compute:** Physics-only CEM wall \(\approx 7.65\,\mathrm{s}\)
   vs PureNN reference \(\approx 0.35\,\mathrm{s}\); Hybrid \(\approx 22\)–\(27\,\mathrm{s}\).

Therefore CAP-X2 freezes (a) a calibrated outside-library residual family,
(b) per-\(\rho\) references + competence gate, (c) an **oracle-only**
planning-harness preflight **CAP-X2-P0**, and (d) runtime reporting that
forbids MLP-MAC-only “compute savings.”

## One question

\[
\boxed{
\text{As the fraction of dynamics unexplained by explicit physics }\rho
\uparrow,\text{ how does }R_P(\rho)\text{ decay?}
}
\]

This is the claim that answers:

> To what extent can physics replace latent / neural dynamics capacity
> once the plant leaves the matched family?

Primary objects:

\[
\boxed{R_P(\rho)},\qquad
\boxed{\rho_{50}},\qquad
\boxed{\rho_S(\rho,R_P)}
\]

Left endpoint inherited in meaning (not re-derived as a new X1 claim):

\[
R_P(0)\text{ under X2 protocol is the curve’s left end;}
\quad
\text{X1’s }R_P(0)=1.0\text{ remains the historical upper-bound label.}
\]

## 1. Definition of \(\rho\)

Do **not** treat an arbitrary residual coefficient as “mismatch strength.”

\[
\boxed{
\rho
=
\frac{\mathrm{RMS}(\tau_\perp)}
{\mathrm{RMS}(\tau_{\mathrm{phy}})}
}
\]

Frozen grid:

\[
\boxed{
\rho\in\{0,\,0.05,\,0.10,\,0.25,\,0.50,\,1.00\}
}
\]

\(\rho=0\) means \(\tau_\perp\equiv 0\) (same explicit family as X1); the
**definition** is inherited, not reinvented.

## 2. Frozen outside-library residual family (first formal family)

\[
\boxed{
\tau_{\perp,i}
=
\gamma_\rho
\Bigl[
\alpha_i\,\dot q_i|\dot q_i|
+
\beta_i\sin(2q_i)
+
\eta_i\sin(q_j-q_k)
\Bigr].
}
\]

Roles (all **outside** the explicit rigid-body + linear damping library):

| term | role |
|---|---|
| \(\alpha_i\dot q_i|\dot q_i|\) | nonlinear velocity dependence |
| \(\beta_i\sin(2q_i)\) | nonlinear configuration dependence |
| \(\eta_i\sin(q_j-q_k)\) | cross-joint interaction |

Index convention (frozen, 3-DoF): for joint \(i\), \((j,k)\) is the other
two joints in cyclic order \((0{\to}1,2),\ (1{\to}2,0),\ (2{\to}0,1)\).

**Critical fairness lock:**

\[
\boxed{
\text{learner does not observe }\alpha,\beta,\eta,\rho,\gamma_\rho
\text{ or residual-family labels.}
}
\]

Scene generation draws and **freezes** \(\alpha,\beta,\eta\) per scene
(fixed seed schedule; written to oracle-only scene metadata). Learners
see only \(x=[q,\dot q,u,\theta_e]\) as in X1. Otherwise the cell
collapses to low-dimensional physics-parameter ID, not residual-capacity
replacement.

Coulomb \(\mu\) remains **schema-only / value 0** throughout X2 (same as
X1). Enabling Coulomb needs a separate convention-lock cell.

## 3. Per-scene \(\rho\) calibration (held-out)

Each scene generates one additional trajectory:

```text
rho_calibration_trajectory
```

Constraints (frozen):

- **not** in train;
- **not** in val;
- **not** in test;
- **invisible** to every learner.

On that trajectory, with unscaled \(g(q,\dot q)\) (the bracket in §2 at
\(\gamma_\rho=1\)):

\[
R_{\perp,e}=\mathrm{RMS}(g(q,\dot q)),
\qquad
R_{\mathrm{phy},e}=\mathrm{RMS}(\tau_{\mathrm{phy}}).
\]

Then

\[
\gamma_{\rho,e}
=
\rho\cdot
\frac{R_{\mathrm{phy},e}}{R_{\perp,e}+\varepsilon}
\quad(\varepsilon=10^{-12}).
\]

So \(\rho=0.25\) means the **same relative unmodeled-force strength**
across scenes, not “strong on some scenes, weak on others.”

Train/val/test pools at each \(\rho\) apply the frozen \(\gamma_{\rho,e}\)
and scene \((\alpha,\beta,\eta)\) when integrating the plant.

## 4. Information parity (all models)

Input for every learner:

\[
x=[q,\dot q,u,\theta_e].
\]

| family | predictor |
|---|---|
| **PureNN** | \(f_\psi(q,\dot q,u,\theta_e)\to\hat{\ddot q}\) |
| **Physics-only** | \(F_{\mathrm{phy}}(q,\dot q,u;\theta_e)\) — **no** \(\tau_\perp\) |
| **Hybrid** | \(\hat{\ddot q}=F_{\mathrm{phy}}+M^{-1}\tau_{\mathrm{res},\psi}\) with residual input **only** \(x\) |

Forbidden to residual / PureNN: residual-family id, \(\rho\), \(\gamma_\rho\),
\(\alpha,\beta,\eta\).

Hybrid convention remains X1’s: residual outputs generalized force
\(\tau_{\mathrm{res}}\). \(H=0\) skips the residual net (Physics-only).

## 5. Width / train protocol (inherit X1; no peeking)

\[
\boxed{
H_{\mathrm{pure}}\in\{8,16,32,64,128,256\}
}
\]

\[
\boxed{
H_{\mathrm{hybrid}}\in\{0,8,16,32,64,128,256\}
\quad(H=0\equiv\text{Physics-only})
}
\]

Inherited frozen training (identical across \(\rho\) and \(H\)):

| item | value |
|---|---|
| optimizer | AdamW |
| lr | \(10^{-3}\) |
| weight decay | \(10^{-4}\) |
| batch | 512 |
| epochs | 40 + cosine |
| early-stop | patience 8 on val one-step NRMSE; restore best |
| seeds | \(\{101,102,103,104,105\}\) |
| per-width / per-\(\rho\) tuning | **forbidden** |

Splits / scene counts / traj recipe follow CAP-X0/X1 unless a new X2 data
cell must regenerate scenes for \(\tau_\perp\) (same 128/32/64 counts and
disjoint seed discipline). Do **not** change grids after seeing
\(R_P(\rho)\).

## 6. Per-\(\rho\) reference (critical)

Do **not** reuse \(\rho=0\) PureNN \(H=256\) as the reference for all
\(\rho\).

\[
\boxed{
Q_{\mathrm{ref}}(\rho)
=
\text{PureNN }H=256\text{ trained and evaluated at the same }\rho
}
\]

Aggregate 5-seed test means:

\[
E_{1,\mathrm{ref}}(\rho),\quad
E_{\mathrm{roll},\mathrm{ref}}(\rho),\quad
S_{\mathrm{plan},\mathrm{ref}}(\rho).
\]

Question at each \(\rho\): how much Hybrid capacity is needed to match
PureNN **at the same difficulty**.

## 7. Reference competence gate

\[
\boxed{
E_{1,\mathrm{ref}}(\rho)\le 0.25
}
\]

Else:

```text
reference_incompetent
R_P(rho) = undefined
```

**Forbidden:** expand to \(H=512/1024\) (or retune) to complete the curve.
Incompetent points stop the replacement claim (Pattern D).

## 8–9. Planning: CAP-X2-P0 harness lock (before capacity sweep)

X1 planning is not retuned. X2 declares a **new** evaluation instrument.

### CAP-X2-P0 (oracle-only preflight)

\[
\boxed{\textbf{CAP-X2-P0}}
\]

- Uses **oracle** plant dynamics only (true \(F_{\mathrm{phy}}+\tau_\perp\) at
  the \(\rho\) used for harness lock — default lock at \(\rho=0\) plant,
  unless prereg amendment says otherwise **before** any capacity run).
- **Does not** train or inspect PureNN / Hybrid capacity curves.
- **Does not** unlock or rewrite CAP-X1.

#### Reachable-target construction

Do **not** sample \(q^\star\) from a random train-\(q\) box.

From \(s_0\), draw a random torque sequence \(u_{0:H-1}^{\mathrm{oracle}}\)
with the **same action bounds** later used by CEM, roll out on the true
plant to \(s_H\), and set

\[
q^\star = q_H.
\]

Thus at least one feasible open-loop sequence exists by construction.

Other planning cost / horizon defaults (candidates/iters chosen by P0):

```text
horizon        = 1.0 s
model_dt       = 0.01 s
action_knots   = 20
elite_frac     = 0.10
ε_q            = 0.15 rad (per-joint max-norm)
λ_v            = 0.05
λ_u            = 0.01
```

#### P0 budget grid (frozen; choose once)

\[
(\mathrm{cand},\mathrm{iter})
\in
\{
(256,5),\ (512,5),\ (512,8),\ (1024,8)
\}.
\]

Select

\[
\boxed{
\text{smallest budget with oracle success }\ge 0.80
}
\]

and **permanently freeze** that \((\mathrm{cand},\mathrm{iter})\) for all
X2 capacity models and all \(\rho\).

If the **largest** budget still has \(S_{\mathrm{oracle}}<0.80\):

\[
\boxed{\text{planning component disabled for X2}}
\]

Then X2 reports **prediction/rollout capacity only**. **Do not** enlarge
the budget grid.

## 10. Matched performance at each \(\rho\)

If P0 **locks** a planner:

\[
\begin{aligned}
M_1&:\ E_1\le 1.05\,E_{1,\mathrm{ref}}(\rho)\\
M_2&:\ E_{\mathrm{roll}}\le 1.10\,E_{\mathrm{roll},\mathrm{ref}}(\rho)\\
M_3&:\ S_{\mathrm{plan}}\ge S_{\mathrm{plan},\mathrm{ref}}(\rho)-0.05\\
M&=M_1\land M_2\land M_3.
\end{aligned}
\]

If P0 **disables** planning:

\[
M=M_1\land M_2
\]

and all papers / reports must state:

> X2 capacity replacement is measured for **predictive dynamics**, not
> planning.

Do not keep a non-discriminative \(S_{\mathrm{plan}}\) gate as fake planning
evidence.

## 11. Replacement curve

\[
P_{\mathrm{pure}}^{\min}(\rho)
=
\min_H P(H_{\mathrm{pure}})
\quad\text{s.t. }M=1
\]

\[
P_{\mathrm{hybrid}}^{\min}(\rho)
=
\min_H P(H_{\mathrm{hybrid}})
\quad\text{s.t. }M=1
\]

\[
\boxed{
R_P(\rho)
=
1-
\frac{P_{\mathrm{hybrid}}^{\min}(\rho)}
{P_{\mathrm{pure}}^{\min}(\rho)}
}
\]

**Allow \(R_P<0\)** (Hybrid needs more learned \(P\) than PureNN). **Do not
clip to 0.** Negative values are scientific boundary, not defects.

If no PureNN width matches \(Q_{\mathrm{ref}}\) under \(M\) at a competent
\(\rho\), or reference is incompetent: \(R_P(\rho)=\mathrm{undefined}\)
(no fabricated ratio).

## 12. Shape hypothesis (not pretty percentages)

Do **not** preregister claims like \(R_P(0.1)>0.70\).

Frozen shape hypothesis on **reference-competent** points only:

\[
\boxed{
R_P(0)\ge R_P(0.05)\ge R_P(0.10)\ge R_P(0.25)\ge R_P(0.50)\ge R_P(1.0)
}
\]

allowing small statistical noise (isotonic / trend-compatible), with
primary association test:

\[
\boxed{
\rho_S(\rho,R_P)\le -0.7
}
\]

(Spearman; competent points only). Secondary narrative: \(\mathrm{corr}(\rho,R_P)<0\).

## 13. Half-replacement point

\[
\boxed{
\rho_{50}
=
\sup\{\rho:\,R_P(\rho)\ge 0.5\}
}
\]

(over competent points with defined \(R_P\)). Interpretive summary, e.g.
“unmodeled generalized-force RMS up to \(\sim\rho_{50}\) of nominal
physics still replaces \(\ge 50\%\) of learned dynamics parameters.”

## 14. Runtime facts (must measure; must not hide)

Continue reporting, for every \((\rho,H,\mathrm{family})\):

\[
T_{\mathrm{step}}(\rho,H),\qquad T_{\mathrm{plan}}(\rho,H)
\]

plus trainable \(P\) and neural MACs (neural module only).

\[
\boxed{
\text{parameter efficiency }\neq\text{ runtime efficiency}
}
\]

**Forbidden:** because Physics/Hybrid CEM is slower, redefine primary
compute claim as MLP-MAC savings alone. X1 already showed Physics CEM
\(\gg\) PureNN CEM wall-clock; X2 records this as a **real limitation**.
Batched / analytic dynamics acceleration is a **future** cell, not an X2
escape hatch.

## 15. Frozen result patterns (label after formal; do not retune)

| id | pattern | meaning |
|---|---|---|
| **A** | `structured_capacity_decay` | \(R_P(0)\approx 1\) and clear decay in \(\rho\) — capacity demand tracks unexplained force |
| **B** | `persistent_structure_advantage` | high \(R_P\) even at \(\rho=1\) — stronger than expected; do not anticipate |
| **C** | `early_crossover` | small \(\rho\) already \(R_P\approx 0\) or \(R_P<0\) — structure fragile outside matched family |
| **D** | `reference_failure` | large-\(\rho\) PureNN \(H=256\) incompetent — curve stops; no width rescue |

## 16. Primary figures

\[
\boxed{x=\rho,\quad y=R_P(\rho)}
\]

and companion

\[
\boxed{P_{\mathrm{hybrid}}^{\min}(\rho)}
\]

(with competent / undefined markers). X1 answered only
\(\rho=0\Rightarrow 100\%\) parameter replacement (upper bound). X2
answers **how fast that advantage disappears** off the ideal family.

## Execution order (after explicit start)

```text
1) Implement residual family + calibration data gen (no capacity train yet)
2) CAP-X2-P0 oracle harness → freeze (cand,iter) OR disable planning
3) Only then: per-ρ datasets → width×seed sweeps → R_P(ρ), ρ_50, ρ_S
4) Write CAPX2_REPORT.md; update ledgers
```

No step-3 peeking that changes residual family, \(\rho\) grid, widths,
matched thresholds, competence gate, or P0 budget set.

## Gates

| id | requirement |
|---|---|
| **G-P0** | Oracle harness run recorded; budget frozen **or** planning disabled with explicit predictive-only claim |
| **G0** | At \(\rho=0\), oracle accounting still \(\mathrm{NRMSE}<10^{-4}\) on \(\tau_{\mathrm{phy}}\) path (\(\mu=0\)) |
| **G-calib** | Every scene has held-out `rho_calibration_trajectory`; learners never see it |
| **G-info** | No model receives \(\alpha,\beta,\eta,\rho,\gamma_\rho\) / family labels |
| **G-ref** | Per-\(\rho\) PureNN \(H=256\) reference logged; competence gate applied |
| **G-grid** | Full \(H\) grids × 5 seeds for every **competent** \(\rho\) used in \(R_P\) |
| **G-curve** | \(R_P(\rho)\) (allowing \(<0\) / undefined) + \(\rho_{50}\) + Spearman on competent points |
| **G-runtime** | \(T_{\mathrm{step}}\), \(T_{\mathrm{plan}}\) reported; no MAC-only compute primary |
| **G-label** | Artifacts under `runs/cap_x2/`; does not unlock R10; X1 not rewritten |

## Claims ceiling

**Allowed if PASS:** under calibrated outside-library \(\tau_\perp\), report
how \(R_P(\rho)\) decays (shape / \(\rho_S\) / \(\rho_{50}\)) at matched
predictive (and planning iff P0 locked) performance vs per-\(\rho\)
PureNN \(H=256\).

**Forbidden:** real-world 100% replacement; treating X1 \(R_P(0)=1\) as
off-family evidence; Coulomb solved; R10 unlock; post-hoc width/budget
rescue; clipping \(R_P\) at 0; pretending parameter savings are runtime
savings.

## Unlock

\[
\text{CAP-X2 formal complete}
\;\Rightarrow\;
\text{preregister CAP-X3 (few-shot }\theta_e\text{ vs latent)}
\]

## Ledger

```text
CAP-X0    = PASS
CAP-X1    = PASS
R_P(0)    = 1.0 matched-family upper bound

CAP-X2-P0 = PASS / planning DISABLED
CAP-X2    = PASS protocol / Pattern D reference_failure
            R_P defined only at rho=0 (R_P=1.0); rho>=0.05 undefined
            report = REPORT/REP/CAPX/CAPX2_REPORT.md

CAP-X3 = LOCKED
CAP-X4 = LOCKED
PLAN-X = PLAN-X0 PASS; PLAN-X1 FAIL anisotropy_no_value (H1 yes, H2 no);
         PLAN-X2 LOCKED not triggered; CAP-X3 LOCKED
R10    = LOCKED
```

Implementation / P0 / \(\rho\) sweep starts only on explicit request after
this freeze (“开写” / “开始” / “run CAP-X2” / “run CAP-X2-P0”).
