# CAP-X Preregistration — Physics-to-Neural Capacity Replacement

Date: 2026-08-17  
Status: **FROZEN** (family); **CAP-X0 PASS**; **CAP-X1 PASS**;
**CAP-X2 COMPLETE / `reference_failure`**; **CAP-X3 PASS**
(`structured_adaptation_advantage`); X4 locked; **PLAN-X closed on this
host** (X1 `anisotropy_no_value`, X1.5 `iso_sufficient`; PLAN-X2 not
triggered); does not unlock R10-C0;
does not reopen VIS/SIM/R9 toy claims  
Depends on: VIS-EXT0 PASS (visual optional later as CAP-X4 only);
`aprwm_v0/mujoco_force.py`  
Does not: 1-DoF hinge as capacity host; latent-dim-only “capacity”;
retuning gates after formal capacity curves; RoboCasa as CAP host

## One question

\[
\boxed{
\textbf{显式结构物理 + 少量场景参数 + 小 residual，
能否以显著更小的神经容量达到纯 neural dynamics 相同的预测与规划性能？}
}
\]

The answer must include how that advantage **decays with structural
mismatch** \(\rho\), not a single compression percentage.

## Route (frozen)

\[
\boxed{
\text{CAP-X0}
\rightarrow
\text{X1}
\rightarrow
\text{X2}
\rightarrow
\text{X3}
\rightarrow
\text{X4(optional visual)}
}
\]

| Cell | Question |
|---|---|
| **CAP-X0** | Benchmark / accounting / baseline fairness |
| **CAP-X1** | Matched physics: how much neural capacity is replaceable |
| **CAP-X2** | Replacement rate vs unmodeled residual \(\rho\) |
| **CAP-X3** | New-scene few-shot: structured \(\theta_e\) vs latent \(c_e\) |
| **CAP-X4** | Does capacity advantage survive frozen visual front-end |

**CAP-X0 PASS.** CAP-X1 formal PASS (\(R_P(0)=1\) upper bound).
**CAP-X2 prereg FROZEN** (`CAPX2_PREREG.md`): calibrated \(\rho\),
outside-library \(\tau_\perp\), per-\(\rho\) reference + competence gate,
oracle-only **CAP-X2-P0** planner lock (or planning disabled), runtime
honesty. No \(\rho\) capacity sweep until explicit start. Never retune X1
gates after seeing capacity curves.

## Host plant (not 1-DoF)

\[
\boxed{\texttt{capx\_arm3.v1}}
\]

3-DoF MuJoCo serial arm: revolute joints; `qfrc_applied` drive; no
contact; shared kinematic structure \(\psi_{\mathrm{structure}}\);
scenes vary only low-dimensional \(\theta_e\).

## Parameter split

\[
\theta_e
=
\{
m_i,\,I_i,\,b_i,\,\mu_i,\,m_{\mathrm{payload}}
\}
,\qquad
d_\theta\approx 10\text{–}20.
\]

## Three model families (fairness)

- **A. Pure Neural:** \(\hat{\ddot q}=f_\psi(q,\dot q,u,c)\)
- **B. Physics Only:** rigid-body inverse dynamics, \(d_z=0\)
- **C. Physics + Neural Residual:** same + \(\tau_{\mathrm{res},\psi}\)

Capacity must be reported as trainable **parameters \(P\)**, **FLOPs/step
\(F\)**, **inference memory \(M\)** — not latent width alone.

## Capacity metrics (X1+)

At matched prediction/planning quality vs PureNN reference:

\[
R_P=1-\frac{P_{\mathrm{hybrid}}^{\min}}{P_{\mathrm{pure}}^{\min}},
\qquad
R_F=1-\frac{F_{\mathrm{hybrid}}^{\min}}{F_{\mathrm{pure}}^{\min}}.
\]

## Locked ledger

```text
VIS-X0–X3 / VIS-EXT0 = PASS
CAP-X0                = PASS
CAP-X1                = PASS  R_P(0)=1.0 matched-family upper bound
CAP-X2-P0             = PASS / planning DISABLED
CAP-X2                = COMPLETE  pattern=reference_failure
CAP-X3-P0             = PASS
CAP-X3                = PASS  structured_adaptation_advantage
                         (beyond dim reduction; R_K=0.5)
CAP-X3B / CAP-X4      = LOCKED
PLAN-X0               = PASS
PLAN-X1               = FAIL  anisotropy_no_value
H1                    = SUPPORTED  (search location)
H2                    = REJECTED on this host/task
PLAN-X1.5             = FAIL  iso_sufficient
PLAN action prior     = μ_φ + σ0² I sufficient; no diffusion
PLAN-X2               = LOCKED (not triggered)
RoboTwin-X0-P0        = PASS  (proxy drawer)
RoboTwin-X0-smoke     = FAIL  (Vulkan / SapienRenderer)
RoboTwin-X0           = FROZEN (official 3-task locked)
R10                   = LOCKED
```

Principle:

\[
\boxed{
\textbf{When environment variation is largely captured by a
low-dimensional physical parameterization, explicit structure
substitutes for learned dynamics capacity; residual capacity
tracks remaining structural mismatch.}
}
\]

CAP-X3 (matched few-shot; not a capacity-\(R_P\) claim):

\[
\boxed{
\textbf{causal parameterization provides adaptation efficiency
beyond dimensionality reduction}
}
\]

CAP-X3B / CAP-X4 / R10 remain locked. Do not open residual-\(\rho\)
or visual cells to “complete” X3.
