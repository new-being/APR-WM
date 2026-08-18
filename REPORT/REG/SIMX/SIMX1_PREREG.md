# SIM-X1 Preregistration — Oracle Damping Mismatch Closure

Date: 2026-08-17  
Status: **FROZEN**; **PASS** (`REPORT/REP/SIMX/SIMX1_REPORT.md`);
unlocks SIM-X2 only; **not an R10 gate**  
Depends on: `REPORT/REP/SIMX/SIMX0_REPORT.md`,
`REPORT/REG/SIMX/SIMX0_FORCE_ACCOUNTING.md`,
`aprwm_v0/simx_plant.py` (`simx_hinge.v1`)  
Does not: unlock R10-C0; SIM-X2/X3; validity; certificate;
\(I(\mathcal V;Y\mid a)\); action ranking; actuator saturation;
payload / frictionloss / geometry change; copy true damping into
learner nominal

## One question

\[
\boxed{
\text{On the same X0 host, does a known damping intervention
appear as the correct generalized-force mismatch under frozen
learner nominal?}
}
\]

Not: which action is more informative. That is X2.

## Host (frozen from X0)

Same kinematic / drive plant: `simx_hinge.v1`.

Unchanged vs X0:

- geometry, inertia, timestep, gravity
- frictionloss \(=0\), contact off, limits off
- drive = `qfrc_applied` only (`nu=0`)

Only `dof_damping` may change.

## Parameter isolation (anti-leakage)

\[
\boxed{
b_{\mathrm{true}}\text{ variable},
\qquad
b_{\mathrm{learner}}=b_0=0.10\text{ forever frozen}
}
\]

| object | symbol | X1 values |
|---|---|---|
| `true_plant_params.damping` | \(b_{\mathrm{true}}\) | \(\{0.05,\,0.10,\,0.15\}\) |
| `learner_nominal_params.damping` | \(b_0\) | \(0.10\) always |

**Forbidden:** reading MuJoCo `dof_damping` into `NominalPassiveParams`.
If \(b_{\mathrm{true}}=0.15\) and the learner also gets \(0.15\), the
oracle intervention vanishes.

`raw_truth/qfrc_passive` remains audit-only.

## Oracle residual

\[
\Delta b=b_{\mathrm{true}}-b_0,
\qquad
\tau_{\mathrm{passive}}=-b\dot q
\]

Learner always accounts with \(b_0\), so

\[
\boxed{
r_{\mathrm{oracle}}=-(b_{\mathrm{true}}-b_0)\dot q=-\Delta b\,\dot q.
}
\]

\(\dot q\) is the **pre-`mj_step`** velocity (same timing as X0).

## Matrix

\[
\{9101,9111,9121\}
\times
\{\mathrm{sine},\mathrm{chirp},\mathrm{piecewise}\}
\times
\{0.05,0.10,0.15\}
=
\boxed{27\text{ cells}}
\]

Same excitation family as X0 so the conclusion is not one-waveform.
**Do not** compare which action is more informative here.

## Gates

### G0 — nominal guard

When \(b_{\mathrm{true}}=b_0=0.10\):

\[
\mathrm{NRMSE}_{\tau}<10^{-4}.
\]

X1 parameterization must not break X0 closure.

### G1 — oracle residual reconstruction

\[
e_r=r-r_{\mathrm{oracle}},
\qquad
E_{\mathrm{oracle}}
=
\frac{\mathrm{RMSE}(r-r_{\mathrm{oracle}})}{\mathrm{RMS}(r_{\mathrm{oracle}})+\epsilon}.
\]

For both mismatch classes (\(b_{\mathrm{true}}\in\{0.05,0.15\}\)):

\[
E_{\mathrm{oracle}}<10^{-4}.
\]

Convention gate, not a beauty contest toward \(10^{-16}\).

### G2 — signed coefficient recovery

Fit no-intercept \(r=\hat s\,\dot q\). Oracle \(s^*=-\Delta b\).

\[
\frac{|\hat s-s^*|}{|s^*|}<10^{-4}
\]

on mismatch cells. Catches correct magnitude with wrong sign.

### G3 — no spurious intercept

Fit \(r=\hat s\,\dot q+\hat c\). Require \(\hat c\) on the numerical
floor relative to torque scale, e.g.

\[
\frac{|\hat c|}{\mathrm{RMS}(\tau_{\mathrm{known}})+\epsilon}<10^{-4}.
\]

Confirms \(r\propto\dot q\), not a constant bias.

## Forbidden in this cell

No detector. No validity. No certificate. No \(I(\mathcal V;Y\mid a)\).
No “informative vs conservative” ranking. No WM / NetVoI. No X2
language.

## PASS may claim

\[
\boxed{
\text{known damping intervention on the MuJoCo host is correctly
manifested as generalized-force mismatch under frozen nominal}
}
\]

and

\[
\boxed{r=-\Delta b\,\dot q}
\]

holds on this independent engine.

## PASS may **not** claim

- validity observability
- policy-induced blindness
- informative \(a\) beats conservative \(a\)
- certificate validity
- real physics

Those wait for X2+.

## Fail ⇒ interface / leakage, not capacity

If G0 fails: X1 broke the host adapter — fix before interpreting
mismatch. If G1–G3 fail: check sign, timing (\(\dot q\)), or
learner receiving \(b_{\mathrm{true}}\). Do not train a network.

## After a pass

Unlock **SIM-X2** only (action-conditioned information on the same
host / same damping family). R10-C0 stays locked.
