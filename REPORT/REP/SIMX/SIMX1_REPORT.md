# SIM-X1 Report — Oracle Damping Mismatch Closure

Date: 2026-08-17  
Status: **`sim_x1_passed=true`**; unlocks **SIM-X2 only**  
Does not: unlock R10-C0; claim real physics; validity; certificate;
\(I(\mathcal V;Y\mid a)\); informative-vs-conservative ranking

## Question

On the same X0 host (`simx_hinge.v1`), with learner nominal frozen at
\(b_0=0.10\), does a known plant damping intervention appear as

\[
r_{\mathrm{oracle}}=-\Delta b\,\dot q,\qquad \Delta b=b_{\mathrm{true}}-b_0?
\]

## Isolation

| object | value |
|---|---|
| `true_plant_params.damping` | \(\{0.05,0.10,0.15\}\) |
| `learner_nominal_params.damping` | \(0.10\) always |
| host | unchanged kinematic / drive from X0 |
| drive | `qfrc_applied` only |

Learner never reads plant `dof_damping`. `raw_truth/qfrc_passive`
audit-only.

## Matrix

\[
3\text{ seeds}\times 3\text{ trajectories}\times 3\text{ dampings}=27\text{ cells}.
\]

## Gates

| gate | metric | result |
|---|---|---|
| G0 nominal guard | max \(\mathrm{NRMSE}_{\tau}\) at \(b_{\mathrm{true}}=0.10\) | \(9.56\times 10^{-17}\) |
| G1 oracle reconstruction | max \(E_{\mathrm{oracle}}\) mismatch | \(8.18\times 10^{-16}\) |
| G2 signed coefficient | max \(\lvert\hat s-s^*\rvert/\lvert s^*\rvert\) | \(5.55\times 10^{-16}\) |
| G3 no spurious intercept | max \(\lvert\hat c\rvert/\mathrm{RMS}(\tau)\) | \(1.58\times 10^{-16}\) |

All \(<10^{-4}\). `unlocks_r10_c0=false`. `unlocks_sim_x2=true`.

Artifacts: `runs/sim_x1/formal/`.

## Claims allowed

\[
\boxed{
\text{known damping intervention on the MuJoCo host is correctly
manifested as generalized-force mismatch under frozen nominal}
}
\]

\[
\boxed{r=-\Delta b\,\dot q}
\]

on this independent engine.

## Claims forbidden

validity observability; policy-induced blindness; action ranking;
certificate; real physics. Those are X2+.

## Next

**SIM-X2**: action-conditioned information on this same host / damping
family. Do not interpret \(|r|=|\Delta b|\,|\dot q|\) as blindness yet.
