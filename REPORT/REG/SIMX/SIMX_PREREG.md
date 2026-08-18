# SIM-X Preregistration — Cross-Engine Shadow Validation

Date: 2026-08-17  
Status: **STOPPED** (family); **SIM-X0/X1/X2 PASS**; **SIM-X3 FAIL**
(`false_revoke`); stop doc `REPORT/REP/SIMX/SIMX_FAMILY_STOP.md`;
**does not count as R10**; does not unlock R10-C0

\[
\boxed{
\text{R10-C0 = locked/deferred}
\qquad
\text{SIM-X = STOP after X3}
}
\]

Do not mix the two ledgers. No dwell rescue cell.

Depends on: `REPORT/REP/R9/R9_TOY_FAMILY_STOP.md`,
`REPORT/REG/R10/R10_PREREG.md`  
Does not: call MuJoCo / PhysX / Isaac “real physics”; lower R10-C0;
delete H1 contracts; buy hardware; RoboCasa / RoboTwin as first cell;
DROID / OXE as C0; adaptive B1; new toy \(F_{\max}\) waveforms;
dwell rescue of SIM-X3; claim stale-license lifecycle blindness

## Two meanings of “physics”

APR-WM already uses **explicit physics** for the *model*: the
structured force / dynamics terms the expert writes
(\(I\ddot q+\tau_f+\cdots\)). That is not hardware.

**R10 real physics** means the **closed-loop dynamics of a real
mechanical system**: true mass, friction, compliance, actuator delay,
sensor noise, and unmodeled effects on a physical plant, observed
through a contracted sensor chain.

Without that plant, R10-C0 **cannot** be executed. Simulators remain
simulators:

| stack | backend | role now |
|---|---|---|
| ManiSkill | SAPIEN / PhysX | existing pipeline; small incremental evidence |
| robosuite | **MuJoCo** | **first independent engine** |
| RoboCasa | MuJoCo | after a minimal MuJoCo cell |
| RoboTwin | simulation / data gen | later expansion |
| OmniGibson / BEHAVIOR | Isaac Sim → PhysX | later; **not** a second engine vs ManiSkill |
| RLBench | task/obs | not physics-main |
| LIBERO / MimicGen | demo / prior | not validity-physics main |
| DROID / Bridge / OXE / AgiBot | real *logs* | optional shadow **audit**; **not** C0 |

\[
\boxed{\text{simulation}\neq\text{real physics}.}
\]

\[
\boxed{\text{real data}\neq\text{real controllable experiment}.}
\]

Public real datasets can test whether a representation / predictor
blows up on real interaction *distributions*. They do not, by default,
supply the C0 tuple \((q,\dot q,\ddot q,\tau_{\mathrm{meas}},
\tau_{\mathrm{nominal}},t)\) with a frozen timebase and a
**controlled** intervention (\(F_{\max}\), payload, damping). Audit
per dataset if ever used; never rename a demo log as
`source=hardware` C0.

## Why SIM-X (not H1 CAD, not a robot purchase)

R10-C0 stays **LOCKED** at the same bar
(`REPORT/REG/R10/R10_C0_PREREG.md`). H1 CAD0 remains a **deferred**
hardware contract (`REPORT/REG/R10/R10_C0_H1_CAD0.md`).

The missing cheap falsification is:

\[
\boxed{\text{PhysX/SAPIEN}\;\to\;\text{MuJoCo}}
\]

The family *later* tests the frozen R7–R9 mechanism on a second
engine. **X0 is not that test.** X0 is only force-space accounting
on the host plant, so X2 cannot be adapter error.

## What SIM-X is allowed to claim

\[
\boxed{\text{cross-engine mechanism robustness}}
\]

**Not** real-physics validation. Simulator evidence only. Shadow
commit only. A SIM-X pass never writes R10-C0.

## Status (frozen)

```text
R9 toy-family      = STOP
SIM-X0             = PASS
SIM-X1             = PASS
SIM-X2             = PASS
SIM-X3             = FAIL (false_revoke)
SIM-X lifecycle    = NOT established
SIM-X family       = STOP after X3

cross-engine action-conditioned observability = supported
cross-engine evidence-rate ordering           = supported
cross-engine persistent blindness             = not supported

R10-C0             = LOCKED / DEFERRED
real physics       = not claimed
```

Stop doc: `REPORT/REP/SIMX/SIMX_FAMILY_STOP.md`.

Language correction:

\[
\boxed{
\textbf{policy-induced epistemic attenuation}
\text{ is more robust than }
\textbf{policy-induced epistemic closure}.
}
\]

## Cells (in order)

1. **SIM-X0 — Nominal force accounting**  
   `REPORT/REG/SIMX/SIMX0_PREREG.md`,
   `REPORT/REG/SIMX/SIMX0_FORCE_ACCOUNTING.md`.  
   One question: does frozen nominal accounting put \(r\) on the
   numerical floor? No detector, validity, certificate, or
   \(I(\mathcal V;Y\mid a)\). Not a full APR-WM port.

2. **SIM-X1 — Oracle damping mismatch**  
   `REPORT/REG/SIMX/SIMX1_PREREG.md`,
   `REPORT/REP/SIMX/SIMX1_REPORT.md`.  
   \(b_{\mathrm{true}}\in\{0.05,0.10,0.15\}\), \(b_{\mathrm{learner}}=0.10\)
   frozen. PASS.

3. **SIM-X2 — Action-conditioned observability**  
   `REPORT/REG/SIMX/SIMX2_PREREG.md`,
   `REPORT/REP/SIMX/SIMX2_REPORT.md`.  
   Analytic per-step \(I_a\) under \(\sigma_{\mathrm{obs}}=0.01\). PASS.

4. **SIM-X3 — Shadow lifecycle**  
   `REPORT/REG/SIMX/SIMX3_PREREG.md`,
   `REPORT/REP/SIMX/SIMX3_REPORT.md`.  
   Canonical license + 3-hypothesis Bayes. **FAIL** / `false_revoke`.
   Channel→lifecycle robustness **not** claimed.

X1–X3 **must not** be recorded as R10-C1–C3.

## Family closed

**SIM-X0–X2 PASS; SIM-X3 FAIL; family STOP.**
No dwell rescue. Do **not** open RoboCasa / R10 to “fix” X3.

## Route (frozen)

\[
\boxed{
\begin{array}{c}
\text{R9 = STOP}\\
\downarrow\\
\textbf{SIM-X STOP after X3}\\
\text{(channel yes; lifecycle blindness no)}\\
\downarrow\\
\textbf{R10-C0 remains LOCKED}\\
\downarrow\\
\text{future hardware when available}
\end{array}
}
\]

Dwell, if ever, is a **new** question about permission semantics — not
“make SIM-X3 pass.”
