# APR-WM Synthesis — R7–R9 + SIM-X (+ REAL-LOG-A0)

Date: 2026-08-17  
Status: **PAPER-GRADE FREEZE**  
Scope: positive certification → validity lifetime → cross-engine
attenuation → real-physics **unmade**  
Does not: unlock R10-C0; reopen R9/SIM-X/dwell rescue; claim hardware
validation; start a third simulator family

## Stage boundary (not a scientific no-go)

\[
\boxed{
\text{R9 STOP}
\rightarrow
\text{SIM-X STOP}
\rightarrow
\text{REAL-LOG-A0 LIMITED}
\rightarrow
\text{R10-C0 LOCKED}
}
\]

\[
\boxed{
\text{insufficient data source for real-force residual}
\quad\neq\quad
\text{algorithmic falsification of APR-WM}
}
\]

---

## Core mechanism sentence (paper)

\[
\boxed{
\textbf{Control policy changes not only task outcomes, but also the
rate at which a world model can validate its own assumptions.}
}
\]

## Core boundary sentence (paper)

\[
\boxed{
\textbf{Policy-induced epistemic attenuation is cross-engine robust;
persistent epistemic closure is not established.}
}
\]

Prefer **attenuation** over blanket **blindness** in cross-engine claims.

---

## Four-layer claim ledger

Each block: claim → support → failure boundary → allowed / forbidden
wording.

### L1 — Positive certification is feasible; validity is conditional

**Claim.** An alternative-specific certificate can authorize a switch
from a default/robust action without collapsing into eternal abstention;
predictor capacity and certificate permission are separable.

**Supporting experiments.**

| cell | role |
|---|---|
| R7-P0 / A0 | certificate object + conformal \(L\); `GO=true` |
| R7-P1 | saturated plant admitted |
| R7-A1 / A2 | robust / recall recovery without safety loss |
| R7-B0 | WM-mediated positive certification; `GO=true` |
| R7-B1 | shift validity issues (`shift_valid=false`) — validity is conditional |

**Failure boundary.** Certificate success on one fresh family does not
imply validity under every shift; B1 marks conditional validity, not
“certification solved.”

**Allowed wording.**

- positive certification / alternative-specific permission
- abstention ≠ the only safe policy
- predictor vs permission can be separated

**Forbidden wording.**

- universal validity of the WM
- “R7 proves safe autonomous revision in the real world”

---

### L2 — Validity information has a lifetime; normal policy may not observe it

**Claim.** Validity evidence can be amortized; a persistent belief /
license can pay back; but under some policies the applicability channel
closes, so lifetime is not generally inferable from normal operation.
Finite certificate age can buy safety without being NetVoI-optimal.

**Supporting experiments.**

| cell | role |
|---|---|
| R9-P0 | amortization; \(K_{\min}^{\mathrm{realized}}=2\) |
| R9-A0 | persistent \(b(\mathcal V)\); \(\Delta K=0\); `GO=true` |
| R9-B0 | policy-induced epistemic blindness on the **toy**; `GO=false` on passive revoke |
| R9-B1-P0 | safety-effective age bound ≠ expected-utility optimality; `PASS=false` |
| R9 family STOP | three frozen toy conclusions |

**Failure boundary.** R9 blindness is a **toy / saturated-plant**
mechanism. Do not export “permanent closure under cons” as
cross-engine fact (see L3). B1-P0 NetVoI failure must not be rewritten
as constrained-control success.

**Allowed wording.**

- validity information can be amortized
- lifetime need not be visible under licensed conservative operation
  (**on the toy**)
- safety effectiveness ≠ expected utility optimality

**Forbidden wording.**

- “R9 proves real robots cannot observe validity”
- salvaging B1-P0 as an optimality win
- adaptive B1 / \(M\)-sweep as the next scientific step

---

### L3 — Cross-engine: attenuation yes; persistent closure no

**Claim.** On an independent MuJoCo host (`simx_hinge.v1`), action
changes mismatch exposure and the **rate** of validity information;
policy-induced **epistemic attenuation** is robust. Strong
**persistent epistemic closure** / stale-license lifecycle blindness is
**not** established.

**Supporting experiments.**

| cell | result |
|---|---|
| SIM-X0 | force-space adapter closure; `PASS` |
| SIM-X1 | \(r=-\Delta b\,\dot q\); `PASS` |
| SIM-X2 | \(I_{\mathrm{info}}=0.265\), \(I_{\mathrm{cons}}=0.0039\), \(I_{\mathrm{high\_f}}\sim 10^{-5}\) bit/sample; iso-energy; `PASS` |
| SIM-X3 | `FAIL` / `false_revoke`; G4 high_f still revokes (~1 s median) |
| SIM-X family STOP | channel yes; lifecycle blindness no |

**Mechanism chain (supported).**

\[
a \to E_v \to |r| \to I(\mathcal V;Y\mid a)
\]

**Secondary diagnostic (not PASS).** Revoke-rate / \(S_{\mathrm{stale}}\)
order tracks \(I_a\), but weak channels still identify via time.

**Failure boundary.**

- G1: existential \(b_t<0.5\) is fragile under noise (brief dips).
- G4: near-zero \(I_a\) \(\not\Rightarrow\) lifecycle blindness.
- Dwell rescue would conflate permission semantics with manufactured
  persistence — **not opened**.

**Allowed wording.**

- cross-engine action-conditioned validity observability
- policy-dependent evidence-acquisition **rate**
- **policy-induced epistemic attenuation**
- weak evidence × enough observations → eventual identification

**Forbidden wording.**

- cross-engine persistent / stale-license lifecycle blindness
- “SIM-X3 proves blindness” / “SIM-X3 PASS”
- equating R9-B0 toy closure with MuJoCo lifecycle closure
- real-physics validation via MuJoCo

---

### L4 — Real-physics validation remains explicitly unmade

**Claim.** Public real-robot logs audited so far do not supply the
contract residual chain; R10-C0 stays locked until hardware (or a future
force-auditable release).

**Supporting experiments.**

| cell | result |
|---|---|
| REAL-LOG-A0 | `A0-LIMITED`: official DROID lacks \(\tau\); AgiBot effort/wrench unavailable; no current→torque cal |
| R10-C0 | locked/deferred; simulators ≠ hardware residual.h5 |
| H1/CAD0 | deferred contracts only |

**Failure boundary.** “Real robot dataset” ≠ calibrated
\(\tau_{\mathrm{meas}}+t+q+\dot q+u\). Third-party remasters (e.g.
Cosmos3-DROID torque fields) do not upgrade official DROID without a
separate provenance cell.

**Allowed wording.**

- real-physics claim remains **explicitly unmade**
- bottleneck is **data source / hardware**, not missing algorithm cells
- public logs may still serve representation / action-distribution audits

**Forbidden wording.**

- REAL-LOG / DROID / AgiBot as R10-C0
- \(I(\mathcal V;Y\mid a)\) on logs with unknown \(\mathcal V\)
- inventing \(\tau\) from uncalibrated current

---

## Paper outline map (claims only)

| section theme | primary cells | strength |
|---|---|---|
| Certificate vs abstention | R7-A0…B0 | strong (toy + structured plants) |
| Validity lifetime / amortization | R9-P0/A0 | strong (toy) |
| Policy closes evidence channel | R9-B0 | strong **on toy**; attenuate wording for engines |
| Safety age ≠ NetVoI | R9-B1-P0 | strong split |
| Cross-engine attenuation | SIM-X0–X2 | strong |
| Persistent closure | SIM-X3 | **negative** / not established |
| Real force residual | REAL-LOG-A0 + R10 lock | **unmade** |

---

## Global allowed / forbidden glossary

| prefer | avoid |
|---|---|
| epistemic **attenuation** | permanent **blindness** (outside R9-toy scope) |
| evidence-acquisition **rate** | “validity unobservable forever” |
| cross-engine **channel** robustness | cross-engine **lifecycle** blindness |
| real-physics **unmade** | “validated on real robots” via public demos |
| data-source bottleneck | “APR-WM failed realism” |

---

## What not to do next (frozen)

1. Third **physics** simulator family “for realism.”
2. Dwell rescue of SIM-X3.
3. REAL-LOG-A1 force residual without A0-GO.
4. Lowering R10-C0 because SIM-X or demos exist.
5. Adaptive B1 / R9 toy reopen.
6. Jump to RoboCasa/RLBench before VIS-X0 on `simx_hinge`.

## What to do next

1. **Paper architecture** (this freeze): three-pillar spine in
   `REPORT/REP/PAPER/APRWM_PAPER_ARCHITECTURE.md`. No new cells.
2. Hardware / R10-C0 only when a real force chain exists.
3. Do **not** open CAP-X3B, CAP-X4, PLAN-X2, or diffusion to extend
   CAP-X3. CAP-X2 Pattern D stays a boundary, not a rescue target.

## Pointers

- R9 stop: `REPORT/REP/R9/R9_TOY_FAMILY_STOP.md`
- SIM-X stop: `REPORT/REP/SIMX/SIMX_FAMILY_STOP.md`
- REAL-LOG-A0: `REPORT/REP/REALLOG/REALLOG_A0_REPORT.md`
- R10 lock: `REPORT/REG/R10/R10_PREREG.md`
