# CAP-X2 Report — Structural-Mismatch Capacity Decay

Date: 2026-08-18  
Status: **`cap_x2_passed=true`**; pattern **`reference_failure`**  
Prereg: `REPORT/REG/CAPX/CAPX2_PREREG.md`  
P0: `REPORT/REP/CAPX/CAPX2_P0_REPORT.md` (planning **disabled**)  
Artifacts: `runs/cap_x2/formal/`  
Claim scope: **predictive dynamics only** (\(M=M_1\land M_2\))

Does not: planning-equivalent replacement; expand \(H>256\); clip
\(R_P\); unlock R10; retune \(\rho\) grid or residual family.

## Question

\[
\boxed{
\text{As }\rho=\mathrm{RMS}(\tau_\perp)/\mathrm{RMS}(\tau_{\mathrm{phy}})
\uparrow,\text{ how does }R_P(\rho)\text{ decay?}
}
\]

## Primary result

\[
\boxed{
R_P(0)=1.0
\quad\text{(only competent point)}
}
\]

\[
\boxed{
\rho\ge 0.05:\quad
E_{1,\mathrm{ref}}(\rho)>0.25
\quad\Rightarrow\quad
R_P(\rho)=\mathrm{undefined}
}
\]

**Not** \(\rho=0.05\Rightarrow\) physics capacity advantage vanished:
there is no competent PureNN denominator off-family. Accurate claim:

\[
\boxed{
\text{PureNN }H\le256\text{ is insufficient as an off-family
capacity-replacement ruler.}
}
\]

Pattern **`reference_failure`**. Decay shape / Spearman cannot be
measured. \(\rho_{50}=0\) only as the sup over the single competent
point. **Do not add \(H=512\).** Why 5% RMS residual collapses PureNN
generalization is a **future cell**, not an X2 amendment.

What CAP-X currently holds:

\[
\boxed{
\text{matched-family upper bound established;}
\text{ off-family replacement decay not yet measured.}
}
\]

## Curve

| \(\rho\) | \(E_{1,\mathrm{ref}}\) | competent | \(R_P\) | \(P_{\mathrm{pure}}^{\min}\) | \(P_{\mathrm{hybrid}}^{\min}\) |
|---:|---:|:---:|---:|---:|---:|
| 0 | 0.091 | yes | **1.0** | 9987 (\(H=64\)) | 0 |
| 0.05 | 1.031 | no | undefined | — | — |
| 0.10 | 0.913 | no | undefined | — | — |
| 0.25 | 0.850 | no | undefined | — | — |
| 0.50 | 111.2 | no | undefined | — | — |
| 1.00 | 0.750 | no | undefined | — | — |

At \(\rho=0\), Physics-only still matches the per-\(\rho\) PureNN
\(H=256\) reference (\(E_1\sim10^{-10}\), rollout \(0.453\) vs ref
\(0.474\)). That left endpoint agrees with CAP-X1’s matched-family
upper bound (X2’s \(P_{\mathrm{pure}}^{\min}=9987\) is vs this cell’s
own \(H=256\) reference, not a rewrite of X1).

For \(\rho\ge0.05\), PureNN \(H=256\) one-step NRMSE exceeds \(0.25\);
open-loop rollouts often explode (\(E_{\mathrm{roll}}\) huge / NaN).
Physics-only \(E_1\) is \(10^4\)-scale (unmodeled \(\tau_\perp\) not in
the library). Residual Hybrid does not restore a competent PureNN
reference.

## Protocol notes (not retunes)

- Planning disabled by P0; no CEM in this claim.
- Residual family, \(\rho\) grid, widths, competence gate unchanged
  after seeing the curve.
- Runtime: parameter \(\neq\) wall-clock remains a limitation (not
  re-defined as MAC savings).

## Gates

All instrument gates PASS (G-P0 through G-label). Scientific pattern
is **D**, not a protocol fail.

## Claims ceiling

**Allowed:** under this outside-library \(\tau_\perp\) and PureNN
\(H\le256\), explicit structure fully replaces learned dynamics
parameters only at \(\rho=0\); off-family, the PureNN reference is
incompetent so replacement is undefined.

**Forbidden:** “\(R_P\) decays smoothly to X”; real-world 100%;
planning claim; Coulomb; R10; width rescue.

## Unlock

\[
\text{CAP-X2 formal complete (pattern D)}
\;\Rightarrow\;
\text{CAP-X3 remains locked until explicit prereg}
\]

A *new* cell would be required to change residual magnitude
calibration, horizon, or reference architecture — not a silent X2
amendment.
