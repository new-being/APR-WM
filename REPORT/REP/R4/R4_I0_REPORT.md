# R4-I0 Report — Contact-Regime Closure

Date: 2026-08-17  
Prereg: `REPORT/REG/R4/R4_I0_PREREG.md`  
Artifacts: `runs/r4_i0/closure/`  
Scientific result: **none** (infrastructure)

v2 (stiffness swap): `REPORT/REP/R4/R4_I0_V2_REPORT.md` — `R4_I0_PASS=true`.
This file is the **v1 \(\mu\)-mismatch** no-go and stays in the ledger.

\[
\boxed{R4\_I0\_PASS=\mathrm{false}}
\]

R4-C0 formal stays locked. R4-I1 is not unlocked.

This is the intended use of preflight: the first grasp–block protocol
**does not yet** realize

\[
\text{stick vs incipient slip overlapping on }h^{S}
\text{ but split on }(p,\tau_x,\tau_y).
\]

## What did close

Native MuJoCo (not Door): sliding pad taxel grid + world-fixed pad +
cuboid on a 1-D tangential slide. Local maps \(p,\tau_x,\tau_y\) are
finite and non-zero under load.

On the low-\(\mu\) side, **macro regimes are ordered**:

stick \(\to\) incipient (creep) \(\to\) gross slip.

So the simulator can produce the three contact regimes. That is
necessary, not sufficient.

## What failed

Hidden variable was scalar \(\mu\) (C0 high, C1 low), same \(F_t(t)\)
ramp.

Elliptic-cone contact **couples \(\mu\) into normal force**. Example
from this run: \(\mathbb E[F_n]^{\mathrm{C0}}\approx 2.24\) vs
\(\mathbb E[F_n]^{\mathrm{C1}}\approx 7.96\). After requiring wrench
overlap (\(\Delta F_n/F_n<0.25\) and \(\Delta F_t/F_t<0.25\)) **zero**
matched counterexample frames remain.

A linear probe on unfiltered “same-time” windows previously reached
\(\mathrm{AUROC}(h^{S})\approx 0.99\) — exactly the leak I1 is designed
to catch. That audit was not counted as I1 (I0 did not pass).

High-\(\mu\) C0 also failed “never gross / stay stick” after a stronger
squeeze retune.

## What this does not authorize

- lowering I1 AUROC gates to force a pass
- training a tactile encoder
- R4-C0 formal
- going back to Door V7E

## Next protocol (still I0, not C0)

Do **not** hide the label in \(\mu\) if \(\mu\) leaks through wrist
\(F_n\). Next I0 attempts should keep **matched \(\mu\) and matched
preload**, and hide a local contact property that wrench integrates
away, e.g. patch support / \(k_t\) spatial pattern, while checking
wrench-overlap frames before any probe.
