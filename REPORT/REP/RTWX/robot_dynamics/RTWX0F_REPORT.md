# RTWX-X0F Report — Force-Channel / Clock Audit

Date: 2026-08-27
Status: **RUN COMPLETE**; **`rtwx_x0f_passed=false`**;
pattern **`force_channel_unresolved`**
Prereg: `REPORT/REG/RTWX/RTWX0F_PREREG.md`
Artifacts: `runs/rtwx_x0f/{header.json,run.header.txt,summary.json,logs.npz,grid_train.json,run.log}`
Does not: nets; \(\phi\) fit; capacity; TASK-XL; RGB; R10

## One-line

Every logged generalized-force candidate was crossed with
\(\{qacc^{pre},qacc^{post},\Delta\dot q/\Delta t\}\) and
\(\ell\in\{-2,\ldots,+2\}\). **No** \((k,\ell)\) reached
\(\min_i|\mathrm{corr}|\ge 0.3\) on train **and** the same triple on
held-out. Best train: \(\tau_{\mathrm{cmd}}\) vs \(qacc^{post}\),
\(\ell=0\), \(\min|\rho|=0.173\). Same triple held-out \(0.094\).
Test-best (not the frozen G0 pick) is \(0.267\) (\(\tau\) vs
\(\ddot q_{fd}\), \(\ell=+1\)) — still below \(0.3\).

`ext_api=false` (no `compute_generalized_external_force` on this
PhysxArticulation). `qf_post_set` / `qf_post` match `tau_cmd` in the
grid (same corr). Passive/ext add nothing.

Numpy plant: **`channel_reproduced`** (gates not vacuous).

**Physics-capacity claims are not licensed** on this RoboTwin/SAPIEN
instrumentation until a real integrator-force channel is exposed.

G1 oracle closure was **not** run (G0 failed). Order remains
X0F → oracle closure → \(\phi\)-ID → capacity.

## Ledger

```text
RTWX-X0S = RAN / timing_mismatch
RTWX-X0F = RAN / force_channel_unresolved
TASK-XL  = DESIGN FROZEN ONLY; LOCKED
R10      = LOCKED
```
