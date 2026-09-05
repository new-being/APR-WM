# RTWX-X0S Report — Cabinet Dynamics Structure Audit

Date: 2026-08-27
Status: **RUN COMPLETE**; **`rtwx_x0s_passed=false`**; pattern **`timing_mismatch`**
Prereg: `REPORT/REG/RTWX/RTWX0S_PREREG.md`
Artifacts: `runs/rtwx_x0s/{header.json,run.header.txt,summary.json,trajectories.npz,run.log}`
Does not: neural residual; capacity \(R_P\); TASK-XL; RGB; R10

## One-line

Cabinet-only \((q,\dot q,\tau)\to\ddot q\) was audited with nested M0–M3
and **command-only** `set_qf`. **G0 FAIL:** \(\mathrm{corr}(\tau,\ddot q)\approx -0.018\)
(\(<0.15\)). Commanded torque is **not** the same-time force that produces
measured \(\ddot q\). Oracle M2 and LS families all **lose to** identity
\(\hat{\ddot q}=0\) (\(E=1\) vs oracle \(2.57\), best LS M3 \(E=6.05\)).
Not a capacity result. Not “APR-WM failed on RoboTwin.”

## Frozen before collect

`delta_qdd=0.05`, `corr_min=0.15`, `g2_rel_drop=0.02`; 24/12 ep × 60
steps; `apply_mode=command_only`; `dt=0.004` × 8 substeps. Numpy M2
plant (unit test) **PASSes** `structure_closed` — the runner is not
vacuously broken.

## Gates

| gate | result | note |
|---|---|---|
| **G0** same-time | **FAIL** | all 3 DoF \(\mathrm{corr}\in[-0.026,-0.014]\); lag-1 not better |
| **G1** phy \(<\) identity | **FAIL** (not scored after G0) | oracle M2 \(E_{\ddot q}=2.57>1\); LS M3 still \(>1\) |
| **G2** nested RMS | logged | train RMS 24.6 → 14.7 (M0→M3) but on an unaligned \(\tau\) |
| **G3** rollout | skipped | G0/G1 failed |

`d_q=3`. Per-DoF \(\mathrm{rms}(\ddot q)\sim 8\), so motion exists;
\(\tau\) std \(\sim 21\)–\(27\) is not a dead command.

X0S **narrowed** the failure: not “physics family too simple”, but
the two sides of the ODE are **not on the same clock / force channel**.

\[
\operatorname{corr}(\tau_{\mathrm{cmd}},\ddot q)\approx -0.018
\qquad\text{all 3 DoF},\qquad
\mathrm{rms}(\ddot q)\sim 8.
\]

The cabinet **moves**; command is **not** constant; they have **no**
same-time relation. Oracle M2 is worse than \(\hat{\ddot q}=0\). Extra
\(b,\mu,g(q)\) or \(\phi\)-ID would fit a **wrong pairing**. Numpy plant
PASS excluded a broken gate. Next: **RTWX-X0F** force/clock audit
Next: **RTWX-X0F** (`REPORT/REP/RTWX/RTWX0F_REPORT.md`) —
**`force_channel_unresolved`**.


## Ledger

```text
RTWX-X0S = RAN / FAIL  pattern=timing_mismatch
RTWX-X0F = RAN / FAIL  force_channel_unresolved
TASK-XL  = DESIGN FROZEN ONLY; LOCKED
R10      = LOCKED
capacity = not opened
```
