# RoboTwin-X0-P0 Report — Drawer Instrument Preflight

Date: 2026-08-23  
Status: **`rtwx_x0_p0_passed=true`**; official RoboTwin task **not** executed  
Prereg: `REPORT/REG/RTWX/RTWX0_P0_PREREG.md`  
Host: `rtwx_drawer1.v1` (inline SAPIEN prismatic drawer)  
Artifacts: `runs/rtwx_x0/p0/`  
Does not: RGB; 3-task formal; capacity \(R_P\); Efficient-WAM; R10

## Result

Instrument **PASS**. Official assets (`objects/`, `embodiments/`) are
**missing**; `G_official=false` as frozen. Formal RoboTwin-X0 stays locked.

| gate | result |
|---|---|
| G-sapien | PASS (RoboTwin conda, SAPIEN 3.0.0b1) |
| G-norgb | PASS \(s=(q,\dot q)\), force action |
| G-ident | PASS median \(\lvert\Delta q\rvert=0.035>5\times 10^{-3}\) |
| G-phy | PASS \(E_1(F_{\mathrm{phy}};\phi_{\mathrm{true}})=0\) (deterministic replay) |
| G-latent | PASS \(E_1=0.386<0.80\) (sanity only) |
| G-split | PASS train box vs test \((m,\mu)=(1.5,0.5)\) |

Diagnostic (not a gate): a **single** \(\phi\) fit on the train box then
evaluated at the composition point has \(E_1=0.24\) (hold \(0.33\)).
That is expected: composition needs **per-scene** \(\phi\), which is the
CAP-X3 lesson, not a P0 fail.

P0 latent is a short MLP on CPU; \(E_1\) is not better than a hold
predictor. Do **not** read a capacity ranking from this cell.

## Unlock

P0 unlocks **code + split protocol**, not the 3-task formal matrix.
Next: install RoboTwin assets, then a documented `put_object_cabinet`
smoke, then freeze formal episode counts.
