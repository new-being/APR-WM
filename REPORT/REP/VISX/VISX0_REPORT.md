# VIS-X0 Report — GT Seg + Depth Interface Smoke

Date: 2026-08-17  
Status: **`vis_x0_passed=true`**; unlocks **VIS-X1 only**  
Prereg: `REPORT/REG/VISX/VISX0_PREREG.md`  
Artifacts: `runs/vis_x0/formal/`  
Does not: claim real perception; unlock R10-C0; open VIS-X2

## Question

Can simulator GT segmentation + depth recover \(\hat q\) on the SIM-X
hinge so the force-residual interface still closes under matched
nominal physics?

## Setup

| item | value |
|---|---|
| host | `simx_hinge_vis.v1` (dynamics sibling of `simx_hinge.v1`) |
| camera | fixed `fixed_cam`, 320×240, fovy 45° |
| perception | GT geom seg + depth → pivot-centered PCA in XY |
| \(\hat{\dot q}\) | local-linear slope on flip-aware unwrapped \(\hat q\) |
| residual visual | \(M(\hat q)\ddot q_{\mathrm{plant}}+b(\hat q,\hat{\dot q})-\tau-\mathrm{passive}(\hat{\dot q};b_0)\) |
| drive | PD tracks bounded refs (sine/chirp/piecewise); oracle state used **only** for \(u\) |
| cells | 3 seeds × 3 trajectories × 4 s @ 2 ms |

## Floors (frozen this campaign)

| gate | floor |
|---|---|
| G-phys | \(\mathrm{NRMSE}(r_{\mathrm{oracle}})<10^{-4}\) |
| G-geom | \(\mathrm{RMSE}(\hat q-q)<0.05\) (wrapped) |
| G-res | \(\mathrm{NRMSE}(r_{\mathrm{visual}})<0.10\) |
| G-label | `runs/vis_x0/`, `source=simulator`, `real_perception=false` |

## Results

| metric | value |
|---|---|
| `vis_x0_passed` | **true** |
| max \(\mathrm{NRMSE}_{\mathrm{oracle}}\) | \(\sim10^{-16}\) |
| max \(\mathrm{RMSE}(\hat q)\) | \(0.049\) |
| max \(\mathrm{NRMSE}_{\mathrm{visual}}\) | \(0.031\) |
| gates | G-phys / G-geom / G-res / G-label all true |

Sine/chirp recover \(\hat q\) at \(\mathrm{RMSE}\sim0.008\text{–}0.013\).
Piecewise sits near the geom floor (\(\sim0.035\text{–}0.049\)) because
discontinuous references excite higher \(\hat{\dot q}\) error; residual
NRMSE remains \(\ll0.10\) because \(\tau\) magnitude is large under PD.

## Interpretation

- Physics host accounting still closes → visual assets did not break
  the SIM-X force identity.
- GT seg+depth → residual plumbing works.
- Simulator segmentation is **not** real perception; VIS-X0 is an
  interface smoke only.
- `r_pseudo = r_visual - r_oracle` is logged for VIS-X1 comparison.

## Unlock

**VIS-X1** (drop oracle segmentation; RGB-D → state) may open.  
R10-C0 remains locked. SIM-X3 / REAL-LOG-A1 stay closed.
