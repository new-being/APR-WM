# RoboTwin-X0-P0 Preregistration — Drawer Instrument Preflight

Date: 2026-08-23  
Status: **FROZEN**; **PASS** (`rtwx_x0_p0_passed=true`; official task false)  
Depends on: `REPORT/REG/RTWX/RTWX0_PREREG.md`  
Does not: official 3-task formal; RGB; asset download; Efficient-WAM

## One question

\[
\boxed{
\text{Can we close oracle-state extraction, parameter-split data, a
1-DoF physics drawer, and a pure latent MLP — without RGB — on this
machine?}
}
\]

## Host (P0 only)

Official RoboTwin `assets/objects` / `embodiments` are **absent**. P0
therefore uses an inline SAPIEN host:

\[
\boxed{\texttt{rtwx\_drawer1.v1}}
\]

1-DoF **prismatic drawer** (the user’s Task-B physics object, not a
renamed laptop hinge). Action = applied force. State \(s=(q,\dot q)\).
Scene parameters \(\phi=(m,b,\mu)\).

This is **not** `put_object_cabinet`. G-official logs False.

## Protocol

- Train \(\phi\): \(m\sim U[0.8,1.2]\), \(\mu\sim U[0.2,0.4]\), \(b\)
  in a frozen band.
- Test composition: \(m=1.5\), \(\mu=0.5\).
- Random force excitation; 40 train / 16 test episodes × 80 steps (P0
  scale, not 1000).
- Physics-only **accounting**: \(F_{\mathrm{phy}}(\cdot;\phi_{\mathrm{true}})\)
  on held-out episodes (closure). A **single** \(\phi\) fit on the train
  box then evaluated at the composition test point is logged only
  (expected to be worse — that is CAP-X3’s per-scene \(\theta\), not a
  P0 fail).
- Latent: MLP \((s,a)\to s'\), hidden 64, no encoder bottleneck required
  in P0 (full \(s\)).
- Seeds: `{501}`. Device: CUDA if present else CPU.

## Gates

| id | requirement |
|---|---|
| **G-sapien** | `sapien` imports in the RoboTwin env |
| **G-norgb** | no camera tensors in \(s\) |
| **G-ident** | median \(\lvert\Delta q\rvert\) at test \(\phi\) vs train-nominal \(\phi\) \(> 5\times\) same-action noise floor |
| **G-phy** | \(F_{\mathrm{phy}}(\cdot;\phi_{\mathrm{true}})\) on test \(E_1\le 0.10\) (accounting; **not** a single global \(\phi\) fit on train then OOD) |
| **G-latent** | trained latent test \(E_1 < 0.80\) (sanity, not capacity) |
| **G-split** | train \(\phi\) box disjoint from test point |
| **G-official** | `official_robotwin_task_executed=false` |
| **G-label** | `runs/rtwx_x0/p0/`; no RGB; no \(R_P\) claim |

`rtwx_x0_p0_passed` iff G-sapien \(\land\) G-norgb \(\land\) G-ident
\(\land\) G-phy \(\land\) G-latent \(\land\) G-split \(\land\) G-label.

If G-official is False (expected): P0 may still PASS the instrument.
Formal 3-task RoboTwin remains locked until assets exist **and** a new
smoke on `put_object_cabinet`.

## Claims ceiling

P0 may say: drawer physics + latent MLP + composition split is closed
on SAPIEN without RGB.

P0 may **not** say: physics reduces capacity on RoboTwin tasks.
