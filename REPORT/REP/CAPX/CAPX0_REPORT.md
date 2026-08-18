# CAP-X0 Report — Benchmark / Accounting / Baseline Fairness

Date: 2026-08-17  
Status: **`cap_x0_passed=true`**  
Prereg: `REPORT/REG/CAPX/CAPX0_PREREG.md`  
Host: `capx_arm3.v1`  
Artifacts: `runs/cap_x0/formal/`  
Does not: claim \(R_P\) / \(R_F\); planning; \(\rho\) sweep; unlock R10

## Question

\[
\boxed{
\text{Is the CAP-X host, scene generator, excitation, split, and
oracle physics accounting fair and closed enough to support later
capacity claims?}
}
\]

## Setup

| item | value |
|---|---|
| plant | 3-DoF serial arm, no contact, `qfrc_applied` |
| \(\theta_e\) | \(\mathbb{R}^{13}\) (mass/inertia scales, damping, \(\mu{=}0\) slots, payload) |
| splits | 128 / 32 / 64 scenes (disjoint seeds + fingerprints) |
| traj | 8 × 2.0 s / scene; model rate 100 Hz |
| PureNN | MLP \(H{=}256\) (3×SiLU), input \([q,\dot q,u,\theta_e]\) |

Coulomb \(\mu\) slots are present but frozen at 0 in X0 (MuJoCo
frictionloss timing vs \(-\mu\mathrm{sign}(v)\) broke \(10^{-4}\)
accounting). **CAP-X1 keeps \(\mu=0\)** (schema only); a separate
convention-lock cell is required before any Coulomb capacity claim.

## Gates

| gate | result | detail |
|---|---|---|
| **G0** | PASS | \(\mathrm{NRMSE}(r_{\mathrm{phy}})=1.2\times10^{-15}\) |
| **G1** | PASS | all DoF \(\mathrm{Var}(q,\dot q,\ddot q)\) above floors |
| **G2** | PASS | no \(\theta\)/seed leakage across splits |
| **G3** | PASS | PureNN val \(\mathrm{NRMSE}(\hat{\ddot q})=0.101\le0.25\) |

## PureNN sanity (not a capacity claim)

| item | value |
|---:|
| trainable \(P\) | 138243 |
| train samples | 204800 |
| train NRMSE | 0.026 |
| val NRMSE | **0.101** |
| device | cpu |

## Unlock

\[
\boxed{\text{CAP-X0 PASS}\;\Rightarrow\;\text{CAP-X1 prereg frozen}}
\]

CAP-X1 prereg: `REPORT/REG/CAPX/CAPX1_PREREG.md` (\(\rho=0\), \(\mu=0\),
width grids + matched triple + CEM budget). **No width sweep until
explicit start.** R10 remains locked.
