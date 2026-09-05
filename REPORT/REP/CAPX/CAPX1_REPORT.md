# CAP-X1 Report — Matched-Family Capacity \(R_P(0)\)

Date: 2026-08-18  
Status: **`cap_x1_passed=true`**  
Prereg: `REPORT/REG/CAPX/CAPX1_PREREG.md` (frozen; no post-hoc retune)  
Host: `capx_arm3.v1`  
Artifacts: `runs/cap_x1/formal/`  
Data: `runs/cap_x0/formal/` (\(\rho=0\), \(\mu=0\))

Does not: \(\rho>0\); Coulomb convention lock; unlock R10; claim
real-world 100% replacement.

## Question

\[
\boxed{
\text{Under matched physics }(\rho=0),\text{ how much trainable neural
capacity does explicit structure replace at matched prediction
\textit{and} planning?}
}

## Primary result

\[
\boxed{
R_P(0)=1.0
\quad\text{(matched-family replacement upper bound)}
}
\]

Physics-only (\(H_{\mathrm{hybrid}}=0\), \(P=0\)) matched the PureNN
\(H=256\) reference on the frozen triple gate. Per prereg wording this
is an **upper bound under \(\rho=0\) matched family**, not “physics
replaces 100% of latent capacity in the real world.”

| quantity | value |
|---|---:|
| \(P_{\mathrm{pure}}^{\min}\) | 138243 (\(H=256\) only) |
| \(P_{\mathrm{hybrid}}^{\min}\) | 0 (Physics-only) |
| \(R_P(0)\) | **1.0** |
| claim label | matched-family replacement upper bound |

## Setup (frozen)

| item | value |
|---|---|
| PureNN widths | \(\{8,16,32,64,128,256\}\) |
| Hybrid widths | \(\{0,8,16,32,64,128,256\}\) (\(H=0\) = Physics-only) |
| seeds | \(\{101..105\}\), AdamW / lr \(10^{-3}\) / wd \(10^{-4}\) / batch 512 / 40 ep / patience 8 |
| Hybrid convention | \(\hat{\ddot q}=q_{\mathrm{phy}}+\mathrm{M}^{-1}\tau_{\mathrm{res}}\) |
| CEM | horizon 1.0 s, \(dt=0.01\), knots 20, cand 256, iters 5, elite 0.10 |
| match | \(E_1\le1.05 E_{\mathrm{ref}}\), \(E_{\mathrm{rollout}}\le1.10 E_{\mathrm{rollout,ref}}\), \(S_{\mathrm{plan}}\ge S_{\mathrm{ref}}-0.05\) |
| device | cpu |

## Reference (PureNN \(H=256\), 5-seed test mean)

| \(E_1\) | \(E_{\mathrm{rollout}}\) | \(S_{\mathrm{plan}}\) | step latency | CEM wall |
|---:|---:|---:|---:|---:|
| 0.0831 | 0.460 | 0.0 | \(1.23\times10^{-4}\) s | 0.35 s |

Physics-only test \(E_1=3.5\times10^{-10}\) (oracle accounting).

## Width aggregates (5-seed means)

### PureNN

| \(H\) | \(P\) | \(E_1\) | \(E_{\mathrm{rollout}}\) | \(S_{\mathrm{plan}}\) | matched |
|---:|---:|---:|---:|---:|:---:|
| 8 | 355 | 0.368 | 1.121 | 0.0 | no |
| 16 | 963 | 0.195 | 0.805 | 0.0 | no |
| 32 | 2947 | 0.114 | 0.556 | 0.0 | no |
| 64 | 9987 | 0.092 | 0.477 | 0.0 | no |
| 128 | 36355 | 0.092 | 0.509 | 0.0 | no |
| 256 | 138243 | 0.083 | 0.460 | 0.0 | **yes** |

### Hybrid / Physics-only

| \(H\) | \(P\) | \(E_1\) | \(E_{\mathrm{rollout}}\) | \(S_{\mathrm{plan}}\) | CEM wall | matched |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0 | \(\sim10^{-10}\) | 0.453 | 0.0 | 7.65 s | **yes** |
| 8–256 | 355–138243 | \(\sim10^{-10}\) | 0.453 | 0.0 | ~22–27 s | yes |

Residual nets at \(\rho=0\) add negligible prediction error vs Physics-only
(as expected when the plant lies in the explicit family).

## Planning note (not a retune)

Under the frozen CEM budget, \(S_{\mathrm{plan}}=0\) for **all** widths
(including Physics-only and PureNN reference). The planning match gate is
therefore non-discriminative here; matched status is decided by \(E_1\) and
\(E_{\mathrm{rollout}}\). CEM hyperparameters were **not** changed after
seeing curves. Diagnostic: even PD with train-max \(|u|\) clips only
partially reaches random box targets in 1 s — the frozen sampler is
underpowered for \(\varepsilon_q=0.15\) on this host.

## Gates

| gate | result |
|---|---|
| **G0** | PASS (oracle residual NRMSE \(\lt10^{-4}\), \(\mu=0\)) |
| **G-ref** | PASS |
| **G-grid** | PASS (13 widths × 5 seeds) |
| **G-match** | PASS (\(P_{\mathrm{pure}}^{\min}\), \(P_{\mathrm{hybrid}}^{\min}\) exist) |
| **G-label** | PASS (`rho=0`, `mu=0`, upper-bound wording) |

## Claims ceiling

**Allowed:** at \(\rho=0\) on `capx_arm3.v1`, explicit rigid-body structure
matches PureNN \(H=256\) prediction/rollout with \(P_{\mathrm{hybrid}}=0\),
hence \(R_P(0)=1\) as a **matched-family upper bound**.

**Forbidden / deferred:** real-world 100% replacement; Coulomb; \(\rho>0\)
decay (CAP-X2); R10 unlock; primary FLOPs claim from MLP MACs alone.

## Unlock

\[
\boxed{\text{CAP-X1 PASS}\;\Rightarrow\;\text{CAP-X2 prereg frozen}}
\]

CAP-X2 prereg: `REPORT/REG/CAPX/CAPX2_PREREG.md` (calibrated \(\rho\),
outside-library \(\tau_\perp\), per-\(\rho\) PureNN reference, CAP-X2-P0
planning harness). **No \(\rho\) sweep until explicit start.** R10 remains
locked.
