# CAP-X3-P0 Report — Identifiability Preflight

Date: 2026-08-21  
Status: **`cap_x3_p0_passed=true`**; unlocks CAP-X3 formal  
Prereg: `REPORT/REG/CAPX/CAPX3_P0_PREREG.md`  
Artifacts: `runs/cap_x3/p0/`  
Does not: \(K_{90}/R_K\); \(\rho>0\); active probe; R10

## What P0 is for

Not the number \(E_1^{\mathrm{phy}}=0.001\) as an adaptation claim.

P0 only shows that the **formal question is well-posed**:

1. The 10-D active \(\theta_E=(m,I,b,m_p)\) is recoverable from finite
   passive calibration (\(K=16\), \(T=1\,\mathrm{s}\)).
2. A same-dim latent \(z_E\) has a real optimization basin
   (\(E_1(z_{\mathrm{fit}})<E_1(z=0)\)).
3. Formal A vs B is not “unlearnable physics vs learnable latent.”

## Gates

| gate | result |
|---|---|
| G-id-pred | **PASS** median query \(E_1=0.0010\le 0.05\) |
| G-id-param | **PASS** median \(E_\theta=0.062\le 0.35\) (no `identification_slack`) |
| G-latent | **PASS** median \(E_1(z)/E_1(0)=0.55\le 0.80\) |
| \(d_\theta\) | **10** (Coulomb slots remain unused) |

Oracle query \(E_1\sim 10^{-15}\). Nominal-\(\theta\) \(E_1\) is large
(\(\sim 0.4\) on probe scenes): there is something to adapt.

Latent query \(E_1(z_{\mathrm{fit}})\approx 0.21\) is **not** a \(R_K\)
result — different model class, P0 budget, 8 probe scenes.

## Unlock

\[
\text{CAP-X3-P0 PASS}
\;\Rightarrow\;
\text{run CAP-X3 formal }K\in\{1,2,4,8,16\}
\]

Active set is **not** shrunk. Formal may not add probe, \(\rho\), residual
\(z\), or PLAN-X.
