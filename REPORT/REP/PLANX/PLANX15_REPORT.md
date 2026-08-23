# PLAN-X1.5 Report — Learned Action Density vs Mean+Isotropic

Date: 2026-08-21  
Status: **`plan_x15_passed=false`**; pattern **`iso_sufficient`**  
Prereg: `REPORT/REG/PLANX/PLANX15_PREREG.md` (gates **not** retuned)  
Data: `runs/plan_x0/formal/` (2048 test conditions)  
Artifacts: `runs/plan_x15/formal/`  
Does not: revive \(H_J^{-1}\); diffusion; PLAN-X2; unlock R10;
retune PLAN-X1 gates; CAP-X2 rescue

## Question

At the same rollout grid \(B\), does a learned density \(q_\phi(A\mid c)\)
beat X1’s winning proposal \(\mathcal N(\mu_\phi,\sigma_0^2 I)\)?

Supervise teacher \(A^\star\) only. **Never** use \(h\).

## Pattern (pre-declared)

\[
\boxed{\texttt{iso\_sufficient}}
\]

Mix-4’s \(AUC_S\) point estimate is slightly above Iso, but the 95%
paired CI **includes 0**. Learned diagonal variance is indistinguishable
from Iso. Mean + fixed isotropic width is enough on this host.

## Layer 1 — H1 still holds

| contrast | result |
|---|---|
| \(J(\mu_{\mathrm{iso}})\) vs \(J(0)\) | \(451\) vs \(755\); \(\Delta=304\), CI \((293,316)>0\) → **G-H1 PASS** |
| \(AUC_S\) Iso − broad | \(0.322-0.182=+0.141\), CI \((0.120,0.161)>0\) → **G-iso PASS** |

Same story as PLAN-X1: the learned basin center is real.

## Layer 2 — density vs Iso **core, fails**

Same \(B\in\{16,32,64,128,256\}\), 2048 conditions:

| sampler | \(AUC_S\) |
|---|---:|
| broad \(\mathcal N(0,0.75^2 I)\) | 0.182 |
| **Iso** \(\mu+\sigma_0^2 I\) | **0.322** |
| Diag learned \(\sigma_j\) | 0.322 |
| Mix-4 | 0.335 |

| contrast | \(\Delta AUC_S\) | 95% CI |
|---|---:|---|
| Mix − Iso (**G-mix**) | \(+0.012\) | \((-0.004,+0.027)\) includes 0 → **FAIL** |
| Diag − Iso | \(-0.0004\) | \((-0.014,+0.014)\) includes 0 |

Success vs budget (not a gate; logged):

| \(B\) | broad | Iso | Diag | Mix |
|---:|---:|---:|---:|---:|
| 16 | 0.045 | **0.209** | 0.159 | 0.156 |
| 32 | 0.085 | **0.272** | 0.243 | 0.245 |
| 64 | 0.161 | 0.324 | 0.320 | **0.345** |
| 128 | 0.261 | 0.375 | 0.403 | **0.419** |
| 256 | 0.394 | 0.427 | 0.482 | **0.502** |

Iso is better at **small** \(B\); Mix/Diag catch up at **large** \(B\).
The primary \(AUC_{\log B}\) therefore stays inside noise. Do **not**
redefine the gate as “\(S(256)\)” after seeing this table.

## CEM (secondary, not required for PASS)

| seed | CEM \(AUC_S\) over \(\{128,256,512,1024\}\) |
|---|---:|
| vanilla broad | 0.027 |
| Iso-seed | **0.097** |
| Mix-seed | 0.092 |

Mix − Iso CEM: \(\Delta=-0.005\), CI \((-0.013,+0.004)\) includes 0
→ **G-cem FAIL**. Warm-start credit remains **H1 \(\mu\)**, not the
learned mixture.

## What this does **not** mean

- Not “mixture capacity failed, go to diffusion.” Mix was not worse;
  it was **not significantly better**. Diffusion is a **new** cell if
  someone wants a stronger density class; it is **not** unlocked by
  this FAIL.
- Not PLAN-X2 (`proposal_failure` still did not occur).
- Not a resurrection of H2. Density learning without Hessian also
  failed to beat Iso on the frozen AUC.

## Claims ceiling

**Allowed:** on this oracle reachable-target host, mean + isotropic
\(\sigma_0=0.25\) is a sufficient action proposal; Mix-4 / learned
diag do not pass the preregistered \(AUC_S\) gate.

**Forbidden:** flipping G-mix by looking only at \(B=256\); calling
this a diffusion result; mixing into CAP \(R_P\); unlocking R10.

## Architecture freeze (this host)

\[
\boxed{
\textbf{学在哪里搜有效；把分布形状学得更复杂没有额外显著价值。}
}
\]

\[
q(A\mid c)\approx\mathcal N(\mu_\phi(c),\sigma_0^2 I)
\]

Do not promote Mix-4 failure into diffusion. A later diffusion cell
requires a **declared multimodal** task family.

## Ledger

```text
PLAN-X0      = PASS
PLAN-X1      = FAIL  anisotropy_no_value (H1 yes, H2 no)
PLAN-X1.5    = FAIL  iso_sufficient (density ≉ better than mean+iso)
PLAN-X2      = LOCKED (not triggered)
CAP-X3-P0    = RUNNING
CAP-X3       = FROZEN (blocked on P0)
R10          = LOCKED
```
