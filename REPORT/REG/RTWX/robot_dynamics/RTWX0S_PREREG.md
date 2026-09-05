# RTWX-X0S Preregistration — Cabinet Dynamics Structure Audit

Date: 2026-08-27
Status: **FROZEN**; **formal RAN 2026-08-27**; **`rtwx_x0s_passed=false`**;
pattern **`timing_mismatch`**. Report: `REPORT/REP/RTWX/RTWX0S_REPORT.md`.
Depends on: RoboTwin-X0R RAN (`excitation_failure`; G1 PASS; G0/G2/G3 FAIL)
  `REPORT/REP/RTWX/RTWX0R_REPORT.md`
Does not: neural residual; capacity \(R_P\); scene-ID as first question;
TASK-XL0; RGB; R10; retune X0R gates after seeing X0S curves

## One question

\[
\boxed{\textbf{RTWX-X0S — Cabinet Dynamics Structure Audit}}
\]

\[
\boxed{
\text{Without a neural residual, what is the smallest explicit dynamics
family that stably beats identity on real RoboTwin cabinet trajectories?}
}
\]

Do **this** before capacity, before \(\phi\)-ID as a scientific claim,
and before TASK-XL.

## Honest bottleneck (after X0R)

Not “did \(\Delta s\) move?” X0R already has \(\Delta s\neq 0\) and
SAPIEN \((q,\dot q,\ddot q,\tau)\). Force **audit** is enough.

The failure is:

\[
\boxed{\textbf{the explicit cabinet dynamics family is structurally wrong.}}
\]

Evidence: \(E_1^{\mathrm{identity}}=0.585 < E_1^{\mathrm{physics}}\approx 2.05\),
and scene \(\hat\phi\) **worse** than hardcoded \(\phi\). That is not
“tune \(\phi\) harder.” A misspecified \(F_{\mathrm{phy}}(\cdot;\phi)\)
only finds a pseudo-true parameter.

Order frozen:

\[
\boxed{
\text{structure closure}
\rightarrow
\text{parameter identification}
\rightarrow
\text{capacity comparison}
}
\]

These FAILs are **not** “APR-WM failed on RoboTwin.”

## Coordinate (phase 1)

Cabinet **active DoF only**. Drop the rest of the RoboTwin observation.

\[
x_t=(q_t,\dot q_t),\qquad
\text{in }(q_t,\dot q_t,\tau_t),\qquad
\text{out }\ddot q_t
\]

or residual

\[
r_\tau=M(q)\ddot q+h(q,\dot q)-\tau.
\]

Question: **does cabinet joint dynamics close?** Not: is the full
\(d_s=27\) vector predictable.

## Nested families (no big net)

Keep a term only if G2 says RMS(\(r_\tau\)) drops.

| id | family |
|---|---|
| **M0** | \(\tau=I\ddot q\) |
| **M1** | \(+\,b\dot q\) |
| **M2** | \(+\,\mu\mathrm{sign}(\dot q)\) (split \(\mu_\pm\) if needed) |
| **M3** | \(+\,g(q)\) (gravity about cabinet axis) |
| **M4** | \(+\,\tau_{\mathrm{constraint}}\) if hand/cabinet contact |

Target accounting:

\[
\boxed{
M(q)\ddot q+h(q,\dot q)=\tau_{\mathrm{applied}}+\tau_{\mathrm{constraint}}
}
\]

If contact generalized force is unread, every earlier predictor is
structurally wrong.

## Interface audit (before fitting \(\phi\))

Likely mismatch (same class as prior SAPIEN/MuJoCo force bugs):

1. \(q,\dot q,\ddot q\) at the **same** evaluation time;
2. \(\tau\) = command vs applied vs **total** generalized force;
3. SAPIEN already applying passive damping/friction;
4. omitted constraint/contact;
5. joint axis vs gravity projection;
6. \(\Delta t\) / `qacc` buffer pre- vs post-step.

Priority: is predicted \(\tau\) the force the integrator actually uses?

## Do not start with scene ID

X0R G2 (\(\hat\phi\) worse than fixed \(\phi\)) does **not** mean
parameters are unidentifiable. Structure first.

## Gates (frozen)

Numeric margins in the runner header **before** first X0S collect.

| id | requirement |
|---|---|
| **G0** | same-time accounting of \(q,\dot q,\ddot q,\tau\) |
| **G1** | oracle simulator \(\phi\): \(E_{\ddot q}^{\mathrm{phy}}<E_{\ddot q}^{\mathrm{identity}}\) with a declared margin. Else **`structure/interface still wrong`** — STOP |
| **G2** | each kept term reduces \(\mathrm{RMS}(r_\tau)\); drop dead terms |
| **G3** | only after G1/G2: rollout \(H\in\{10,50\}\) on cabinet DoF only |

`rtwx_x0s_passed` iff G0 \(\land\) G1 \(\land\) G2 \(\land\) G3.

Capacity sweep **only after**: oracle physics \(<\) identity **and**
identified physics \(\approx\) oracle physics, **then** a stable latent
baseline. Otherwise physics-vs-latent compares two uncalibrated
instruments.

## Patterns

| pattern | meaning |
|---|---|
| `structure_closed` | G1–G3; ID/capacity still later |
| `structure_interface_wrong` | G1 fail even with oracle \(\phi\) |
| `contact_unaccounted` | M4 required and \(\tau_{\mathrm{constraint}}\) unread |
| `timing_mismatch` | G0 fail |
| `capacity_claim` | **forbidden** |

## Claims ceiling

May say: which minimal family beats identity on cabinet \(\ddot q\),
or that the interface is still wrong. May **not** say: physics reduces
latent capacity; X0R Euler is validated; TASK-XL unlocked.

## Ledger

```text
RoboTwin-X0
= RAN / capacity claim WITHHELD
原因：
- A/C 接近 identity predictor
- residual 反而恶化
- φ 未辨识
- latent rollout 不稳定

TASK-X0
= FAIL
cup    : p_no_nll
stamp  : p_no_nll   ← 当前探针下是真负结果
cabinet: coverage_hole ← 仪器不足
TASK-X1 = LOCKED

RoboTwin-X0R
= FAIL / excitation_failure
G1 force-auditable = PASS
G0 physics > identity = FAIL
G2 φ-identification = FAIL
G3 latent rollout sanity = FAIL
honest bottleneck = F_phy structure / force accounting
  (Δs≠0; audit OK)

RTWX-X0S = RAN / FAIL  timing_mismatch
           G0: corr(τ,q̈)≈0; command ≠ integrator force
TASK-XL  = DESIGN FROZEN ONLY; LOCKED
R10      = LOCKED
```
