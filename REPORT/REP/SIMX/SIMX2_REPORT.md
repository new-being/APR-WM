# SIM-X2 Report — Action-Conditioned Validity Information Channel

Date: 2026-08-17  
Status: **`sim_x2_passed=true`**; unlocks **SIM-X3 only**  
Does not: unlock R10-C0; claim real physics; certificate lifecycle

## Question

Under synthetic observation resolution \(\sigma_{\mathrm{obs}}=0.01\,\mathrm{N\cdot m}\),
does per-step analytic

\[
I_a=\mathbb E_t\bigl[I(\mathcal V;\tilde r_t\mid a,t)\bigr]
\]

reproduce action-conditioned validity observability (and a conservative
near-blind channel) on `simx_hinge.v1`?

## Setup (frozen)

| item | value |
|---|---|
| host | `simx_hinge.v1` |
| \(b_0\) | \(0.10\) frozen |
| \(\mathcal V\) | valid iff \(b_{\mathrm{true}}=0.10\) |
| \(\sigma_{\mathrm{obs}}\) | \(0.01\) synthetic surrogate (not hardware) |
| MI | analytic mixture integral; no KNN/KDE |
| window | steady \(t\ge 2\,\mathrm{s}\) |

Actions: `cons` \(0.02\sin(2\pi 0.25 t)\), `info`/`low_f`
\(0.10\sin(2\pi 0.25 t)\), `high_f` \(0.10\sin(2\pi 5 t)\).

## Results (seed-mean)

| action | \(I_a\) [bit/sample] | \(E_v=\mathbb E[\dot q^2]\) |
|---|---:|---:|
| cons | \(0.00392\) | \(0.00556\) |
| info | \(0.265\) | \(0.139\) |
| low_f | \(0.265\) | \(0.139\) |
| high_f | \(3.73\times 10^{-5}\) | \(4.81\times 10^{-4}\) |

\[
I_{\mathrm{info}}-I_{\mathrm{cons}}=0.261,\qquad
I_{\mathrm{low\text{-}f}}-I_{\mathrm{high\text{-}f}}=0.265.
\]

## Gates

| gate | result |
|---|---|
| G0 X1 carryover | pass |
| G1 action isolation | pass |
| G2 primary ordering | pass (\(>0.20\) gap) |
| G3 blindness | pass (\(I_{\mathrm{cons}}<0.05\), \(I_{\mathrm{info}}>0.25\)) |
| G4 iso-energy | pass |

Chain holds:

\[
E_v(\mathrm{info})>E_v(\mathrm{cons})
\Rightarrow
|r|_{\mathrm{info}}>|r|_{\mathrm{cons}}
\Rightarrow
I_{\mathrm{info}}>I_{\mathrm{cons}},
\]

and the same under equal torque RMS (`low_f` vs `high_f`).

## Claims allowed

\[
\boxed{\textbf{action-conditioned validity observability reproduced on MuJoCo}}
\]

\[
\boxed{
\textbf{a simulator instance of policy-induced epistemic blindness
reappears on an independent physics engine}
}
\]

Still only **cross-engine mechanism robustness**. Not real physics.

## Next

**SIM-X3**: channel → lifecycle (\(b(\mathcal V)\), shadow commit).
Do not fold accumulation back into X2. R10-C0 stays locked.

Artifacts: `runs/sim_x2/formal/`.
