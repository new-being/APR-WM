# SIM-X2 Preregistration — Action-Conditioned Validity Information Channel

Date: 2026-08-17  
Status: **FROZEN**; **PASS** (`REPORT/REP/SIMX/SIMX2_REPORT.md`);
unlocks SIM-X3 only; **not an R10 gate**  
Depends on: `REPORT/REP/SIMX/SIMX1_REPORT.md`,
`aprwm_v0/simx_plant.py` (`simx_hinge.v1`)  
Does not: unlock R10-C0; SIM-X3 lifecycle; certificate;
\(b(\mathcal V)\); stale/revoke; hardware noise; retune
\(\sigma_{\mathrm{obs}}\) / amplitudes / frequencies after seeing formal
results; KNN / histogram / classifier MI estimators

## Split (frozen)

\[
\boxed{
\textbf{X2 = information channel}
\qquad
\textbf{X3 = information lifecycle}
}
\]

X2 asks: how much validity information does **one** residual sample
carry under a known action? Not how that information accumulates into
a certificate.

## Validity variable

\[
\mathcal V=
\begin{cases}
1,& b_{\mathrm{true}}=0.10\\
0,& b_{\mathrm{true}}\in\{0.05,0.15\}.
\end{cases}
\]

\[
P(\mathcal V=1)=P(\mathcal V=0)=\tfrac12,
\qquad
P(b=0.05\mid\mathcal V=0)=P(b=0.15\mid\mathcal V=0)=\tfrac12.
\]

Question: is the **nominal damping assumption** still valid? Not:
predict the sign of \(\Delta b\).

Learner nominal stays \(b_0=0.10\). Host = `simx_hinge.v1`.

## Observation channel (synthetic)

Truth residual (from X1):

\[
r_t=-\Delta b\,\dot q_t.
\]

Decision-visible:

\[
\boxed{
\tilde r_t=r_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma_{\mathrm{obs}}^2),
\qquad
\sigma_{\mathrm{obs}}=0.01\,\mathrm{N\cdot m}
}
\]

Name: **synthetic observation-resolution surrogate**.

Not: hardware sensor noise; real noise floor; R10 measurement model.
Without this, X1’s \(10^{-16}\) floor makes “blindness” vacuous.

## Actions

### Primary (same shape, different amplitude)

\[
a_{\mathrm{cons}}(t)=0.02\sin(2\pi\cdot 0.25\,t),\qquad
a_{\mathrm{info}}(t)=0.10\sin(2\pi\cdot 0.25\,t)
\quad(\mathrm{N\cdot m}).
\]

Mechanism from X1: \(|r|=|\Delta b|\,|\dot q|\). Predict:

\[
I(\mathcal V;\tilde r\mid a_{\mathrm{info}})
>
I(\mathcal V;\tilde r\mid a_{\mathrm{cons}}).
\]

### Iso-energy control (same torque RMS, different \(f\))

\[
a_{\mathrm{low\text{-}f}}(t)=0.10\sin(2\pi\cdot 0.25\,t),\qquad
a_{\mathrm{high\text{-}f}}(t)=0.10\sin(2\pi\cdot 5\,t).
\]

\(\mathrm{RMS}(u)\) equal. Expect \(E_{\dot q,\mathrm{low\text{-}f}}\gg
E_{\dot q,\mathrm{high\text{-}f}}\) hence
\(I_{\mathrm{low\text{-}f}}\gg I_{\mathrm{high\text{-}f}}\).

Falsifies “informative only because more torque energy.”

Note: \(a_{\mathrm{info}}\equiv a_{\mathrm{low\text{-}f}}\) by design
(primary and iso-energy share the informative waveform).

## Information (analytic, per-step)

Do **not** compute one MI over a whole trajectory (that mixes channel
with long-horizon accumulation → X3).

\[
\boxed{
I_a=\mathbb E_t\bigl[I(\mathcal V;\tilde r_t\mid a,t)\bigr]
\quad(\mathrm{bits/sample})
}
\]

on **steady-state** samples only (freeze: discard \(t<2\,\mathrm{s}\)).

Generative densities (known; no KNN / KDE / bins):

\[
p(\tilde r_t\mid\mathcal V=1,a)=\mathcal N(0,\sigma_{\mathrm{obs}}^2),
\]

\[
p(\tilde r_t\mid\mathcal V=0,a)
=
\tfrac12\mathcal N(r_t^{0.05},\sigma_{\mathrm{obs}}^2)
+
\tfrac12\mathcal N(r_t^{0.15},\sigma_{\mathrm{obs}}^2).
\]

Integrate

\[
I(\mathcal V;\tilde r_t\mid a,t)
=
\sum_v p(v)\int p(y\mid v,a,t)
\log_2\frac{p(y\mid v,a,t)}{p(y\mid a,t)}\,dy
\]

numerically. Report mean over seeds \(\{9101,9111,9121\}\).

## Mechanism diagnostic

\[
E_v(a)=\mathbb E_t[\dot q_t^2\mid a]
\]

(same steady window). Valuable chain:

\[
E_v(a_{\mathrm{info}})>E_v(a_{\mathrm{cons}})
\Rightarrow
|r|_{\mathrm{info}}>|r|_{\mathrm{cons}}
\Rightarrow
I_{\mathrm{info}}>I_{\mathrm{cons}}.
\]

## Gates (frozen thresholds — do not retune)

| id | question | pass |
|---|---|---|
| **G0** | X1 carryover under X2 harness | \(r\approx-\Delta b\dot q\) (\(E_{\mathrm{oracle}}<10^{-4}\)) |
| **G1** | action isolation | same \(a\): identical pre-registered `qfrc_applied` across \(b\in\{0.05,0.10,0.15\}\) |
| **G2** | primary ordering | \(I_{\mathrm{info}}>I_{\mathrm{cons}}\) and \(I_{\mathrm{info}}-I_{\mathrm{cons}}>0.20\) bit/sample |
| **G3** | blindness | \(I_{\mathrm{cons}}<0.05\) and \(I_{\mathrm{info}}>0.25\) bit/sample |
| **G4** | iso-energy | \(E_v(\mathrm{low\text{-}f})>E_v(\mathrm{high\text{-}f})\) and \(I_{\mathrm{low\text{-}f}}-I_{\mathrm{high\text{-}f}}>0.20\) |

## Forbidden

No retuning \(\sigma_{\mathrm{obs}}\), amplitudes, or frequencies to
rescue formal results. No detector training. No certificate. No
whole-trajectory MI as the primary metric. No “real noise” claim.

## PASS may claim

\[
\boxed{\textbf{action-conditioned validity observability reproduced on MuJoCo}}
\]

If G3 also holds:

\[
\boxed{
\textbf{a simulator instance of policy-induced epistemic blindness
reappears on an independent physics engine}
}
\]

Still only:

\[
\boxed{\text{cross-engine mechanism robustness}}
\]

Not real-physics evidence.

## After a pass

Unlock **SIM-X3** (lifecycle: accumulate → \(b(\mathcal V)\) →
shadow commit). R10-C0 stays locked.
