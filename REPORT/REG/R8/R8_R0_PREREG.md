# R8-R0 Preregistration — Billed Probe→Reset Feasibility

Date: 2026-08-17  
Status: **RAN**; **`R8_R0_PASS=false`**; recovery_failure; A0 locked  
Depends on: `REPORT/REP/R8/R8_P2_REPORT.md` (open-loop family frozen)  
Does not: validity detector; \(\tau\); \(\pi\); NetVoI; \(T_{\mathrm{probe}}/u_0\)
change; gain fit on \(F_{\max}=1.5\), \(S_{\mathrm{epi}}\), \(A\), or \(J\)

## Question

Can the **frozen P0 one-way probe** keep its validity evidence if a
separate, task-independent, fully billed reset restores
\((y,\dot y)\to(0,0)\) before the original decision action?

\[
\boxed{
\text{probe (evidence)}
\;\to\;
\text{reset (physical state)}
\;\to\;
\text{same absolute episode horizon for }J
}
\]

Open-loop P1/P2 showed that one waveform cannot both excite and return.
R0 splits those jobs. No validity gate.

## Frozen probe (P0, unchanged)

\[
u_{\mathrm{epi}}=1,\qquad T_{\mathrm{probe}}=0.10\,\mathrm{s},\qquad n=50.
\]

\[
S_{\mathrm{epi}}=\lvert y(T_{\mathrm{probe}})-\hat y_{\mathrm{WM},k=0}\rvert
\]

recorded **before** reset. Same G1: commit-set gap \(\ge 10^{-3}\).

Physical state may be reset; epistemic evidence is not.

## Frozen reset (nominal PD, state only)

Design plant: unsaturated
\(\ddot y=\alpha_{\mathrm{des}}u-c\dot y\) with
\(\alpha_{\mathrm{des}}=F_{\max}^{\mathrm{ID}}=2.0\), \(c=4\).
Pole placement: \(\omega_n=10\,\mathrm{rad/s}\), \(\zeta=1\).

\[
k_p=\omega_n^2/\alpha_{\mathrm{des}}=50,\qquad
k_d=(2\zeta\omega_n-c)/\alpha_{\mathrm{des}}=8.
\]

\[
u_{\mathrm{reset}}(y,\dot y)
=
\mathrm{clip}(-k_p y-k_d\dot y,-1,1).
\]

Inputs: only \(y,\dot y\). Not \(F_{\max}\), \(\mathcal V\),
\(S_{\mathrm{epi}}\), \(A\), or \(J\).

Stop at first step with \(D=\mathrm{hypot}(y,\dot y)\le 2.2\times10^{-3}\),
or when the frozen episode budget is exhausted.

## Frozen billing horizon

\[
T_{\mathrm{total}}=0.50\,\mathrm{s}
\]

(same as original task \(J\) window / `PROBE_S`). After hold-at-rest,
every strategy is scored on the same \(n=250\) steps:

- baseline: \(u=a\) for all \(250\) steps (original \(J(a)\) from rest);
- acquisition: \(50\) probe steps, then reset, then remaining steps \(u=a\).

Reset time is billed as lost task time. No extra \(\beta_T T_{\mathrm{reset}}\).

\[
C_{\mathrm{epi}}=J(\text{probe}\to\text{reset}\to a)-J(a\text{ from rest}).
\]

\(a=u_{\mathrm{default}}=1\) on the ID commit-set. Gate:
\(\overline{C}_{\mathrm{epi}}/\overline{|A|}\le 1\), same \(|A|\) as P0 G3.

## Gates (all required)

- **G1** evidence (pre-reset \(S_{\mathrm{epi}}\)):
  \(\min S_{\mathrm{invalid}}-\max(S_{\mathrm{ID}},S_{\mathrm{benign}})\ge 10^{-3}\)
- **G-reset** all of ID / benign / invalid on the commit-set:
  \(D_{\mathrm{reset}}\le 2.2\times10^{-3}\)
- **G-cost** \(\overline{C}_{\mathrm{epi}}/\overline{|A|}_{\mathrm{commit,ID}}\le 1\)
- **G-indep** \(u_{\mathrm{reset}}=\pi_{\mathrm{reset}}(y,\dot y)\) only

## Outcomes

| G1 | G-reset | G-cost | reading |
|---|---|---|---|
| ✓ | ✓ | ✓ | unlock R8-A0 |
| ✓ | ✓ | × | recoverable, not worth this margin; stop family |
| ✓ | × | — | recovery/controllability limit; no detector |
| × | — | — | must not happen if P0 probe is reused |

No detector. No fifth open-loop waveform.
