# R4-C2 Preregistration — Continuous Future-Consequence Epistemic Value

Date: 2026-08-17  
Status: **FROZEN** (run; `R4_C2_GO=false`; R4 family stop)  
Depends on: `REPORT/REP/R4/R4_C1_REPORT.md` (F0 true; F1 unevaluable)  
Does not change: I0 v2 physics, \(a^{\mathrm{diag}}\), \(H\), \(t_\star\),
\(C=f(Y^{\mathrm{future}})\), C1 F0/F1, C0 GO, I1 \(h^{S}\)  
Locks: lowering \(c_{\min}\); strengthening the probe; \(e=f(X)\);
encoder/CNN; I0 retune; treating I1 A/B AUROC as C2 GO

## Why not C1.x

C1 F0 showed a weak future fork. C1 F1 used \(e=\mathbf{1}\{C>c_{\min}\}\),
which saturated on train. That is **binary warrant degenerate**, not a
negative tactile result. C2 changes the **target**, not the physics.

\[
\boxed{\textbf{R4-C2 — Continuous Future-Consequence Epistemic Value}}
\]

中文名：**连续未来后果的认识价值**。

## Question

\[
\boxed{
I(X_{t_\star};C\mid h^{S}_{t_\star})>0\ ?
}
\]

Given currently matched macros, does local tactile help predict **how
much** the future will fork — not which of A/B it is?

\[
C=f(Y^{A}_{t_\star:t_\star+H},Y^{B}_{t_\star:t_\star+H}),
\qquad
Y=\{q,\dot q,F,M\}.
\]

No \(X\) in \(C\). Pair-level: A and B at the same \(q_0\) share one \(C\).
I1’s \(X\to A/B\) classifier therefore cannot solve C2 by itself.

## Frozen probe (identical to C1)

\(t_\star=\) last hold frame; \(F_t^{\mathrm{cmd}}=4.8\) for \(0.40\,\mathrm{s}\);
same \(C\) formula as C1. **Do not** enlarge \(C\) by retuning \(\mu\),
`solref`, or \(a^{\mathrm{diag}}\).

## Statistics only: more \(q_0\)

I0 physics frozen. Add **fresh initial** \(q_0\) only, split by
trajectory:

- Train: \(17\) values \(\mathrm{linspace}(-8,+8)\times10^{-4}\)
- Held: \(\{\pm10.5,\pm12,\pm13.5,\pm15\}\times10^{-4}\)

## Models (linear; no encoder; no Adam)

\[
P_0:h^{S}_{t_\star}\to C,
\qquad
P_X:(h^{S}_{t_\star},X_{t_\star})\to C.
\]

\(h^{S}\) is I1’s over-complete \(W=8\) history. Ridge linear map.
Target is **z-scored \(C\)** using train pair statistics (unitless).

\[
L_C=\mathrm{MSE}_{\mathrm{held}}(\hat z,z(C)),
\qquad
\Delta_C=L_C(P_0)-L_C(P_X).
\]

If train \(\mathrm{std}(C)<10^{-6}\): degenerate, no GO.

## GO

\[
\texttt{R4\_C2\_GO}
\iff
\Delta_C>0.05.
\]

Chance-level \(L_C(P_0)\) is near \(1\) if \(h^{S}\) cannot predict
\(z(C)\). Predicting the train mean yields \(L\approx 1\).

## Diagnostics (not GO)

- **canonicalize** \(X^B(x)\mapsto X^B(-x)\): gain should drop if the
  predictor used left/right allocation rather than totals.
- **shuffle** \(X\) across train/held rows, keep \(C\) and \(h^{S}\):
  gain should vanish if the increment is time/instance-locked.

## Stop

| Outcome | Action |
|---|---|
| GO true | tactile has **lead time** on future-macro magnitude; still no CNN this stage |
| GO false | \(\text{future }C>0\not\Rightarrow X\text{ quantifies }C\mid h^{S}\); **R4 family stop**; no encoder, no probe retune |

## What this does not authorize

- C1 \(c_{\min}\) shopping
- stronger \(a^{\mathrm{diag}}\) to widen \(C\)
- Door V7E / DINO / fusion / larger GRU
