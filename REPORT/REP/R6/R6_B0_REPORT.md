# R6-B0 Report — Frozen Uncertainty-Abstention Realization

Date: 2026-08-17  
Status: **`B0_POLICY_COLLAPSE=true`**; realization branch **STOP**  
Prereg: `REPORT/REG/R6/R6_B0_PREREG.md`  
Artifacts: `runs/r6_b0/formal/summary.json`  
Simulator rerun: **false** (algebraic D0 lookup)  
Does not sweep: \(z_{\min}\)

## Rule (frozen)

\[
\pi_B=
\begin{cases}
\pi_X & \text{if }\pi_X\neq\pi_0\text{ and }z_{\pi_0,\pi_X}\ge 1,\\
\pi_0 & \text{otherwise.}
\end{cases}
\]

\(z_{\min}=1\) is A1’s RMS threshold. No \(\lambda=0\) special case.

## Policy-identity preflight

\[
\boxed{\texttt{B0\_POLICY\_COLLAPSE}=\text{true}}
\]

\(\pi_B(\lambda)=\pi_0(\lambda)=\mathrm{mid}\) at every \(\lambda\).
No new rollouts. \(\bar R_B=\bar R_0=1.72\times10^{-5}\).

| \(\lambda\) | \(a^\star\) | \(\pi_0\) | \(\pi_X\) | \(z_{\pi_0,\pi_X}\) | \(\pi_B\) | decision |
|---|---|---|---|---|---|---|
| 0 | cons | mid | mid | — | mid | no proposal |
| 2 | mid | mid | cons | 0.474 | mid | abstain |
| 4–10 | mid | mid | mid | — | mid | no proposal |
| 12 | mid | mid | cons | 0.002 | mid | abstain |

## Harmful vs useful

| Quantity | Value |
|---|---|
| \(N_{\mathrm{harmful\ proposed}}\) | 2 (\(\lambda=2,12\)) |
| \(N_{\mathrm{harmful\ retracted}}\) | **2** |
| \(N_{\mathrm{useful\ proposed}}\) | **0** |
| \(N_{\mathrm{useful\ accepted}}\) | **0** |
| \(\bar R_0\) | \(1.72\times10^{-5}\) |
| \(\bar R_X\) | \(2.18\times10^{-5}\) |
| \(\bar R_B\) | \(1.72\times10^{-5}\) |
| \(\mathrm{VoI}_\Pi\) recovered | **false** |

\(\lambda=0\) never becomes a useful proposal: \(\pi_X\) does not leave
\(\pi_0\). Abstention cannot recover a switch that was not proposed.

## What this is not

Not “uncertainty-aware planner success.” Collapse with
\(\bar R_B=\bar R_0\) means:

\[
\boxed{
\text{uncertainty repaired harmful realization by abstaining,
but did not recover positive }\mathrm{VoI}_\Pi.
}
\]

\[
\boxed{
\text{uncertainty can veto unsupported decisions}
\neq
\text{uncertainty can identify the correct alternative}.
}
\]

A1’s softening at \(\lambda=0\) (\(z:2.71\to 0.22\)) is
**epistemic softening**, not **decision correction**: the sign of
\(\hat m\) stays wrong, and B0 therefore never sees a certified
cons proposal.

## Branch stop

This realization branch **stops**. Do not try \(z_{\min}\in\{0.8,0.5,0.2\}\)
on seven \(\lambda\). Do not train a classifier on this STOP family.

A1’s retained claim:

\[
\boxed{
\text{low confidence can be a valid reason not to act,
without being sufficient evidence for what action should be taken instead.}
}
\]

A later stage is **R7 switch certification**
(`REPORT/REG/R7/R7_PREREG.md`), locked until a **new** decision family.
Not threshold fitting here. Ledger:
`REPORT/REP/R6/R6_REALIZATION_FREEZE.md`.
