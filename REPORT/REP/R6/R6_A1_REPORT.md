# R6-A1 Report — Pairwise Margin-Uncertainty Audit

Date: 2026-08-17  
Status: diagnostic (not a GO contest)  
Prereg: `REPORT/REG/R6/R6_A1_PREREG.md`  
Artifacts: `runs/r6_a1/formal/summary.json`  
Depends on: frozen R6-A0 and R5-D0  
Does not train: planner, dynamics, or \(J\)-weighted loss  
Does not use: \(\nabla J^\top\Sigma_Y\nabla J\)

## Question

Can the frozen D0 maps know how unreliable \(\hat m_{ab}\) is?
Scale is nested-LOO RMS in **cost-margin** space, all six ordered
pairs, \(z=\lvert\hat m\rvert/\sigma\), low confidence iff \(z<1\).

## Diagnostics

| | \(\pi_0\) | \(\pi_X\) |
|---|---|---|
| rank errors | 1 | 3 |
| U1 low-conf fraction | **0** | **1** |
| confident wrong | 1 (\(\lambda=0\)) | **0** |
| U2 median \(z\) error vs preserved \(a^\star\)-pairs | 2.71 vs 0.33 (false) | 0.22 vs 0.44 (true) |
| \(\lambda=2\) veto (\(z\ge 1\)) | — | **false** (\(z=0.474\)) |
| D0 actions reconstructed | true | true |

Formal fork field: `b0_candidate` (`b0_licensed=true` under the frozen
rule: every \(\pi_X\) error has \(z<1\), and \(\lambda=2\) is not a
confident error). A1 does **not** run B0.

## The three PX cases

| \(\lambda\) | \(a^\star\to\hat a\) | deciding \(\hat m\) | \(\sigma\) | \(z\) | \(0\in I\) |
|---|---|---|---|---|---|
| 0 | cons \(\to\) mid | \(-4.26\times10^{-6}\) | \(1.93\times10^{-5}\) | 0.221 | yes |
| 2 | mid \(\to\) cons | \(-2.89\times10^{-5}\) | \(6.10\times10^{-5}\) | 0.474 | yes |
| 12 | mid \(\to\) cons | \(-9.25\times10^{-8}\) | \(4.96\times10^{-5}\) | 0.002 | yes |

**\(\lambda=0\):** wrong sign, low confidence. The point estimate only
just crossed to mid; a 1-RMS interval contains 0. The planner still
committed.

**\(\lambda=12\):** numerical tie. If this point had been high-\(z\)
cons, the scale would have been rejected. It is not.

**\(\lambda=2\) (pressure test):** \(\rho=-5.35\) is a real pair-specific
cost error, not a \(10^{-8}\) knife edge, but it still sits inside one
inner-LOO RMS (\(z=0.474\)). Nested-LOO **does** mark it low-confidence.
The veto against B0 does not fire.

## What A1 does *not* show

Low confidence is not exclusive to errors. For \(\pi_X\), the
**decision-critical pair mid vs cons has \(z<1\) at every \(\lambda\)**,
including the four \(\lambda\) where ranking is already correct
(\(z\in\{0.025,0.38,0.56,0.50\}\)). High \(z\) lives on pairs against
`agg`, which are not the D0 failure mode.

So:

\[
\boxed{
\text{PX ranking errors are uncertain}
}
\]

and also

\[
\boxed{
\text{PX never has a 1-RMS-confident cons vs mid ranking}.
}
\]

An LCB / \(z>1\) switch rule on that pair is predicted to **abstain
everywhere** and collapse toward \(\pi_0\) (constant mid): it would
likely retract the two injected cons errors at \(\lambda=2,12\), and
it would **not** recover cons at \(\lambda=0\).

\(\pi_0\) itself is the opposite object at \(\lambda=0\):
\(\hat m_{\mathrm{cons,mid}}=-3.50\times10^{-5}\),
\(\sigma=1.29\times10^{-5}\), \(z=2.71\) — **confidently wrong**.
Giving \(X\) to the map lowers that \(z\) from 2.71 to 0.22 (the sign
stays wrong). That is epistemic softening, not a calibrated flip.

## Fork

**R6-B0** (`REPORT/REP/R6/R6_B0_REPORT.md`) ran the frozen abstention
preflight: `B0_POLICY_COLLAPSE=true`, \(\pi_B\equiv\pi_0\), no simulator
rerun. Harmful switches at \(\lambda=2,12\) retracted; useful switch at
\(\lambda=0\) never proposed. Realization branch **stopped**.
