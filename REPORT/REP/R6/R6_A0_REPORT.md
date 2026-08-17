# R6-A0 Report — Decision-Sensitive Error Audit

Date: 2026-08-17  
Status: diagnostic (not a GO contest)  
Prereg: `REPORT/REG/R6/R6_A0_PREREG.md`  
Artifacts: `runs/r6_a0/formal/summary.json`  
Depends on: frozen R5-D0 (`GO=false`)  
Does not train: planner, world model, encoder, or \(J\)-weighted loss

## Question

Why did trajectory \(L_Y\) fall under \(\pi_X\) while action ranking
worsened? Attribute D0 errors to cost-margin perturbations

\[
\hat m_b=m_b+(e_b^J-e_{a^\star}^J)=m_b(1+\rho_b),\qquad
\rho_b<-1\Rightarrow\text{pair ranking flips}.
\]

## Gates

| Gate | Result |
|---|---|
| D0 actions reconstructed | **true** (LOO ridge maps match D0 \(\hat a\)) |
| D1 ranking-error attribution | **true** |
| D2 metric mismatch | **true** (via more margin violations, not larger mean \(\lvert\Delta e^J\rvert\)) |
| D3 \(R^2\) of \(g^\top\delta Y\) vs \(e^J\) | **0.114** (nonlinear flag) |

D0 `GO=false` and the self-stress family STOP are unchanged.

## Core table (\(\pi_X\); \(\pi_0\) in artifact)

Mean \(L_Y\): \(\pi_0=1.544\), \(\pi_X=0.723\) (relative drop \(53.1\%\)).
Rank errors: \(1\to 3\). Pair violations \(\hat m_b<0\): \(1\to 3\).
Mean nearest \(\lvert\Delta e^J\rvert\) actually **fell**
(\(1.02\times10^{-4}\to 4.65\times10^{-5}\)).

| \(\lambda\) | \(a^\star\) | \(\hat a_X\) | nearest | \(m_{\mathrm{near}}\) | \(\Delta e^J_{\mathrm{near}}\) | \(\rho_{\mathrm{near}}\) | preserved |
|---|---|---|---|---|---|---|---|
| 0 | cons | mid | mid | \(1.21\times10^{-4}\) | \(-1.25\times10^{-4}\) | \(-1.035\) | no |
| 2 | mid | **cons** | cons | \(6.66\times10^{-6}\) | \(-3.56\times10^{-5}\) | \(-5.35\) | no |
| 4 | mid | mid | cons | \(2.67\times10^{-5}\) | \(-2.53\times10^{-5}\) | \(-0.947\) | yes (knife) |
| 6 | mid | mid | cons | \(3.09\times10^{-5}\) | \(-1.07\times10^{-5}\) | \(-0.348\) | yes |
| 8 | mid | mid | agg | \(1.24\times10^{-5}\) | \(+5.51\times10^{-5}\) | \(+4.46\) | yes |
| 10 | mid | mid | agg | \(1.63\times10^{-5}\) | \(-8.86\times10^{-6}\) | \(-0.542\) | yes |
| 12 | mid | **cons** | agg | \(1.43\times10^{-5}\) | \(+6.54\times10^{-5}\) | \(+4.56\) | no |

The \(\lambda=12\) flip is **not** the nearest pair. Against oracle
winner, \(\rho_{\mathrm{cons}}=-1.004\) and
\(\hat m_{\mathrm{cons}}=-9.2\times10^{-8}\).

## The three D0 failures

**\(\lambda=0\) (should switch to cons, both planners stay mid).**
True \(m_{\mathrm{mid}}=1.21\times10^{-4}\). \(\pi_0\): \(\rho=-1.29\).
\(\pi_X\): \(\rho=-1.035\) — closer to the flip line, still negative.
PX reduced \(e_{\mathrm{cons}}^J\) (\(3.72\times10^{-5}\to 6.46\times10^{-6}\))
but left \(e_{\mathrm{mid}}^J\approx -1.18\times10^{-4}\). Mid remains
under-costed relative to cons by more than the true margin.

**\(\lambda=2\) (true mid; PX picks cons).**
True \(m_{\mathrm{cons}}=6.66\times10^{-6}\). \(\pi_0\) is safe
(\(\rho=+2.28\)). PX: \(e_{\mathrm{cons}}^J=-2.41\times10^{-5}\),
\(e_{\mathrm{mid}}^J=+1.15\times10^{-5}\), \(\rho=-5.35\). \(L_Y\) and
\(\lVert\delta Y^\perp\rVert\) both fall; the **pair** cost error still
swamps a \(10^{-6}\) margin.

**\(\lambda=12\) (true mid; PX picks cons).**
\(\rho_{\mathrm{cons}}\) moves from \(-0.235\) to \(-1.004\). Flip is a
numerical knife-edge, not a large trajectory-MSE event.

## \(L_Y\) vs decision-sensitive error

|  | \(\pi_0\) | \(\pi_X\) |
|---|---|---|
| mean \(L_Y\) | 1.544 | 0.723 |
| mean \(\lvert g^\top\delta Y\rvert\) | \(2.06\times10^{-4}\) | \(9.40\times10^{-5}\) |
| mean \(\lVert\delta Y^\perp\rVert\) | 31.3 | 22.1 |

PX improved **both** the insensitive residual and the mean
\(\lvert g^\top\delta Y\rvert\). The hypothesized slogan
“predictive gain lived only in \(\delta Y^\perp\)” is **too strong**
for the averages. What D2 actually shows:

\[
\boxed{
\text{global }L_Y\downarrow
\ \not\Rightarrow\
\text{fewer }a^\star\text{-pair margin violations}.
}
\]

True action margins are \(O(10^{-5})\). Mean \(\lvert\Delta e^J\rvert\)
can fall while more pairs still satisfy \(\rho_b<-1\).

D3: \(R^2=0.114\). Frozen \(J\) is quadratic in \(q_T\), \(\overline{u^2}\),
and limit excess, so \(e^J\approx g^\top\delta Y\) is a weak local
model. Later work should not treat gradient weighting as a sufficient
proxy for ranking.

## Fork after A0 (not executed)

Not licensed: retrain the world model on \(J\).

- **B is the tighter reading of these numbers:** predicted ranking sits
  inside \(O(10^{-5})\) (and \(10^{-8}\)) margins. Next scientific object
  is \(\hat m_{ab}\) versus uncertainty \(\sigma(m_{ab})\), not a new
  hidden state.
- **A remains open as a secondary fact:** pair-specific \(\Delta e^J\)
  (especially \(\lambda=2\)) is still large relative to \(m_b\), even
  when average \(\lvert g^\top\delta Y\rvert\) falls. That is
  epistemic allocation on ranking-sensitive comparisons, not
  task-rewritten dynamics.

\[
\boxed{
\begin{aligned}
\text{sensor information}
&\neq \text{conditional information}\\
&\neq \text{consequential predictive information}\\
&\neq \text{oracle decision relevance}\\
&\neq \text{realized decision value}.
\end{aligned}
}
\]

R6 asked which prediction errors matter for ranking. A0’s answer:

\[
\boxed{
e_b^J-e_{a^\star}^J
\quad\text{versus}\quad
m_b,
\quad\text{not}\quad
\lVert\hat Y-Y\rVert^2.
}
\]

**R6-A1** (`REPORT/REP/R6/R6_A1_REPORT.md`) then audits nested-LOO
uncertainty of those pairwise margins. It does not retrain.
