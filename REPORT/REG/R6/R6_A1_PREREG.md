# R6-A1 Preregistration — Pairwise Margin-Uncertainty Audit

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R6/R6_A0_REPORT.md`, frozen R5-D0  
Does not change: \(\mathcal A\); \(J\); outer LOO ridge \(\Pi\); D0 `GO=false`  
Locks: \(J\)-weighted WM loss; delta-method
\(\nabla J^\top\Sigma_Y\nabla J\); nearest-competitor-only audit;
threshold-tuned planners. **R6-B0** is licensed only as frozen
abstention closure, not as planner optimization.

## Role

Diagnostic split only, not a GO contest:

\[
\boxed{
\text{uncertain wrong ranking}
\quad\text{vs}\quad
\text{confident wrong ranking}.
}
\]

Question: can the frozen D0 maps know how unreliable \(\hat m_{ab}\) is?
Not: can we make \(\hat m\) more accurate?

A0 already established

\[
\|\hat Y-Y\|^2\downarrow
\not\Rightarrow
\#\{\hat m_{a^\star b}<0\}\downarrow
\]

and \(R^2(g^\top\delta Y,\,e^J)=0.114\). Trajectory MSE and first-order
\(J\)-sensitivity are **not** sufficient proxies for decision
uncertainty. A1 audits in pairwise cost-margin space.

## Frozen pipeline

Outer held \(\lambda_i\): same D0 ridge-then-\(J\) maps for \(\pi_0\)
and \(\pi_X\). Inner-LOO uses only that outer training set. No new
estimator class. No Adam.

## Pairs

All ordered pairs, not nearest competitor:

\[
\forall(a,b)\in\mathcal A^2,\ a\neq b.
\]

A0 \(\lambda=12\) flipped via cons while nearest was agg.

## Scale (primary)

For each outer fold, planner, and pair, inner-LOO margin errors

\[
\eta_{ab}^{(j)}=\hat m_{ab}^{(j)}-m_{ab}^{(j)},
\qquad
\sigma_{ab}=\mathrm{RMS}_j(\eta_{ab}^{(j)}).
\]

MAD is logged, not used for diagnostics. Interval is one RMS:

\[
I_{ab}=[\hat m_{ab}-\sigma_{ab},\,\hat m_{ab}+\sigma_{ab}].
\]

\[
z_{ab}=\frac{|\hat m_{ab}|}{\sigma_{ab}},
\qquad
\text{low confidence}\iff z_{ab}<1
\iff 0\in I_{ab}.
\]

Uncertainty is obtained by pushing \(\hat Y\) through frozen \(J\), not
by linearizing \(J\).

## Diagnostics (not pass/fail GO)

- **U1:** fraction of ranking errors (\(\hat a\neq a^\star\)) whose
  **deciding pair** \((a^\star,\hat a)\) has \(z<1\).
- **U2:** median \(z\) on those deciding error pairs versus median \(z\)
  on all \((a^\star,b)\) pairs of ranking-preserved \(\lambda\).
- **U3:** full six-pair tables at \(\lambda=0,2,12\) for \(\pi_X\).

**\(\lambda=2\) veto:** if \(\pi_X\) is wrong at \(\lambda=2\) and that
deciding pair has \(z\ge 1\), A1 does **not** license R6-B0, even if
\(\lambda=0\) and \(\lambda=12\) are low-confidence.

Fork:

- all \(\pi_X\) errors low-confidence **and** \(\lambda=2\) not a
  confident error \(\rightarrow\) B0 **abstention** candidate (WM frozen;
  \(z_{\min}=1\) fixed);
- otherwise \(\rightarrow\) ranking-sensitive epistemic allocation,
  still without writing \(J\) into dynamics.
