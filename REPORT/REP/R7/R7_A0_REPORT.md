# R7-A0 Report — One-Sided Switch Certificate

Date: 2026-08-17  
Status: **`R7_A0_GO=true`**  
Prereg: `REPORT/REG/R7/R7_A0_PREREG.md`  
Artifacts: `runs/r7_a0/formal/summary.json`  
Does not use: P0 \(\alpha\) grid; \(|\hat A|/\sigma\); neural probe  
Frozen: \(\delta=10^{-3}\), \(\varepsilon=0.10\), degree-2 OLS in \(X\)

## Question

Can \(X\) support a one-sided split-conformal lower bound \(L=\hat A-q\)
that commits to cons only when \(L>\delta\)?

## Gates (held test, \(n=10\), aligned \(X\))

| Gate | Value | Pass |
|---|---|---|
| G-cert \(P(A\ge L)\) | **0.90** \(\ge 0.90\) | true |
| G-safe precision \(P(A>0\mid L>\delta)\) | **1.0** (\(n_{\mathrm{commit}}=5\), not vacuous) | true |
| G-recall \(P(L>\delta\mid A\ge\delta)\) | **1.0** (5/5 useful) | true |
| G-VoI \(\mathbb{E}[A\,1\{L>\delta\}]\) | **\(5.43\times10^{-3}\)** | true |
| G-ctrl \(\mathrm{VoI}_{\mathrm{align}}>\mathrm{VoI}_{\mathrm{shuf}}\) | \(5.43\times10^{-3}>0\) | true |

\[
\boxed{\texttt{R7\_A0\_GO}=\text{true}}
\]

Shuffle-\(X\) never commits (vacuous abstain, \(\mathrm{VoI}=0\)). The
certificate is not a base-rate trick.

Held commits are exactly the five useful regimes
(\(\alpha\in\{2.25,2.55,2.85,3.15,3.45\}\)). All five harmful test
points abstain, including \(\alpha=1.95\) (\(A=-5.1\times10^{-4}\),
below \(\delta\)).

## What this is

The frozen \(A(\alpha)\) is quadratic in \(\alpha\) and \(X\propto\alpha\),
so degree-2 \(\hat A(X)\) is nearly interpolating (\(q\approx 0\)). A0
therefore tests the **commit rule and the four-way gate split**, not a
hard residual world model.

That is still the missing R6 piece:

\[
\boxed{
X\rightarrow\hat A\rightarrow L(X)>\delta
\rightarrow\text{useful cons switch}
\rightarrow\mathrm{VoI}_{\Pi_{\mathrm{cert}}}>0.
}
\]

B0 could only say “this deviation is uncertified.” A0 can say “this
specific alternative is certified by at least \(\delta\).”

P0’s eight \(\alpha\) were not in fit, calibration, or test.
\(\delta\) was not moved after seeing regret. No threshold sweep.

## Not claimed

Not claimed: the same conformal wrapper would pass on a misspecified
\(f\). That is **R7-P1/A1**. P1 passed
(`REPORT/REP/R7/R7_P1_REPORT.md`); A1 is not run here.
