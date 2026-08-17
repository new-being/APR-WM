# R7-A0 Preregistration — One-Sided Switch Certificate

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R7/R7_P0_REPORT.md` (`PASS=true`)  
Does not reuse: P0 \(\alpha\) grid as formal evidence; self-stress \(\lambda\);
\(|\hat A|/\sigma\); neural / Adam / ensemble variance  
Does not tune: \(\delta\) or \(\varepsilon\) after seeing held regret

## Question

Can \(X\) support a valid **one-sided** certificate that selectively
proves \(A>\delta\)?

\[
L(X)>\delta\Rightarrow\mathrm{commit\ cons},\qquad
L(X)\le\delta\Rightarrow\mathrm{default/abstain}.
\]

\(A>0\) iff cons beats mid. \(L\) is a lower bound for \(A\), not a
symmetric margin \(z\).

## Predictor and split

Low-capacity \(\hat A=f(X)\): OLS polynomial of degree 2 in \(X\).
No \(h^{S}\) in \(f\). Splits are by \(\alpha\) (no \(\alpha\) in two
splits). P0 grid
\(\{0.6,1.0,1.4,1.8,2.2,2.6,3.0,3.4\}\) is **development only**.

\[
\begin{aligned}
\mathrm{fit}&:
\{0.55,0.85,1.15,1.45,1.75,2.05,2.35,2.65,2.95,3.25\}\\
\mathrm{cal}&:
\{0.65,0.95,1.25,1.55,1.85,2.15,2.45,2.75,3.05,3.35\}\\
\mathrm{test}&:
\{0.75,1.05,1.35,1.65,1.95,2.25,2.55,2.85,3.15,3.45\}
\end{aligned}
\]

## Certificate

Calibration scores \(s_i=\hat A(X_i)-A_i\). Split-conformal upper
quantile at \(\varepsilon=0.10\):

\[
k=\lceil(n_{\mathrm{cal}}+1)(1-\varepsilon)\rceil,\quad
q=s_{(k)}\ \text{or }+\infty\text{ if }k>n_{\mathrm{cal}}.
\]

\[
L(X)=\hat A(X)-q,\qquad P(A\ge L)\ge 1-\varepsilon.
\]

## Frozen \(\delta\) and regions

\(\delta=\delta_{\mathrm{useful}}=10^{-3}\) (same as P0 \(m_{\min}\)):
minimum worthwhile \(J\) improvement. Not \(\delta=0\).

- \(A<0\): harmful switch
- \(0\le A<\delta\): gray (not certificate-worthy)
- \(A\ge\delta\): useful switch

## Gates (aligned \(X\) only)

- **G-cert:** held empirical \(P(A\ge L)\ge 1-\varepsilon\).
- **G-safe:** \(n_{\mathrm{commit}}>0\) and
  \(\mathrm{Precision}=P(A>0\mid L>\delta)=1\).
  Zero commits is **vacuous**, not a pass.
- **G-recall:** \(P(L>\delta\mid A\ge\delta)\ge 0.25\).
- **G-VoI:** \(\mathbb{E}[A\,1\{L>\delta\}]>0\).

GO iff all four pass. Shuffle-\(X\) (seed 0, permute \(X\) inside each
split) is a **control**, not a tuning signal. Report
\(\mathrm{VoI}_{\mathrm{align}}>\mathrm{VoI}_{\mathrm{shuf}}\) as
G-ctrl (required).

If GO fails, record exactly one primary locus:
coverage / safety / recall / VoI / ctrl.
Coverage+safety with recall fail is the R6 boundary again:
safe abstention \(\neq\) positive certification.
