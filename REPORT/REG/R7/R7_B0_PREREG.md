# R7-B0 Preregistration — World-Model-Mediated Switch Certification

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R7/R7_A2_REPORT.md` (`GO=true`)  
A-series: **FROZEN** (A0–A2). No A3. No spline search.  
Does not change: \(L=\hat A-q\); \(L>\delta\Rightarrow\mathrm{commit}\);
\(\delta=10^{-3}\); \(\varepsilon=0.10\); plant \(F_{\max}=2.0\)  
Does not: \(J\)-weighted WM loss; advantage supervision; action
classifier; \(X\to a^\star\); distribution shift; neural probe

## Question

Is task-independent future prediction accurate enough that the **same**
one-sided certificate still yields positive realized \(\mathrm{VoI}\)?

\[
\boxed{
(h^S,X,a)\rightarrow\hat Y^{future}_a\rightarrow J(\hat Y_a)
\rightarrow\hat A_{\mathrm{WM}}\rightarrow L_{\mathrm{WM}}
\rightarrow\mathrm{commit/abstain}.
}
\]

The only new object is the source of \(\hat A\). Direct \(X\to\hat A\)
(A0–A2) is the frozen mechanism block, not a competitor gate.

## Plant and splits

Saturated actuator family (P1). Fresh \(\alpha\), disjoint from P0, P1,
A0, and A1/A2 grids. Split by \(\alpha\):

- **WM-fit** \(\{0.42,0.66,0.94,1.14,1.38,1.62,1.86,2.14,2.34,2.58,2.82,3.06,3.34,3.54\}\)
- **certificate-cal** \(\{0.46,0.74,0.98,1.22,1.42,1.66,1.94,2.18,2.42,2.62,2.86,3.14,3.38,3.62\}\)
- **held-test** \(\{0.54,0.62,0.78,0.82,1.02,1.06,1.26,1.34,1.54,1.58,1.74,1.82,1.98,2.02,2.22,2.26,2.46,2.54,2.74,2.78,2.94,3.02,3.18,3.22,3.42,3.46,3.66,3.74\}\)

\(A\) and \(J\) **do not enter WM-fit**. They enter only after \(f_{\mathrm{WM}}\)
is frozen, on cal and test (and on WM-fit solely for a labeled
diagnostic direct-\(f\), never for GO).

## World model (one, locked)

Hold \(h^S\) is macro-ambiguous (\(u=y=\dot y=0\)). It is recorded and
must stay near rest; it is **omitted from the OLS design** to avoid a
degenerate column. Varying inputs are \((X,a)\).

Predict position at five frozen probe indices
\(\{49,99,149,199,249\}\) (last = \(y_T\)):

\[
\hat y(t_k)=\phi(X,u)^\top\beta_k,\quad
\phi=(1,X,u,Xu,X^2).
\]

Independent OLS per \(t_k\). Loss is mean squared error on those
positions over WM-fit \((\alpha,a)\) pairs (both mid and cons). No \(J\).

Then

\[
\hat A_{\mathrm{WM}}=J(\hat Y_{\mathrm{default}})-J(\hat Y_{\mathrm{alt}}),
\]

with frozen \(J=(y_T-y^\star)^2+10^{-5}\overline{u^2}\) and known \(u\).

## Certificate (frozen)

\[
s_i=\hat A_{\mathrm{WM},i}-A_i,\quad
q=Q_{1-\varepsilon}(s),\quad
L_{\mathrm{WM}}=\hat A_{\mathrm{WM}}-q.
\]

\[
L_{\mathrm{WM}}>\delta\Rightarrow\mathrm{switch\ cons};\quad
\mathrm{else\ default}.
\]

Shuffle-\(X\) seed 0 inside each split.

## Gates (GO; same as A-series)

- G-cert: coverage \(\ge 0.90\)
- G-safe: non-vacuous precision \(=1\)
- G-recall: useful recall \(\ge 0.25\)
- G-VoI: \(\mathbb{E}[A\,1\{L_{\mathrm{WM}}>\delta\}]>0\)
- G-ctrl: \(\mathrm{VoI}_{\mathrm{align}}>\mathrm{VoI}_{\mathrm{shuf}}\)

Not a gate: matching A2 recall or \(q\).

## Pattern

| coverage+safety | recall | meaning |
|---|---|---|
| \(\checkmark\) | \(\checkmark\) | WM-mediated positive certification |
| \(\checkmark\) | \(\times\) | mediation preserves safety, destroys opportunity |
| \(\times\) | \(*\) | trajectory error not absorbed by one-sided \(q\) |

## Mediation cost (diagnostic)

On the **same** B0 cal/test, a cubic truncated-power spline \(X\to A\)
(A2 class, A2 knots) is fit on WM-fit \(\alpha\) **using \(A\)**. Compare
\(q\) and useful recall to the WM certificate. Historical A2 numbers
may be cited as a different-sample reference only.
