# R8-P1 Preregistration — State-Neutral Validity Probe

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R8/R8_P0_REPORT.md` (locus `probe_cost`; A0 locked)  
Does not: train a detector; \(\tau\); \(\pi\); WM update; reset protocol;
sweep \(T\) or \(u_0\); R8-A0

## Question

Does replacing the P0 one-way pulse by a **fixed** zero-moment waveform
keep validity evidence while returning the state, so same-episode
\(C_{\mathrm{down}}\) no longer eats \(|A|\)?

One scientific variable:

\[
\text{non-returning excitation}
\rightarrow
\text{state-neutral excitation}.
\]

## Frozen (inherited)

Classes: ID \(F_{\max}=2.0\), benign \(2.2\), invalid \(1.5\).  
B0 commit-set \(\alpha\). \(\delta\), \(q\), \(f_{\mathrm{WM}}\) frozen.  
\(u_0=1.0\), \(T=0.10\,\mathrm{s}\) (\(n=50\)), \(\mathrm{d}t=0.002\).

## Frozen waveform (no search)

Signs \([+,-,-,+]\). Integer phases summing to 50, palindromic durations:

\[
n=(13,12,12,13).
\]

Open-loop predictor of \(a^{\mathrm{epi}}\): the **frozen ID plant**
(\(F_{\max}=2.0\)), because B0 OLS is a constant-\(u\) knot map, not a
variable-\(u\) stepper. \(S_{\mathrm{epi}}\) still uses only \(Y\) vs
\(\hat Y\), never \(A\) or \(J\):

\[
S_{\mathrm{epi}}=\mathrm{RMSE}_t\bigl(y(t)-\hat y_{\mathrm{ID}}(t)\bigr)
\quad t\in[0,T].
\]

\[
D_{\mathrm{terminal}}=\sqrt{y_T^2+\dot y_T^2}.
\]

\(C_{\mathrm{down}}=J(\text{epi then default }0.50\,\mathrm{s})-J(\text{default from rest})\).

## Gates (all required)

- **G0** \(\max D(h^S)<0.05\)
- **G1** on commit-set:
  \(\min S_{\mathrm{invalid}}-\max(S_{\mathrm{ID}},S_{\mathrm{benign}})\ge 10^{-3}\)
- **G2** probe before mid/cons
- **G-return** mean \(D_{\mathrm{terminal}}\) on ID commit-set \(\le 2.2\times10^{-3}\)
  (one quarter of P0 one-way \(y_T\approx 8.8\times10^{-3}\))
- **G3** \(\overline{C}_{\mathrm{down}}/\overline{|A|}_{\mathrm{commit,ID}}\le 1\)

If G1 fails while return/cost pass: neutrality vs observability conflict.  
If G1 holds and return or G3 fail: observable but not cheaply in-episode.  
No \(T\)/\(u_0\) sweep. No reset protocol. A0 stays locked unless all
five pass.
