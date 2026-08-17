# R9-B0 Preregistration — Passive Unidirectional License Revocation

Date: 2026-08-17  
Status: **RAN**; **`R9_B0_GO=false`**; ordinary licensed-task evidence insufficient  
Depends on: `REPORT/REP/R9/R9_A0_REPORT.md` (`GO=true`)  
Does not: re-probe; bidirectional \(b\); decay-to-zero without evidence;
\(F_{\max}\) estimator; \(A\)/\(J\)/\(a^\star\) in the score; R7 WM/\(q\)
update; tune \(\tau_{\mathrm{revoke}}\) on invalid regret

## Question

Can ordinary task interaction revoke a **stale** validity license
before that license causes **repeated** harmful commits?

\[
\boxed{
\text{UNLICENSED}
\xrightarrow{\text{calibration}}
\text{LICENSED}
\xrightarrow{\text{passive change evidence}}
\text{UNLICENSED.}
}
\]

After revoke: default only. No automatic reacquisition (that is B1).

Not: detect that \(F_{\max}\) changed. \(2.0\to 2.2\) is a change that
must **keep** the license.

## Information structure

No pre-action sensing after calibration. The first post-shift task
may be exposed. B0 does **not** require zero first harmful commit.

\[
\boxed{
\text{first post-shift task supplies }r_k
\to
\text{revoke before subsequent tasks.}
}
\]

## Frozen protocol

Held commit-set \(\alpha\) (B0 test, \(L_{\mathrm{WM}}>\delta\)).
Calibrate once at \(F_{\max}=2.0\) (frozen A0 logistic, \(\tau_{\mathcal V}=0.5\)).
Then \(K=6\) fresh episodes, cut after task \(k_{\mathrm{cut}}=2\):

| name | pre (\(k\le 2\)) | post (\(k>2\)) |
|---|---|---|
| stay | \(2.0\) | \(2.0\) |
| benign | \(2.0\) | \(2.2\) |
| invalid | \(2.0\) | \(1.5\) |

Comparator: A0 permanent persistence (same calibration, never revoke).

## Passive score (task-independent)

Executed \(u\in\{u_{\mathrm{default}},u_{\mathrm{cons}}\}\). Frozen WM
knots vs observed \(y\):

\[
r_k=\mathrm{RMSE}\bigl(Y^{\mathrm{obs}}_{\mathrm{knots}},\hat Y^{\mathrm{WM}}(X,u)\bigr).
\]

No \(A\), \(J\), \(a^\star\).

CUSUM, coefficients frozen from **development ID only**
(B0 fit∪cal, \(L>\delta\), \(F_{\max}=2.0\), executed cons):

\[
s(r)=\frac{r-\mu}{\sigma}-0.5,\qquad
E_k=\max(0,E_{k-1}+s(r_k)),\qquad
\tau_{\mathrm{revoke}}=4.
\]

\(\mu,\sigma\) are residual mean/std on that development ID set.
\(\sigma\leftarrow\max(\sigma,10^{-8})\). \(0.5\) and \(4\) are textbook
CUSUM constants, not fit on \(F_{\max}=1.5\) or held NetVoI.

\[
b_{k+1}=\begin{cases}b_k & E_k<4\\ 0 & E_k\ge 4.\end{cases}
\]

Unidirectional. \(m=1\): revoke is on time if it fires after the first
post-shift observation.

## Gates (all required)

- **G-stable**: stay blocks, \(P(\text{license retained through }K=6)\ge 0.90\)
- **G-benign**: \(2.0\to 2.2\), same retain floor \(0.90\)
- **G-revoke**: \(2.0\to 1.5\), \(P(K_{\mathrm{revoke}}\le 1)\ge 0.80\)
- **G-harm**: \(N_{\mathrm{harmful}}^{\mathrm{revoke}}<N_{\mathrm{harmful}}^{\mathrm{persist}}\)
  on invalid blocks; report first vs repeat
- **G-NetVoI**: mean block \(\mathrm{NetVoI}_{\mathrm{revoke}}>\mathrm{NetVoI}_{\mathrm{persist}}\)
  over held commit-set \(\times\) three sequences

Failure loci (do not retune \(h\) on them):

- benign also revokes \(\to\) generic change detection
- invalid never revokes \(\to\) ordinary task insufficient
- revoke only after many harmful tasks \(\to\) evidence too late (opens B1)

## After GO

Unlock R9-B1: when revoked, when to pay to reacquire. Not this stage.
