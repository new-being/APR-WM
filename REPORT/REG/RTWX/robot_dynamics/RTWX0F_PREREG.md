# RTWX-X0F Preregistration — Force-Channel / Clock Audit

Date: 2026-08-27
Status: **FROZEN**; **formal RAN 2026-08-27**;
**`force_channel_unresolved`**. Report: `REPORT/REP/RTWX/RTWX0F_REPORT.md`.
Depends on: RTWX-X0S RAN `timing_mismatch`
  (`REPORT/REP/RTWX/RTWX0S_REPORT.md`)
Does not: train nets; estimate \(\phi\); add M0–M4 terms; capacity \(R_P\);
TASK-XL; RGB; R10

## One question

\[
\boxed{\textbf{RTWX-X0F — Force-Channel / Clock Audit}}
\]

\[
\boxed{
\text{Which readable force channel, at which time, actually matches
the integrator's }qacc\text{?}
}
\]

X0S already excluded: broken gates (numpy plant PASS) and a static
cabinet (\(\mathrm{rms}(\ddot q)\sim 8\)). Remaining:

\[
\boxed{
\text{the two sides of the dynamics equation are not yet on the same
clock / same generalized-force channel.}
}
\]

Adding \(b,\mu,g(q)\) or fitting \(\phi\) fits a **wrong**
\(\tau\leftrightarrow\ddot q\) pairing.

## Log (every `scene.step`, no macro-smear)

Pre: \(q^{pre},\dot q^{pre},qacc^{pre},qf^{pre}\).
Set \(\tau_{\mathrm{cmd}}\). Then \(qf^{post\_set}\).
Step. Post: \(q^{post},\dot q^{post},qacc^{post},qf^{post}\).
Also \(\tau_{\mathrm{passive}}=\mathrm{compute\_passive\_force}()\) and
any ext/constraint API if present (else log unread).

Candidates:

\[
\tau^{(1)}=\tau_{\mathrm{cmd}},\;
\tau^{(2)}=qf^{post\_set},\;
\tau^{(3)}=\tau_{\mathrm{cmd}}+\tau_{\mathrm{passive}},\;
\tau^{(4)}=qf^{post},\;
\tau^{(5)}=\tau_{\mathrm{cmd}}+\tau_{\mathrm{passive}}+\tau_{\mathrm{ext}}.
\]

\(\ddot q\) sources: \(qacc^{pre}\), \(qacc^{post}\),
\((\dot q^{post}-\dot q^{pre})/\Delta t\).

Lag grid \(\ell\in\{-2,-1,0,+1,+2\}\). Seek \((k^\star,\ell^\star)\)
with stable per-DoF correlation.

## Gates

Frozen in header **before** collect: \(\rho_{\min}=0.3\).

**G0:** \(\min_i |\mathrm{corr}(\tau^{(k^\star)}_i,\ddot q_i)|\ge 0.3\)
on train; **same** \((k^\star,\ell^\star,\ddot q\text{-src})\) on
held-out episodes.

If every candidate fails: **`force_channel_unresolved`** — wrapper does
not expose the integrator force; **unsuitable for physics-capacity**.

**G1 (only if G0):** oracle \(M\ddot q+h\approx\tau_{\mathrm{eff}}\)
(simulator/pinocchio truth, **no** \(\phi\) fit) beats identity.
Then **`force_time_interface_closed`**.

Unlock order:

\[
\boxed{
\text{X0F}\rightarrow\text{oracle closure}\rightarrow\phi\text{-ID}
\rightarrow\text{capacity}
}
\]

Not more X0S structure terms.

## Claims ceiling

May name the winning channel/lag or declare unresolved. May **not**
claim capacity, \(\phi\)-ID, or TASK-XL.
