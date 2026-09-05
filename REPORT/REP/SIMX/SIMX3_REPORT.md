# SIM-X3 Report — Persistent Validity Lifecycle

Date: 2026-08-17  
Status: **`sim_x3_passed=false`**; pattern **`false_revoke`**; does **not**
unlock a real-physics claim; R10-C0 untouched  
Depends on: `REPORT/REG/SIMX/SIMX3_PREREG.md`,
`REPORT/REP/SIMX/SIMX2_REPORT.md`

## Question

Does X2’s per-sample \(I_a\) ordering propagate into persistent
\(b(\mathcal V)\) lifecycle ordering under a **canonical** license
\(b_0(\mathcal V)=0.95\), three-hypothesis analytic Bayes, and the same
\(\sigma_{\mathrm{obs}}=0.01\) synthetic channel?

## Formal matrix

\[
3\ \mathrm{actions}\times 3\ \mathrm{plant}\times 4\ \mathrm{phases}
\times 100\ \mathrm{noise}=3600\ \mathrm{cells}.
\]

Actions: `info`, `cons`, `high_f`. Canonical
\(w_0=(0.025,0.95,0.025)\). No detector / CUSUM / NetVoI.

## Gates

| gate | result | detail |
|---|---|---|
| G0 X2 carryover | **pass** | max \(E_{\mathrm{oracle}}=5.9\times10^{-15}\) |
| G1 valid retention | **fail** | \(P(F_{\mathrm{revoke}})=0.045>0.01\) |
| G2 info revoke | **pass** | \(P(T\le 4\,\mathrm{s}\mid\mathrm{info},\mathrm{inv})=1.0\) |
| G3 stale ordering | **pass** | \(S_{\mathrm{info}}<S_{\mathrm{cons}}<S_{\mathrm{high\_f}}\) on both shifts + aggregate |
| G4 high_f persist | **fail** | \(P(\mathrm{censored}\mid\mathrm{high\_f},\mathrm{inv})=0.014<0.90\) |

\[
\boxed{\texttt{sim\_x3\_passed=false}}
\qquad
\boxed{\texttt{pattern=false\_revoke}}
\]

## Registered interpretation (Pattern C)

Stay-valid licenses **existentially** dip below \(0.5\) often enough to
break G1 (`cons` \(7.5\%\), `high_f` \(5\%\), `info` \(1\%\)). Mean
terminal \(b_T\) on stay is still \(\approx 1\) and mean \(S_{\mathrm{stale}}\)
\(\approx 0.999\): dips are brief, then recover. Under the frozen
definition \(F_{\mathrm{revoke}}=\mathbf{1}[\exists t:\,b_t<0.5]\), this
is still Pattern C:

> treat as belief/update / threshold-sensitivity issue; **do not**
> celebrate invalid revoke as lifecycle blindness.

Do **not** retune \(\sigma_{\mathrm{obs}}\), \(0.5\), or gate numbers to
rescue PASS.

## Secondary diagnostics (not PASS claims)

Aggregate invalid \(S_{\mathrm{stale}}\):

| action | \(S_{\mathrm{stale}}\) | median \(T_{\mathrm{revoke}}\) (uncensored) |
|---|---:|---:|
| info | \(0.00103\) | \(0.002\,\mathrm{s}\) |
| cons | \(0.0288\) | \(0.070\,\mathrm{s}\) |
| high_f | \(0.369\) | \(0.95\,\mathrm{s}\) |

Ordering matches X2 \(I_a\). But `high_f` still revokes on \(\approx 98.6\%\)
of invalid cells within \(4\,\mathrm{s}\) → on this horizon,
**accumulation recovers** (Pattern B *would* apply if G1 had passed).
That is exactly X3’s point:

\[
\text{per-step near-blind}\neq\text{lifecycle-blind}.
\]

## Claims

**Not** allowed: cross-engine lifecycle robustness; real physics;
R9-strong stale license on MuJoCo at this scale.

**Allowed:** formal negative + diagnostic — under canonical license and
existential revoke, G1/G4 fail; stale **ordering** still tracks the
channel, while high_f is not lifecycle-closed at \(4\,\mathrm{s}\).

## Artifacts

`runs/sim_x3/formal/summary.json`, `cells.jsonl`, `exemplars/`.

## Next

**SIM-X family STOP** (`REPORT/REP/SIMX/SIMX_FAMILY_STOP.md`).
No dwell rescue now. G1 and G4 are different failures; fixing
existential threshold semantics would not manufacture high_f
lifecycle blindness. R10-C0 stays locked/deferred.
