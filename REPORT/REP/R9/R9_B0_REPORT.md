# R9-B0 Report — Passive Unidirectional License Revocation

Date: 2026-08-17  
Status: **`R9_B0_GO=false`**; pattern **passive_evidence_missing_or_late**;
primary locus **false_revoke_id**; R9-B1 **locked**  
Prereg: `REPORT/REG/R9/R9_B0_PREREG.md`  
Artifacts: `runs/r9_b0/formal/summary.json`

## Question

Can ordinary task interaction revoke a stale validity license before
repeated harmful commits, without a new probe?

CUSUM on frozen-WM knot RMSE. \(\mu,\sigma\) from development ID
commit-set only. Textbook \(k=0.5\), \(h=4\). Not fit on \(F_{\max}=1.5\).

## Gates

| Gate | stay \(2\!\to\!2\) | benign \(2\!\to\!2.2\) | invalid \(2\!\to\!1.5\) |
|---|---|---|---|
| G-stable retain \(\ge 0.90\) | **× \(0.714\)** | — | — |
| G-benign retain \(\ge 0.90\) | — | **× \(0.714\)** | — |
| G-revoke \(K_{\mathrm{revoke}}\le 1\) \(\ge 0.80\) | — | — | **× \(0.143\)** |
| G-harm \(N_{\mathrm{h}}^{\mathrm{rev}}<N_{\mathrm{h}}^{\mathrm{per}}\) | — | — | **× \(12=12\)** |
| G-NetVoI revoke \(>\) persist | pooled **×** \(0.00601<0.00717\) | | |

\[
\boxed{\texttt{R9\_B0\_GO}=\text{false}}
\]

Two held \(\alpha\) (\(3.66\), \(3.74\)) accumulate CUSUM on **pre-cut
ID cons rollouts** (WM extrapolation vs development \(\alpha\le 3.54\)).
The other five never revoke, including after \(F_{\max}=1.5\).

Invalid harmful commits: persist \(12\), revoke \(12\) (first \(3\),
repeat \(9\)). Revoke does not cut repeated harm.

## Why passive surveillance is blind here

While licensed, \(\pi\) plays \(u_{\mathrm{cons}}=0.4\). On the
commit-set \(\alpha\approx 3.0\)–\(3.74\):

\[
\alpha\,u_{\mathrm{cons}}\le 1.50\le F_{\max}\in\{1.5,2.0,2.2\}.
\]

So cons is **unsaturated for every class**. Observed knots, and therefore
\(r_k\), are identical across stay / benign / invalid until the policy
switches to \(u=1\). After a (false) revoke, default \(u=1\) does
separate \(r\) (invalid \(\approx 0.028\) vs ID \(\approx 0.007\)), but
that is not a licensed-task signal.

\[
\boxed{
\text{the licensed action does not excite the saturation channel
that makes applicability observable.}
}
\]

This is information structure, not a small CUSUM \(h\). Do not retune
\(h\) on invalid regret.

## Reading

Not generic OOD detection of \(2.2\) vs \(2.0\) (those \(r_k\) also
match under cons). Not “almost; need a few more harmful tasks”:
five of seven invalid blocks never revoke.

\[
\boxed{
\text{ordinary task interaction under the licensed conservative
action is insufficient validity evidence.}
}
\]

\[
\boxed{\textbf{policy-induced epistemic blindness}}
\]

\[
u=1\to\text{observable}\to\text{license cons}\to u=0.4
\to\text{unobservable.}
\]

Formal primary locus remains **false_revoke_id** (two held \(\alpha\)
revoke on ID before the cut). The deeper bound: even without those
false revokes, five of seven invalid blocks never revoke, because
\(\alpha u_{\mathrm{cons}}\le 1.50\) so \(\{1.5,2.0,2.2\}\) share one
unsaturated cons trajectory.

**R9-B1-P0** (`REPORT/REP/R9/R9_B1_P0_REPORT.md`) tests finite
certificate age \(M=2\) with the frozen \(u=1\) probe, not a new CUSUM.
