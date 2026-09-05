# R1-RS1B.1 Preregistration — Support-Calibrated Long-Horizon Admissibility

Date: 2026-08-16  
Status: **FROZEN; formal completed** — see `REPORT/REP/R1/R1_RS1B1_REPORT.md` (`RS1B.1_GO=false`)  
Depends on: `REPORT/REP/R1/R1_RS1B_REPORT.md` (`RS1B_GO=false`; fail axis = modeled-support retention)  
Frozen implementation: `aprwm_v0/r1_rs1b1.py`

## Placement

\[
\boxed{
RS1B \rightarrow RS1B.1 \rightarrow RS1C\ (\mathrm{locked}) \rightarrow RS2\ (\mathrm{locked})
}
\]

RS1B showed: on \(\mathcal P_{\mathrm{rev}}\), passivity and H32 *utility* transfer,
but the binary modeled-support predicate (`support_code=0` on every H32 step)
is too strict relative to RMSE. **Do not** turn that confirmatory failure into

\[
I_{\mathrm{exit}}<\tau
\]

as a post-hoc hard filter. That would encode the failure, not explain it.

RS1B_GO, detector, VoI, \(C_{\mathrm{tol}}\), damping, expansion, and H32
stability definitions remain frozen. This stage does not reopen RS1A.

## Scientific question

\[
\boxed{
\text{What does leaving modeled support actually mean for revision reliability?}
}
\]

Operationally:

\[
\boxed{
\text{At what support-departure does predictive reliability start to degrade?}
}
\]

This is **support-risk calibration**, the dual of RS1A.4's mismatch-consequence
calibration:

| Stage | Calibrates |
|-------|------------|
| RS1A.4 | mismatch consequence \(C\) vs detectability |
| RS1B.1 | model-support departure \(d_{\mathcal S}\) vs long-horizon risk |

## Frozen non-goals

- Adding `support-exit < τ` to the RS1B accept rule  
- Changing `stable_h32` or lowering the 0.90 bar  
- Reopening detector / VoI / \(C_{\mathrm{tol}}\)  
- Contact / RS2  
- Treating RS1B's 15 accepts as this stage's confirmatory sample

## Intake (unchanged)

Same frozen RS1A.5 policy and RS1B revision object, **new held-out seeds**:

\[
\{9951,9961,9971,9981,9991\}
\]

\[
5\times\{\alpha=-0.12,-0.18,-0.24\}\times\{1.5,2.0\}A_0
=
\boxed{30\ \mathrm{episodes}}
\]

Smoke: seed `9021` × \(\alpha=-0.24\) × \(1.5A_0\) (plumbing only; no GO).

Analysis population = **dynamics-filter accepted** revise-worthy revisions
(same accept freeze *before* blind H32 as RS1B). Binary `stable_h32` is
recorded as a diagnostic, never as an accept input, and never as the
primary endpoint.

## Distance to modeled support

Modeled interior is the RS1B validity region:

\[
q\in[q_{\min}+m,\,q_{\max}-m],\qquad m=0.05.
\]

\[
d_{\mathcal S}(q)
=
\frac{
\max\bigl(0,\,q_{\min}+m-q,\,q-(q_{\max}-m)\bigr)
}{
q_{\max}-q_{\min}+\epsilon
}
\]

so \(d_{\mathcal S}=0\) inside modeled support and grows continuously outside.

On each H32 query trajectory \(\{q_t\}_{t=1}^{H}\):

\[
D_{\mathrm{exit}}=\max_t d_{\mathcal S}(q_t)
\qquad
I_{\mathrm{exit}}=\frac1H\sum_t\mathbf 1[d_{\mathcal S}(q_t)>0].
\]

Fit-box excursion (RS1B's `support_exit_fraction`) is a **secondary** channel,
not the primary \(d_{\mathcal S}\). Primary matches the geometry that made
`stable_h32` fail.

## Risk (not the binary support label)

Per H32 query, after accept freeze:

\[
\Delta\mathrm{RMSE}
=
\mathrm{RMSE}^{\mathrm{no\text{-}rev}}
-
\mathrm{RMSE}^{\mathrm{R0.6}}
\]

(same scaled \((q,v)\) terminal error as RS1B).

\[
\mathrm{harmful}
=
(\text{not finite})
\lor
(\Delta\mathrm{RMSE}<0).
\]

**Hypothesis to test, not assume:**

\[
\boxed{
I_{\mathrm{exit}}>0
\not\Rightarrow
\mathrm{harmful}
}
\]

Support is an **epistemic confidence boundary**, not automatically a physical
safety boundary.

## Primary endpoints

Query-level pairs \((D_{\mathrm{exit}},\mathrm{RMSE}^{\mathrm{rev}})\) within
each seed, then seed-mean statistics (\(n=5\)).

| Endpoint | Frozen criterion |
|----------|------------------|
| Non-implication | among accepted queries with \(I_{\mathrm{exit}}>0\), mean \(\Delta\mathrm{RMSE}>0\) |
| Distance–error association | seed-mean Spearman\((D_{\mathrm{exit}},\mathrm{RMSE}^{\mathrm{rev}})>0\) |
| Continuous \(>\) binary | \(\lvert\rho_D\rvert > \lvert\rho_I\rvert\) with \(\mathrm{RMSE}^{\mathrm{rev}}\) (seed-mean); if \(\rho_I\) undefined, \(\rho_D>0\) suffices |
| Region split (formula frozen) | see below; mean \(\mathrm{RMSE}^{\mathrm{rev}}\) in `unsupported-risky` \(>\) `supported` when both have \(\ge 3\) queries |

### \(D_{\mathrm{tol}}\) formula (a priori, like \(C_{\mathrm{tol}}\))

On accepted queries of the *current* matrix, not RS1B:

\[
D_{\mathrm{tol}}
=
\mathrm{median}\{D_{\mathrm{exit}}:I_{\mathrm{exit}}=0\}
+
\tfrac12
\mathrm{median}\{D_{\mathrm{exit}}:D_{\mathrm{exit}}>0\}.
\]

Empty zero set \(\Rightarrow\) median \(0\). Empty positive set \(\Rightarrow\)
calibration vacuous, GO false.

Descriptive regions (not an accept rule):

\[
\boxed{
\begin{aligned}
\mathrm{supported}
&\colon D_{\mathrm{exit}}=0\\
\mathrm{extrapolative\text{-}but\text{-}calibrated}
&\colon 0<D_{\mathrm{exit}}\le D_{\mathrm{tol}}\\
\mathrm{unsupported\text{-}risky}
&\colon D_{\mathrm{exit}}>D_{\mathrm{tol}}
\end{aligned}
}
\]

## GO

\[
\begin{aligned}
RS1B.1\_GO
&=
(\#\text{accepted}>0)\\
&\land
(\text{non-implication})\\
&\land
(\text{seed-mean Spearman}(D_{\mathrm{exit}},\mathrm{RMSE}^{\mathrm{rev}})>0)\\
&\land
(\text{continuous beats binary, as above})
\end{aligned}
\]

The three-region RMSE ordering is **reported**; it is a hard gate only when
both `supported` and `unsupported-risky` have \(\ge 3\) queries.

## Unlock

On GO: RS1C may be *drafted* still without installing a support-exit hard
filter; RS2 stays locked until RS1C.

On fail: classify whether \(d_{\mathcal S}\) is uninformative, or whether risk
is dominated by finite/catastrophic events rather than RMSE. Do not add a
binary support hard filter and do not reopen RS1A.
