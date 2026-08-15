# R1-RS1A.1 Preregistration — Compositional Inadequacy Evidence

Date: 2026-08-15  
Status: **FROZEN** — formal completed; see `R1_RS1A1_REPORT.md` (`RS1A.1_GO=false`; C1-L ≠ support-coupling)

## Placement

\[
\boxed{RS1A \rightarrow RS1A.1 \rightarrow RS1B \rightarrow RS1C}
\]

RS1A established:

- \(D_0\) strong on in-library structure, blind on C2
- \(D_3\) (via support) recovers C2 but degrades C1 ranking
- naive persistence \(D_2\) is falsified for this problem
- the open question is **how heterogeneous evidence should compose**

RS1B stays locked until RS1A.1 unlocks it.

## Scientific question

\[
\boxed{
\text{Can support evidence improve outside-library detection
without degrading weak in-library structural detection?}
}
\]

Hypothesis (mechanism, not implementation):

> Support evidence should be additive/conditional with normalized residual
> evidence, not coupled through a harmful temporal accumulator.

## Detectors

Primary comparison:

| ID | Formula | Role |
|----|---------|------|
| \(D_0\) | \(\mathrm{RMS}(r_\perp)\) | magnitude baseline |
| \(D_1\) | \(\mathrm{RMS}(z)\), \(z=\|r_\perp\|/(\sigma+\epsilon)\) | normalized residual |
| \(D_3\) | \(D_2+\gamma_0 S(u)\), \(\gamma_0=1\) (frozen RS1A) | persistence+support (reference) |
| \(D_3'\) | \(D_1+\gamma S(u)\) | compositional candidate |

Diagnostic (not required for GO):

\[
D_g=
\begin{cases}
D_1, & u=0\\
D_1+\gamma S(u), & u>0
\end{cases}
\qquad
S(u)=-\log(1-u+\epsilon)
\]

\(D_2\) may be reported for continuity with RS1A but is **not** a primary method.

Frozen (unchanged from RS1A):

- residual windows, \(\sigma_{\mathrm{pred}}\), support audit (\(n=32\))
- no revision / acceptance / passivity / H32
- no MuJoCo-specific operators

## \(\gamma\) selection (dev-only)

Candidate set (frozen):

\[
\gamma\in\{0.25,0.5,1,2,4\}
\]

Development seeds (independent of formal):

\[
\{9301,9311,9321\}
\]

Selection objective (once, then freeze):

\[
\hat\gamma
=
\arg\max_{\gamma}
\min\bigl(R_{\mathrm{C1\text{-}L}}(D_3'),\,R_{\mathrm{C2}}(D_3')\bigr)
\]

subject to C0 FPR \(\le 1\%\) at the matched operating point.

**Forbidden:** sweeping \(\gamma\) on formal held-out data; optimizing recovery/H32.

## Formal matrix (held-out)

\[
\{9401,9411,9421,9431,9441\}
\times
\{C0,C1\text{-}L,C1\text{-}H,C2\text{-}latch\}
\times
8\ \mathrm{probes}
=
160
\]

CNEG diagnostic-only (+40). Smoke: seed `8961` × 5 × P1.

RS1A formal seeds `{9201…9241}` are **not** reused for confirmatory claims.

## Hard gates (not pooled AUROC)

At the same C0 FPR \(\le 1\%\) operating point:

\[
\boxed{R_{\mathrm{C1\text{-}L}}(D_3') > R_{\mathrm{C1\text{-}L}}(D_0)}
\]

\[
\boxed{R_{\mathrm{C2}}(D_3') > R_{\mathrm{C2}}(D_0)}
\]

Non-degradation:

\[
\boxed{
AUROC_{\mathrm{C1}}(D_3')
\ge
AUROC_{\mathrm{C1}}(D_0)-\delta
},\qquad \delta=0.02
\]

\[
RS1A.1\_GO
=
\text{both recall gates}
\land
\text{non-degradation}
\]

Secondary reports: \(D_1\), \(D_3\), \(D_g\) ablations; seed-level summaries (\(n=5\)).

## Unlock

On GO: freeze compositional evidence class; unlock **R1-RS1B** drafting/run.  
On fail with C2↑ but C1-L flat: conclude C1-L is **not** a support-coupling problem; next study weak-signal detector (local SNR / matched statistic / proper LR), not more composition tweaks.  
Do **not** open RS2.
