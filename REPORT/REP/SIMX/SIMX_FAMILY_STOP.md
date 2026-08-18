# SIM-X Family Stop

Date: 2026-08-17  
Status: **STOPPED** after X3  
Does not: dwell rescue of X3; retune \(F_{\mathrm{revoke}}\) / \(\sigma_{\mathrm{obs}}\) /
\(T_{\mathrm{dwell}}\) post hoc; unlock R10-C0; claim real physics;
upgrade X3 diagnostics into lifecycle PASS

## Formal ledger

```text
R9 toy-family      = STOP
SIM-X0             = PASS
SIM-X1             = PASS
SIM-X2             = PASS
SIM-X3             = FAIL (false_revoke)
SIM-X lifecycle    = NOT established

cross-engine action-conditioned observability = supported
cross-engine evidence-rate ordering           = supported
cross-engine persistent blindness             = not supported

R10-C0             = LOCKED / DEFERRED
real physics       = not claimed
REAL-LOG-A0        = A0-LIMITED (no force-auditable public log)
REAL-LOG-A1        = locked
third sim family   = do not open
```

\[
\boxed{\textbf{SIM-X family = STOP after X3}}
\]

## Four-layer outcome

\[
\boxed{
\begin{array}{ll}
\mathrm{X0}:& \text{MuJoCo force-space adapter closure}\quad\checkmark\\
\mathrm{X1}:& \text{oracle mismatch correctly exposed}\quad\checkmark\\
\mathrm{X2}:& \text{action-conditioned validity channel}\quad\checkmark\\
\mathrm{X3}:& \text{strong persistent lifecycle blindness}\quad\times
\end{array}}
\]

## What SIM-X supports

\[
\boxed{
\textbf{cross-engine robustness of action-conditioned validity observability}
}
\]

\[
\boxed{
\textbf{cross-engine robustness of policy-dependent evidence rate}
}
\]

X2 chain:

\[
a\to E_v\to|r|\to I(\mathcal V;Y\mid a).
\]

X3 secondary ordering (diagnostic only; not a PASS):

\[
I_{\mathrm{info}}>I_{\mathrm{cons}}>I_{\mathrm{high\_f}}
\;\Rightarrow\;
T_{\mathrm{revoke}}^{\mathrm{info}}
<T_{\mathrm{revoke}}^{\mathrm{cons}}
<T_{\mathrm{revoke}}^{\mathrm{high\_f}}
\]

(medians \(0.002<0.070<0.95\,\mathrm{s}\)) and matching \(S_{\mathrm{stale}}\)
order. May write:

\[
\boxed{\text{action-conditioned information controls evidence-acquisition rate}}
\]

May **not** write policy-induced **lifecycle** blindness.

## What SIM-X does not support

\[
\boxed{
\textbf{cross-engine robustness of stale-license lifecycle blindness}
}
\]

X3 formal:

\[
\texttt{SIM-X3 FAIL},\qquad\texttt{pattern=false\_revoke}.
\]

### Two independent failures

1. **G1 — threshold fragility.** Stay-valid recovers
   (\(b_T\approx 1\), \(S_{\mathrm{stale}}\approx 0.999\)) but existential
   \(F_{\mathrm{revoke}}=\mathbf{1}[\exists t:b_t<0.5]\) fires at
   \(P=0.045>0.01\). Diagnosis:

   \[
   \boxed{
   \text{instantaneous threshold crossing is fragile under sequential noisy belief}
   }
   \]

   Do not rewrite as “license retention succeeded.”

2. **G4 — lifecycle identification.** X2
   \(I_{\mathrm{high\_f}}=3.73\times10^{-5}\) bit/sample is near-blind
   per step, yet median \(T_{\mathrm{revoke}}^{\mathrm{high\_f}}=0.95\,\mathrm{s}\)
   and \(\approx 98.6\%\) invalid runs revoke within \(4\,\mathrm{s}\):

   \[
   \boxed{
   \text{near-zero instantaneous information}
   \not\Rightarrow
   \text{persistent epistemic blindness}
   }
   \]

   \[
   \boxed{
   \text{weak evidence}\times\text{enough observations}
   \rightarrow
   \text{eventual identification}
   }
   \]

Dwell would address G1 semantics, **not** G4. Choosing
\(T_{\mathrm{dwell}}\gtrsim 1\,\mathrm{s}\) to keep `high_f` licensed
would be post-hoc persistence manufacturing. **No dwell rescue cell now.**

## Correction to R9 strength language

R9-B0 toy: strong form — channel effectively closed under cons, so
normal operation cannot revoke.

MuJoCo SIM-X: weaker form —

\[
\boxed{
I(\mathcal V;Y\mid a)\text{ can become very small without becoming zero.}
}
\]

Hence:

\[
\boxed{
\textbf{policy-induced epistemic attenuation}
\text{ is more robust than }
\textbf{policy-induced epistemic closure}.
}
\]

Policy → slower epistemic refresh: cross-engine stable.  
Policy → permanent epistemic blindness: **not** established here.

## Future dwell (only as a new question)

If ever reopened, preregister as a **new** problem:

> How should a deployment permission map a noisy posterior into a
> stable discrete state?

Not: “How do we make SIM-X3 pass?”

## R10 / REAL-LOG

R10-C0 remains **LOCKED/DEFERRED**. REAL-LOG-A0 = `A0-LIMITED` (no
force-A1). Public demos ≠ calibrated residual chain. H1/CAD deferred.

## What to do next

**VIS-X** (not a third physics family; not REAL-LOG-A1):
`REPORT/REG/VISX/VISX_PREREG.md`. **VIS-X0–X3 PASS**; **VIS-EXT0 PASS**
(`REPORT/REG/VISX/VISEXT0_PREREG.md`: robosuite Door; raise visual
realism only; RoboCasa locked pending EXT0). R10 still locked.

Paper synthesis remains:
`REPORT/REP/PAPER/APRWM_R7_R9_SIMX_SYNTHESIS.md`.
R10-C0 stays locked until hardware.
