# R7-P0 Preregistration — Fresh Switchability Preflight

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R6/R6_REALIZATION_FREEZE.md`  
Does not implement: \(\hat A\), LCB, commit rule, learner  
Does not reuse: self-stress \(\lambda\); contact pads; D0 \(\mathcal A\) with three actions  
Unlocks R7-A0 **only if** all five gates pass

## Family

Hidden actuator effectiveness \(\alpha\) with

\[
\ddot y=\alpha u-c\dot y.
\]

Hold at \(t_\star\): \(u=y=\dot y=0\), so \(h^{S}(\alpha)\) must be
macro-ambiguous. \(X(\alpha)\) is a local calibration reading, **not**
in \(h^{S}\). Sensing is not the scientific object.

Frozen actions (two only):

\[
a_{\mathrm{default}}=\mathrm{mid}=1.0,\qquad
a_{\mathrm{alt}}=\mathrm{cons}=0.4.
\]

\[
A(\alpha)=J(a_{\mathrm{default}},\alpha)-J(a_{\mathrm{alt}},\alpha).
\]

\(A>0\) iff switching to cons is beneficial. No global consequence
scalar \(C\) besides this frozen signed advantage.

## Frozen constants

- \(c=4\), \(\mathrm{d}t=0.002\), hold \(0.30\,\mathrm{s}\), probe \(0.50\,\mathrm{s}\)
- \(y^\star=0.10\), \(J=(y_T-y^\star)^2+10^{-5}\overline{u^2}\)
- \(\alpha\in\{0.6,1.0,1.4,1.8,2.2,2.6,3.0,3.4\}\)
- \(X=0.25\alpha\) (calibration volts); excluded from \(h^{S}=(y,\dot y,\ddot y,u)\) on the hold window
- \(\varepsilon_h=0.05\), \(\rho_{\min}=0.80\), \(m_{\min}=10^{-3}\), \(N_{\pm}\ge 3\)

## Gates (all required)

- **G0** \(\max_\alpha D(h^{S}_\alpha,h^{S}_{\alpha_0})<\varepsilon_h\)
- **G1** \(|\rho(\alpha,X)|>\rho_{\min}\)
- **G2** \(\exists\alpha: A>0\) and \(\exists\alpha: A<0\)
- **G3** at least \(N_{\pm}\) regimes on **each** sign with \(|A|>m_{\min}\)
- **G4** both signs nonempty after G3 (balanced coverage; not 1 vs 20)

No certificate. If any gate fails, R7-A0 stays locked.
