# R7-P1 Preregistration — Misspecification Preflight

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R7/R7_A0_REPORT.md` (`GO=true`)  
Does not implement: certificate, \(L\), commit, model upgrade  
Does not retune: \(F_{\max}\) or \(E_{\min}\) after seeing \(A(\alpha)\)  
Unlocks R7-A1 **only if** switchability **and** misspecification gates pass

## Family

Same hold / \(X\propto\alpha\) / two-action \(J\) as P0, with saturation:

\[
\ddot y=\operatorname{sat}(\alpha u;F_{\max})-c\dot y,
\qquad
\operatorname{sat}(z;F_{\max})=\mathrm{clip}(z,-F_{\max},F_{\max}).
\]

Hold \(u=y=\dot y=0\) still implies \(D(h^{S})=0\). Sensing stays easy.
The point is **imperfect** degree-2 \(A(X)\), not a new modality.

## Frozen constants

- \(F_{\max}=2.0\) (mid \(u=1\) clips for \(\alpha>2\); cons \(u=0.4\) does not clip on this grid)
- \(\alpha\in\{0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,2.9,3.2,3.5,3.8\}\)
- \(J\), \(c\), hold/probe, \(X=0.25\alpha\), \(m_{\min}=10^{-3}\), \(N_{\pm}\ge 3\) as P0
- **G-mis (frozen before the curve):** leave-one-\(\alpha\)-out degree-2 OLS of \(A\) on \(X\) (same class as A0). Require
  \(\mathrm{RMSE}_{\mathrm{LOO}}>E_{\min}=10^{-4}\) **and**
  \(R^2_{\mathrm{LOO}}<0.99\).
  A0’s linear plant would fail this gate (\(q\approx 0\), \(R^2=1\)).

## Switchability gates (same roles as P0)

G0 macro ambiguity; G1 \(\lvert\rho(\alpha,X)\rvert>\rho_{\min}\);
G2 both signs of \(A\); G3 both sides have \(\ge N_{\pm}\) with
\(\lvert A\rvert>m_{\min}\); G4 not 1-vs-20.

P1 pass \(\iff\) G0–G4 and G-mis. No certificate. A1, if unlocked,
must keep frozen degree-2 OLS + one-sided conformal (not a bigger \(f\)).
