# R7-A1 Preregistration — Certificate Under Misspecification

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R7/R7_P1_REPORT.md` (`PASS=true`)  
Does not upgrade: predictor class (degree-2 OLS); conformal form
\(L=\hat A-q\); \(\delta=10^{-3}\); \(\varepsilon=0.10\)  
Does not reuse: P1 \(\alpha\) grid as formal evidence; A0 linear plant

## Question

When \(\hat A\) is no longer nearly exact, does the frozen one-sided
certificate convert extra residual into **abstention** (\(q\uparrow\))
rather than **unsafe commits**?

Same rule as A0:

\[
L=\hat A-q,\qquad L>\delta\Rightarrow\mathrm{commit\ cons}.
\]

## Splits (by \(\alpha\); disjoint from P1)

P1 grid is development only. Larger held \(n\) than A0 (\(n_{\mathrm{test}}=28\)).

- **fit** \(\{0.45,0.70,0.95,1.20,1.45,1.65,1.90,2.15,2.40,2.65,2.85,3.10,3.35,3.60\}\)
- **cal** \(\{0.55,0.75,1.00,1.25,1.50,1.75,1.95,2.20,2.45,2.70,2.95,3.15,3.40,3.65\}\)
- **test** \(\{0.60,0.65,0.85,0.90,1.05,1.15,1.30,1.35,1.55,1.60,1.80,1.85,2.05,2.10,2.25,2.35,2.50,2.55,2.75,2.80,3.00,3.05,3.25,3.30,3.45,3.55,3.70,3.75\}\)

Plant: P1 saturation \(F_{\max}=2.0\). \(X=0.25\alpha\). Shuffle-\(X\) seed 0
inside each split (control, not tuning).

## Gates (unchanged thresholds)

G-cert, G-safe (non-vacuous precision \(=1\)), G-recall (\(\ge 0.25\)),
G-VoI (\(>0\)), G-ctrl (\(\mathrm{VoI}_{\mathrm{align}}>\mathrm{VoI}_{\mathrm{shuf}}\)).

Primary readout is the **pattern**, not a beauty contest:

| safety | recall | meaning |
|---|---|---|
| \(\checkmark\) | \(\checkmark\) | robust positive certification |
| \(\checkmark\) | \(\times\) | safe but over-conservative |
| \(\times\) | \(*\) | certificate invalid |

Compare \(q\) to A0 (\(q\approx 0\)). Expected under misspecification:
\(q\) rises and commits shrink, without harmful commits.
