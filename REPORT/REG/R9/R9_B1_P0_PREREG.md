# R9-B1-P0 Preregistration — Periodic Active Revalidation Feasibility

Date: 2026-08-17  
Status: **RAN**; **`R9_B1_P0_PASS=false`**; dwell bounded, shift NetVoI not  
Depends on: `REPORT/REP/R9/R9_B0_REPORT.md`  
Does not: train a new model; CUSUM; adaptive \(M\); \(M\)-sweep;
change detector; re-fit logistic / WM / \(q\)

## Principle (from B0, not a new detector)

\[
\boxed{\textbf{policy-induced epistemic blindness}}
\]

\[
u=1\to\text{validity observable}\to\text{license cons}\to u=0.4
\to\text{validity unobservable.}
\]

Do not retune B0’s \(h\). Licensed cons does not generate safety-validating
evidence, so a certificate needs a finite age.

## Question

If we **re-run the frozen calibration** every \(M\) tasks, with no
detector, can stale-license dwell be bounded at acceptable long-run cost?

\[
\mathrm{calibrate}\to\underbrace{\mathrm{task}\cdots\mathrm{task}}_{M}
\to\mathrm{re\text{-}calibrate}\to\cdots
\]

\[
M=K_{\min}=2
\]

from P0/A0 (oracle \(=\) realized). Not chosen from B0’s 12 harmful
commits. Not swept.

## Frozen objects

P0 probe; R0 PD; A0 logistic and \(\tau_{\mathcal V}=0.5\); \(L_{\mathrm{WM}}>\delta\).

Probes **before** tasks \(k\in\{1,3,5\}\) on a \(K=6\) block.
Initial \(F_{\max}=2.0\). Held commit-set only.

## Shift phases (must not align only with probes)

| phase | \(F\) changes at | structural \(K_{\mathrm{stale}}\) |
|---|---|---|
| `at_probe` | \(k=3\) (same instant as a probe) | \(0\) |
| `after_probe` | \(k=2\) (one task, then probe at \(3\)) | \(\le 1\) |

`after_probe` is the worst phase this preflight treats as operational:
change just after the first-window probe has been used for one task.
A full two-task lag after a probe is the same window length as \(M\);
it is **not** the gated worst case (that would make \(K_{\mathrm{stale}}\le 1\)
impossible). Sequences: stay \(2.0\), benign \(2.2\), invalid \(1.5\).

Comparator `never`: one initial calibration, A0 persistence, same cuts.

## Gates (all required)

- **G-revalidation**: first probe at/after the cut: stay and benign
  \(b\ge\tau_{\mathcal V}\), invalid \(b<\tau_{\mathcal V}\)
  (rate \(\ge 0.90\) on commit-set)
- **G-stale**: invalid `after_probe`: \(K_{\mathrm{stale}}\le 1\) and
  \(N_{\mathrm{harmful,repeat}}=0\)
- **G-benign**: \(2.0\to 2.2\) remains licensed after that probe
  (rate \(\ge 0.90\))
- **G-stable-cost**: stay blocks, all \(C_{\mathrm{cal}}\) billed,
  mean \(\mathrm{NetVoI}>0\)
- **G-shift-value**: invalid `after_probe`
  \(\mathrm{NetVoI}_{\mathrm{periodic}}>\mathrm{NetVoI}_{\mathrm{never}}\)

## After GO

Unlock adaptive R9-B1 (re-probe less often than \(M=2\) using age/hazard).
Not this stage.
