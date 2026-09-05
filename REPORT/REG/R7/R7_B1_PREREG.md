# R7-B1 Preregistration — Frozen Certificate Under Dynamics Shift

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R7/R7_B0_REPORT.md` (`GO=true`)  
B0 WM / \(q\): **FROZEN** (no refit, no recalibration)  
Does not: stronger WM; weighted conformal; online \(q\); damping
change; A-series unfreeze; new \(\alpha\) grid

## Question

Under non-exchangeable dynamics, does the ID-calibrated certificate
degrade **safely** (more abstention) or emit **unsupported commits**?

\[
\boxed{
\text{ID calibration does not automatically certify shifted dynamics.}
}
\]

This is a pressure test, not an adaptation stage.

## Frozen objects (from B0 summary)

\[
f_{\mathrm{WM}},\quad q,\quad
\delta=10^{-3},\quad \varepsilon=0.10,\quad
L=\hat A_{\mathrm{WM}}-q,\quad
L>\delta\Rightarrow\mathrm{commit}.
\]

Evaluate on **B0 held-test** \(\alpha\) only. \(X=0.25\alpha\) still.

## Single shift

\[
F_{\max}^{\mathrm{test}}=1.5
\qquad\text{(B0 / P1: }F_{\max}=2.0\text{)}.
\]

Damping \(c\) is unchanged. One parameter only. Tighter saturation
moves \(X\to Y^{future}\to A\) without hiding \(X\).

## Gates (reported; exchangeability not assumed)

Same four numbers as B0: coverage, precision, useful recall, VoI.
Shuffle-\(X\) at **test only** (frozen \(f_{\mathrm{WM}},q\)).

**Shift-valid** iff coverage \(\ge 0.90\) **and** no harmful commit
(zero commits count as safe, not invalid).

Recall / VoI vs B0 are **not** required for shift-valid.

## Pattern (locked before seeing B1)

| coverage + no harmful commit | recall vs B0 held | meaning |
|---|---|---|
| \(\checkmark\) | \(\downarrow\) | safe conservative degradation |
| \(\checkmark\) | \(\ge\) B0 | empirical shift robustness |
| \(\times\) | \(*\) | certificate not shift-valid |

If the last row: do **not** write “conformal failed.” Write: ID
calibration does not certify shifted dynamics. Next layer is shift
detection / epistemic validity, not retuning B0 \(q\).
