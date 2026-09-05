# R3-V7C.2 Report — Tactile Epistemic Information Decomposition

Date: 2026-08-16  
Prereg: `REPORT/REG/R3/R3_V7C2_PREREG.md`  
Depends on: `REPORT/REP/R3/R3_V7C1_REPORT.md`  
Artifacts: `runs/r3_v7c2/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `e53b6cebc159e240ae70389d1527f15ef367d3bc0d73195266c53fe960a494b8`

## Decision (locus, not a method GO)

\[
\boxed{\mathrm{locus}=\textbf{FIELD}}
\]

\[
\boxed{
\text{the 8×8 taxel field adds no incremental warranted-evidence
information beyond }h^{-c}\ (\Delta_X\le\varepsilon)
}
\]

Do **not** swap the CNN. Do **not** open a fusion study. Do **not**
open V7D. Next method experiment, if any, is a **new tactile
observation** prereg.

Smoke did not classify a locus.

## Question (unchanged)

Where is epistemically useful contact information lost, along
\(x\to z\to u\to p\)? Target \(e_t=y\,w_t\), not oracle contact flags.

## Design (frozen)

- Seeds \(\{23101,23111\}\) train, \(23121\) val, \(\{23131,23141\}\) held-out.
- Same GRU32 / 12-D / \(e_t\) BCE.
- **P0:** \(h^{-c}\). **PX:** linear \(64\to 2\) on flattened taxel.
  **PZ:** V7C.1 CNN \(\to 2\).
- \(\varepsilon=0.005\). \(\Delta_X=L_0-L_X\), \(\Delta_Z=L_0-L_Z\).

## Results

| Quantity | Value |
|---|---|
| \(L(P_0)\) | \(0.459\) |
| \(L(P_X)\) | \(0.458\) |
| \(L(P_Z)\) | \(0.478\) |
| \(\Delta_X\) | \(0.001\le 0.005\) |
| \(\Delta_Z\) | \(-0.019\) |
| shuffle \(\Delta\) | \(\approx 0\) |
| H4-style PZ vs N useful | **false** (\(B_{\mathrm{C0}}^{Z}=0.213>0.200\)) |
| matched taxel \(L_2\) early / late | \(0.0\) / \(5.44\) |

\[
\boxed{\Delta_X\le\varepsilon\Rightarrow\textbf{FIELD}}
\]

PZ is **worse** than P0 at predicting \(e_t\). Time-shuffling taxel
does not change \(L(P_Z)\): the CNN path is not using
episode-specific maps.

## How to read the matched-pair \(L_2\)

Late C0/C1 taxel maps **do** differ (\(L_2=5.44\) vs early \(0\)).
That is not the same as incremental \(e_t\) information given
proprioception, residual, and action. The predeclared locus uses
\(\Delta_X\), not pairwise map distance. The field can move with
hidden dynamics and still be **redundant or misaligned** for
warranted belief once \(h^{-c}\) is known.

Oracle-contact Spearman from V7C.1 remains a clue, not this stage’s
target.

## What follows

\[
\textbf{FIELD}\rightarrow\text{redesign tactile observation (new prereg)}
\]

Not encoder architecture. Not fusion. Not V7D. V7B.3 / V7C stay
true: wrist/joint proprioceptive contact already carries epistemic
value; this 8×8 gripper-local pressure map does not add more on
top of \(h^{-c}\).
