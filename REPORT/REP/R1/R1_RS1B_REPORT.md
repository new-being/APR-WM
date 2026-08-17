# R1-RS1B Report — Long-Horizon Admissibility of Revise-Worthy Revisions

Date: 2026-08-15  
Prereg: `REPORT/REG/R1/R1_RS1B_PREREG.md` (frozen; SHA256 recorded in `runs/r1_rs1b/formal/summary.json`)  
Artifacts: `runs/r1_rs1b/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`

## Decision

\[
\boxed{RS1B\_GO = \mathrm{false}}
\]

\[
\boxed{
\text{On } \mathcal P_{\mathrm{rev}},\ \text{passivity and H32 utility transfer;}
\text{ frozen dynamics filters do not yet separate H32-unsafe support exits}
}
\]

RS1C and RS2 remain **locked**. No detector / VoI / \(C_{\mathrm{tol}}\) / diagnostic
threshold was changed after seeing H32.

Smoke remains plumbing only. This 30-episode held-out matrix is the first
run allowed to set `RS1B_GO`.

## Scientific question (unchanged)

\[
\boxed{
\text{Among revise-worthy mismatches, do the frozen dynamics-level filters
separate long-horizon-safe revisions from H32-unsafe ones?}
}
\]

Intake was the frozen RS1A.5 policy, not the old RS1 accept set:

\[
\mathcal P_{\mathrm{rev}}
=
\{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=1\}
\]

with \(C_{\mathrm{tol}}=0.001934\) and per-amplitude \(D_0\) thresholds from
`runs/r1_rs1a5/formal/summary.json`.

## Matrix

\(5\) seeds \(\{9901,9911,9921,9931,9941\}\)
\(\times 3\) \(\alpha\in\{-0.12,-0.18,-0.24\}\)
\(\times 2\) \(A\in\{1.5,2.0\}A_0\)
= **30 episodes**.

H2/H4/H8 remained acceptance-visible. Accept was frozen before blind H32.
H32 queries never entered the accept bit.

## Hard gates

| Gate | Value | Pass |
|------|------:|:----:|
| accepted revise-worthy count \(>0\) | **15** | ✓ |
| passivity violations among accepts | **0** | ✓ |
| seed-mean H32 paired gain \(>0\) | **+0.00418** (95% CI \([0.00195,0.00642]\), \(n=5\)) | ✓ |
| accepted H32-stable rate \(\ge 0.90\) | **0.80** (12/15) | ✗ |

`RS1B_GO = false` solely because of the stability-rate bar. Utility and
passivity gates passed, and the paired-gain CI lower bound is also \(>0\).

Expansion \(\rightarrow\) instability AUROC \(=0.667\) (**mechanism hit = false**;
prereg threshold \(0.75\)). This is diagnostic only and was **not** used to
edit GO.

## Funnel

| Stage | n |
|-------|--:|
| episodes | 30 |
| revise-worthy intake | **18** (60%) |
| frozen R0.6 pipeline executed | 18 |
| pipeline accepted (H2/H4/H8 + passivity/utility) | 15 |
| dynamics-filter accepted (damping \(\ge0\), \(P_r\le0\)) | **15** |
| H32 stable among those accepts | **12** |

All ten \(\alpha=-0.12\) cells missed \(D_0\) detect at the frozen FPR\(\le1\%\)
threshold, so they never entered \(\mathcal P_{\mathrm{rev}}\). That is the
RS1A weak-signal / excitation fact, not an RS1B retune signal.

Selected operator on executed pipelines: `abs_v_v` 16 / `x2` 2. The two `x2`
cells were **not** pipeline-accepted (and showed passivity violation on the
rejected candidate). They are outside the accepted set, so they do not
violate the hard passivity-among-accepts gate.

## What transferred

On all **15** accepted revise-worthy revisions:

- candidate was dissipative `abs_v_v`;
- \(\min d_{\mathrm{eff}}>0\);
- \(\max P_r\le0\) (numerical zeros / tiny negatives);
- **every** paired H32 gain was positive (min \(+0.00178\)).

So on this cleaner population, frozen R0.6 is not repeating the old
“short-safe, long-worse RMSE” pattern. Blind H32 RMSE fell in every accepted
cell.

## What did not transfer

`stable_h32` is the frozen R0.6 support predicate: every H32 query rollout
must stay finite and in the **modeled** validity region (`support_code=0`).
The three unstable accepts all had \(\alpha=-0.24\) and **still improved RMSE**:

| seed | \(A/A_0\) | gain H32 | support-exit | \(\lambda\) excess max |
|-----:|--------:|---------:|-------------:|-----------------------:|
| 9901 | 1.5 | +0.00637 | 0.309 | \(-4.15\times10^{-3}\) |
| 9901 | 2.0 | +0.00626 | 0.142 | \(\approx0\) |
| 9921 | 1.5 | +0.00295 | 0.171 | \(-5.26\times10^{-3}\) |

Mean support-exit: unstable **0.207** vs stable **0.085**. Expansion excess
does **not** mark them: all three unstable cells have \(\le0\) excess, while
one stable cell (9941, \(\alpha=-0.24\), \(2A_0\)) has excess \(+0.0129\).

Axis classification required by prereg (no RS1A reopen):

| Axis | Reading |
|------|---------|
| effective damping | hard filter held; does not explain the 3 fails |
| energy / passivity | 0 accepted violations |
| Jacobian expansion excess | AUROC 0.67; **not** a separator here |
| support exit | **primary residual failure mode** (H32 leaves fit/modeled support) |

## What this does and does not claim

Does:

- confirmatory evaluation of the frozen RS1B protocol on \(\mathcal P_{\mathrm{rev}}\);
- show passivity + seed-level H32 utility on that population;
- locate the remaining miss at **support-calibrated long-horizon admissibility**,
  not at detector/VoI/tolerance.

Does not:

- claim dynamics invariants solved H32 stability;
- unlock RS1C or RS2;
- authorize lowering the 0.90 stability bar, adding a support-exit hard
  filter, or changing Jacobian thresholds after seeing these three cells.

Those would turn the confirmatory matrix into a development set.

## Unlock status

```text
RS1A.5          frozen, passed
RS1B plumbing   passed
RS1B_GO         false
RS1C            locked
RS2             locked
```

On fail, next scientific work (if opened later) should be a **new**
preregistered RS1B.1 on the support-exit / modeled-validity calibration of
H32, still using \(\mathcal P_{\mathrm{rev}}\) only.
