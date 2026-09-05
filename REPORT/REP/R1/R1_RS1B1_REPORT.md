# R1-RS1B.1 Report — Support-Calibrated Long-Horizon Admissibility

Date: 2026-08-16  
Prereg: `REPORT/REG/R1/R1_RS1B1_PREREG.md`  
Artifacts: `runs/r1_rs1b1/formal/` (`episodes.csv`, `queries.csv`, `summary.json`)  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`

## Decision

\[
\boxed{RS1B.1\_GO = \mathrm{false}}
\]

\[
\boxed{
I_{\mathrm{exit}}>0
\not\Rightarrow
\mathrm{harmful}
}
\]

Support departure raises *absolute* revised RMSE but does **not** make the
revision worse than the incumbent. Binary modeled-support retention remains
an epistemic flag, not a physical-safety cut. No support-exit hard filter
was installed. RS1C / RS2 stay **locked**. RS1B_GO is unchanged (`false`).

## Plumbing smoke (not confirmatory)

Seed `9021`, \(\alpha=-0.24\), \(1.5A_0\): revise-worthy, `abs_v_v`,
`stable_h32=false`, \(I_{\mathrm{exit}}>0\), yet H32 gain \(+0.00702\) and
`n_harmful_queries=0`. Schema / accept-freeze / hidden isolation held.
`RS1B.1_GO` was not evaluated on smoke.

## Matrix

New held-out seeds \(\{9951,9961,9971,9981,9991\}\) ×
\(\alpha\in\{-0.12,-0.18,-0.24\}\) × \(\{1.5,2.0\}A_0\) = 30 episodes.
Intake = frozen RS1A.5 policy. Accept frozen before blind H32.

| Count | n |
|------:|--:|
| episodes | 30 |
| revise-worthy | 20 |
| accepted | **20** |
| accepted H32 queries | **160** |
| queries with \(I_{\mathrm{exit}}>0\) | **8** |
| harmful queries (non-finite or gain \(<0\)) | **0** |
| episode-level `stable_h32` among accepts | 12/20 |

All eight exits are \(\alpha=-0.24\). Every accepted episode has
\(\Delta\mathrm{RMSE}>0\).

## Hard gates

| Gate | Value | Pass |
|------|------:|:----:|
| accepted \(>0\) | 20 | ✓ |
| non-implication: mean gain \(\mid I_{\mathrm{exit}}>0\) | **+0.0141**, \(P(\mathrm{gain}>0)=1.0\) (8 queries) | ✓ |
| seed-mean Spearman\((D_{\mathrm{exit}},\mathrm{RMSE}^{\mathrm{rev}})\) | **+0.263** (95% CI \([0.052,0.473]\), \(n=4\)) | ✓ |
| \(\lvert\rho_D\rvert>\lvert\rho_I\rvert\) | 0.263 ≯ 0.266 | ✗ |
| risky RMSE \(>\) supported RMSE | 0.000495 \(>\) 0.000163 (\(n=8\) vs \(152\)) | ✓ |

GO fails only on “continuous beats binary”. The two rank correlations are
essentially tied because 152/160 queries have \(D_{\mathrm{exit}}=I_{\mathrm{exit}}=0\);
the eight exits are a near-collinear step.

## \(D_{\mathrm{tol}}\) and three regions

Frozen formula on this matrix: \(D_{\mathrm{tol}}=0.0399\).

All eight positive \(D_{\mathrm{exit}}\) values lie in \([0.0477,0.1315]\),
so they skip the middle band:

| Region | n | mean \(\mathrm{RMSE}^{\mathrm{rev}}\) | mean gain |
|--------|--:|--------------------------------------:|----------:|
| supported | 152 | 0.000163 | +0.00390 |
| extrapolative | **0** | — | — |
| unsupported-risky | 8 | **0.000495** | **+0.0141** |

Leaving modeled support **triples** revised RMSE relative to the interior,
but **quadruples** paired gain: no-revision error grows faster than revised
error. That is why binary `stable_h32=false` is conservative relative to
prediction *failure*.

Pooled Spearman\((D_{\mathrm{exit}},\mathrm{gain})=+0.278\).

## What this claims

Does:

- falsify “any modeled-support exit is a harmful revision” on a fresh
  \(\mathcal P_{\mathrm{rev}}\) sample;
- show a weak positive association between \(d_{\mathcal S}\) and absolute
  revised H32 RMSE;
- show the RS1A.4-style split is currently **two-state** (supported vs
  jumped-risky), not a populated three-state band.

Does not:

- justify adding \(I_{\mathrm{exit}}<\tau\) to RS1B acceptance;
- claim \(d_{\mathcal S}\) is a calibrated probability of harm (harm rate is 0);
- unlock RS1C / RS2;
- reopen detector / VoI / passivity.

## Mechanism sentence

\[
\boxed{
\text{Physical admissibility prevents catastrophic dynamics,
but epistemic admissibility still requires support-risk calibration.}
}
\]

Here calibration says: modest support exit marks **higher residual
difficulty**, not **revision harm**. Continuous \(d_{\mathcal S}\) does not
yet outrank the binary flag, because exits are rare and clustered.

## Unlock status

```text
RS1B_GO     false  (unchanged)
RS1B.1_GO   false
RS1C        locked
RS2         locked
```

A later RS1B.2, if opened, should ask whether a *populated* middle band
exists under broader queries / longer horizons — not encode today's eight
exits as a hard reject.
