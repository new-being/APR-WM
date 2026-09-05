# R1-RS1B.2 Report — Support-Band Identification

Date: 2026-08-16  
Prereg: `REPORT/REG/R1/R1_RS1B2_PREREG.md`  
Artifacts: `runs/r1_rs1b2/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`

## Decision

\[
\boxed{RS1B.2\_GO = \mathrm{true}}
\]

\[
\boxed{
\text{The intermediate support band is fillable by targeted queries;
it is not a dynamical impossibility on this Door.}
}
\]

This report does **not** install a support-exit hard filter and does **not**
unlock RS2. Distance formula, revision population, passivity, detector, and
VoI were not changed. The support-risk branch **closes here** (no RS1B.3).
A *new* policy prereg may be drafted: `REPORT/REG/R1/R1_RS1C_PREREG.md`. That
draft is not a repair of `RS1B_GO`.

Occupancy was the only GO bar. Mechanism labels below are reported under the
frozen rules; they do **not** reopen RS1B.1’s continuous-beats-binary gate.

## Plumbing smoke

Seed `9031`, \(\alpha=-0.24\), \(1.5A_0\): 4 queries (one intended stratum),
accepted `abs_v_v`, `n_harmful=0`. GO not evaluated.

## Matrix

New seeds \(\{10001,\ldots,10041\}\), same \(\alpha\) / \(A\) as RS1B.
15/30 episodes accepted; **240** accepted H32 queries (16 per accepted episode).

Intended \(\to\) realized confusion is **diagonal** (60/60 in every band).
The sampler reached the frozen bins instead of washing them into a far jump.

## Occupancy (GO)

| Realized bin | \(D_{\mathrm{exit}}\) | n | Pass (\(\ge12\)) |
|--------------|------------------------|--:|:----------------:|
| B0 supported | \(=0\) | **60** | ✓ |
| B1 mild | \((0,0.04]\) | **60** | ✓ |
| B2 mid | \((0.04,0.08]\) | **60** | ✓ |
| B3 far | \(>0.08\) | **60** | ✓ |

RS1B.1’s empty \((0,0.04]\) was an **off-the-shelf query-support** problem,
not proof that mild departure cannot exist.

## Mechanism (frozen classification)

| Quantity | Value |
|----------|------:|
| Spearman\((D_{\mathrm{exit}},\mathrm{RMSE}^{\mathrm{rev}}\mid D>0)\) | **+0.018** |
| \(\overline{\mathrm{RMSE}}(B1)\) | \(3.24\times10^{-4}\) |
| \(\overline{\mathrm{RMSE}}(B2)\) | \(5.70\times10^{-4}\) |
| \(\overline{\mathrm{RMSE}}(B3)\) | \(5.61\times10^{-4}\) |
| \(\overline{\mathrm{RMSE}}(B0)\) | \(\sim10^{-10}\) |
| continuous candidate (\(B1<B3\) and \(\rho>0\)) | **true** |
| jump candidate | false |
| \(P(\mathrm{gain}>0\mid I_{\mathrm{exit}}>0)\) | **0.994** |
| harmful queries | **1 / 240** |

The frozen continuous rule therefore fires. The *shape* is more specific than
a smooth Euclidean slope:

```text
B0 (interior)  ~ 0 RMSE
      ↓  large step when D leaves 0
B1 (mild)      3.2e-4
      ↓  smaller step
B2 ≈ B3        ~5.6e-4  (plateau)
```

Among exited queries, \(\rho=0.018\) is negligible: **how far** past the wall
barely ranks residual error once the wall has been crossed. B2 and B3 means
are indistinguishable. So:

- **not** “binary event is unfillable / B1 cannot exist”;
- **also not** a well-calibrated continuous risk function of \(d_{\mathcal S}\);
- closest reading: **interior vs exterior is the dominant regime**, with a
  weak extra bump from mild to mid, then a plateau.

B0 RMSE/gain near zero is partly the small targeted excitation in deep
interior (both incumbent and revision already match). The scientifically
relevant contrast is B0 vs \(\{B1,B2,B3\}\) and B1 vs B3.

## What this does not authorize

- \(I_{\mathrm{exit}}>0\Rightarrow\mathrm{reject}\) (still anti-utility:
  exit queries keep positive gain);
- treating RS1B.1 `GO=false` as overturned;
- unlocking RS2, or rewriting `RS1B_GO`;
- opening RS1B.3 / retuning \(D_{\mathrm{exit}}\);
- changing `stable_h32`.

RS1B’s 80% binary-stability fail remains on the record. Its interpretation
stays: frozen modeled-support *retention* is a bad harm surrogate. RS1B.2
adds: mild support departure **can** be observed, and the big predictive
gap is crossing \(D=0\), not sliding along \(D\).

## Frozen role of support

\[
\boxed{
\text{support departure is evidence of reduced epistemic confidence,
not evidence of revision harm}
}
\]

\[
\boxed{
\begin{aligned}
RS1A &: \text{error magnitude }\not\Rightarrow\text{ worth revising}\\
RS1B.1 &: \text{support departure }\not\Rightarrow\text{ revision harmful}\\
RS1B.2 &: \text{intermediate }D\text{ is reachable; risk is mostly a }D=0\text{ crossing, not a slope in }D
\end{aligned}
}
\]

## Unlock status

```text
RS1B_GO      false
RS1B.1_GO    false
RS1B.2_GO    true   (support-risk branch closed)
RS1B.3       not opened
RS1C_GO      true
RS2          prereg may be drafted; not implemented
hard filter  not installed
```
