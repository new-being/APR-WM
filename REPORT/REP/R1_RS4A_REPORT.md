# R1-RS4A Report — Interaction-Geometry-Conditioned Structural Evidence

Date: 2026-08-16  
Prereg: `REPORT/REG/R1_RS4A_PREREG.md`  
Depends on: `REPORT/REP/R1_RS3A1_REPORT.md`  
Artifacts: `runs/r1_rs4a/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `2daadffb5d4469164dc1e223aaaf11ecc1bdd1a669d82798ad7ae9509ce050d1`

## Decision

\[
\boxed{RS4A\_GO=\mathrm{false}}
\]

\[
\boxed{
\text{learner-visible interaction geometry does not make }D_0
\text{ semantics transport to an unseen intervention family}
}
\]

This does **not** reopen the closed invariant-scalar search. It does
**not** unlock RS4B/C or RS3B/C. It does **not** authorize one-hot
\(\tau_A,\tau_C\). `pull_push` is not added to RS2 `SCRIPTS`.

Smoke (`runs/r1_rs4a/smoke/`) is plumbing only. `pull_push` has
sign-coverage \(\approx0.49\) vs unidirectional `fast_pull` \(\approx0.10\).

## Question (unchanged)

\[
\boxed{
\text{Can intervention structure explain how evidence semantics transform?}
}
\]

Score: ridge logistic \(p(Y=1\mid D_0,z_{\mathcal I})\). No domain ID.
Three families; leave-one-intervention-out.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 held-out C0 FPR \(\le0.20\) on \(\ge2/3\) folds | **PASS** | \(2/3\): push FPR \(0.50\); pull \(0\); Mode-A \(0\) |
| H2 Brier\((p_z)<\) Brier\((p_{D_0})\) on \(\ge2/3\) folds | **FAIL** | \(0/3\); geometry **worsens** Brier on every fold |
| H3 Mode-A held-out median \(p_{C1}>p_{C0}\) | **PASS*** | \(3.51\times10^{-10}>3.35\times10^{-10}\) (vacuous) |

\*H3 is true only as a numerical inequality. Both medians are \(\approx0\):
a contact-only trainer assigns essentially no inadequacy probability to
any Mode-A episode.

\[
\boxed{RS4A\_GO=H1\land H2\land H3=\mathrm{false}}
\]

## Leave-one-out folds

| Held-out family | C0 FPR | Brier \(p_z\) | Brier \(p_{D_0}\) | AUROC \(p_z\) | AUROC \(p_{D_0}\) |
|-----------------|-------:|--------------:|------------------:|--------------:|------------------:|
| `contact_push` | \(0.50\) | \(0.213\) | \(0.205\) | \(0.875\) | \(0.75\) |
| `contact_pull` | \(0\) | \(0.662\) | \(0.203\) | \(1.00\) | \(0.875\) |
| `mode_a` | \(0\) | \(0.667\) | \(0.324\) | \(0.605\) | \(0.795\) |

On `contact_pull`, \(p_z\) ranks perfectly (AUROC \(1\)) and is still a
worse probability (Brier \(0.66\) vs \(0.20\)). Discrimination without
calibration is the RS3A H1 over-read, now in LOIO form.

On Mode-A held out, geometry features that never appear in contact
training (`contact_frac=0`, \(\log n\approx8.5\)) push \(p_z\) to \(\approx0\)
for C0 **and** C1. \(D_0\)-only is less collapsed (AUROC \(0.80\) vs \(0.61\)).

So \(z_{\mathcal I}\) is not a transport map for evidence semantics. It
behaves like a **family fingerprint** that overfits the two training
mechanisms and fails as a probability on the third.

\[
\boxed{\text{discrimination}\neq\text{calibration}}
\]

\[
\boxed{
\text{interaction geometry may identify intervention families
without transporting calibrated evidential meaning}
}
\]

**Do not** open RS4A.1 / add \(z\) coordinates. Zero-shot transport of
evidential meaning across unknown interventions is **abandoned**.

The authorized next stage is **RS5A**
(`REPORT/REG/R1_RS5A_PREREG.md`): intervention-indexed calibration with
**abstention** outside calibration support. RS4B/C stay locked.

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
RS1C\_GO &= true\\
RS2\_GO &= false\\
RS2A\_GO &= true\\
RS2B\_GO &= true\\
RS3A\_GO &= false\\
RS3A.1\_GO &= false\\
RS4A\_GO &= false\\
RS3B/C &= locked\\
RS4B/C &= locked
\end{aligned}
}
\]
