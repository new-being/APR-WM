# R1-RS5A Report — Intervention-Indexed Calibration and Abstention

Date: 2026-08-16  
Prereg: `REPORT/REG/R1_RS5A_PREREG.md`  
Depends on: `REPORT/REP/R1_RS4A_REPORT.md`  
Artifacts: `runs/r1_rs5a/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `61be3705f40f144484ba6e37aeaa69a719bd1acb4a3957273db465276073b30d`

## Decision

\[
\boxed{RS5A\_GO=\mathrm{true}}
\]

\[
\boxed{
\text{epistemic calibration can be intervention-indexed,
with abstention outside calibration support}
}
\]

This does **not** set `RS4A_GO` or `RS2_GO`. It does **not** unlock RS5B
(typed consequence), revision, VoI, or H32. Zero-shot transport remains
abandoned.

Smoke (`runs/r1_rs5a/smoke/`) is plumbing only. The 90-episode matrix is
the first run allowed to set `RS5A_GO`.

## Question (unchanged)

\[
\boxed{
\text{If epistemic semantics are intervention-dependent,
can a world model remain well-calibrated by explicitly tracking
which intervention calibration is valid?}
}
\]

Not: make \(\hat p\) the same across \(\mathcal I\).
Not: zero-shot \(p\) on `pull_push`.

## Design (frozen)

- Calibrated: Mode-A and contact-pull. Unseen: `pull_push` (never in
  \(\mathcal C\) or \(\tau_{\mathrm{supp}}\)).
- Coarse \(\hat{\mathcal I}\): `contact_frac<0.5` (no family string).
- \(\mathcal C_{\mathcal I}\): ridge logistic \(p(Y=1\mid\log D_0)\).
- Support: Mahalanobis on \((\mathrm{sign\_coverage},\log n,q_{\mathrm{span}})\);
  \(\tau=\) 95th percentile of **development** distances.
- Seeds \(\{15101,15111,15121\}\) development, \(\{15131,15141\}\) held-out.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 indexed Brier \(<\) global on known \(\mathcal I\) | **PASS** | \(0.201<0.214\); ECE \(0.032<0.117\) |
| H2 known-\(\mathcal I\) C0 FPR \(\le0.20\) | **PASS** | Mode-A \(0/4\); contact \(0/4\) |
| H3 unseen false-confidence | **PASS** | abstain FCR \(0/30\); global \(=1\); indexed \(=1\) |
| H4 held-out contact-pull abstain \(\le0.20\) | **PASS** | \(0/12\) |

\[
\boxed{RS5A\_GO=H1\land H2\land H3\land H4=\mathrm{true}}
\]

\(\pi_{\mathrm{indexed+abstain}}\) is the policy that satisfies the new
core gate. \(\pi_{\mathrm{global}}\) and \(\pi_{\mathrm{indexed}}\) are
**always confident** on `pull_push`, which is exactly the illegal reuse
of an old ruler.

## What the numbers say

On held-out known interventions, a separate \(\mathcal C_{\mathcal I}\)
is slightly sharper than one pooled logistic (Brier) and much better
calibrated (ECE). The win is **conditional calibration**, not a shared
numerical \(D_0\) scale.

On `pull_push`, coarse index is still `contact`, so \(\pi_{\mathrm{indexed}}\)
silently applies \(\mathcal C_{\mathrm{pull}}\) and is 100% confident.
The support check marks every `pull_push` episode out of the
contact-pull cloud (`sign_coverage` \(\approx0.49\) vs pull \(\approx0.10\))
and abstains \(30/30\), while leaving held-out contact-pull fully in
support (\(0\) abstain). Geometry is used here as **validity of a
calibration object**, which is the RS4A fingerprint result put in its
correct role—not as a transported probability.

That is the architectural distinction from one-hot \(\tau_A,\tau_C\):
the system can be in `known + calibrated`, or `known-coarse but
out-of-support` (`pull_push`), and the latter is `intervention-uncalibrated`.

## What this does not authorize

- claiming zero-shot transport is solved;
- installing revision / VoI / H32;
- opening RS5B without a new preregistration;
- treating `contact_frac` as a closed-world domain ID that always has a
  \(\tau\);
- treating RS5A as the default next **mainline** for a unified embodied
  world model.

RS5A remains a valid side result: when interventions are treated as
separate calibration objects, indexed \(\mathcal C_{\mathcal I}\) plus
support abstention beats a global ruler. The transport / small-domain
calibration branch is otherwise **closed**; see
`REPORT/REP/R1_TRANSPORT_STAGE_FREEZE.md`. Next main stage is R3-V7A
(mixture-trained contextual belief), not RS5B.

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
RS5A\_GO &= true\\
RS3B/C &= locked\\
RS4B/C &= locked\\
RS5B/C &= locked
\end{aligned}
}
\]
