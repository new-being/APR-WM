# R1-RS4A Preregistration — Interaction-Geometry-Conditioned Structural Evidence

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R1/R1_RS3A1_REPORT.md`  
Does not change: `RS3A_GO`, `RS3A.1_GO`, `RS2_GO`, `RS2A_GO`, `RS2B_GO`,
`RS1C_GO`, C0, \(s^\star\), RS1C policy, RS2 `SCRIPTS`  
Locks: **RS3B**, **RS3C**, **RS4B**, **RS4C**, and any further invariant scalar

## Placement

The intervention-free scalar programme is **closed**. This is a new main
stage, not RS3A.2.

\[
\boxed{
RS3\text{ scalar search closed}
\rightarrow
R1\text{-RS4A (this stage)}
\qquad
RS4B\text{/}C\text{ locked}
}
\]

Stage name:

\[
\boxed{
R1\text{-RS4A — Interaction-Geometry-Conditioned Structural Evidence}
}
\]

中文名：**交互几何条件化的结构证据**。

## Why not another scalar

\[
\boxed{
\text{No intervention-free scalar tested so far preserves
structural-evidence semantics.}
}
\]

\(D_0\), \(S_\perp\), and \(E_{\mathrm{XR}}\) all failed. RS3A.1 further
showed ranking reversal, so the problem is not only scale drift.

Forbidden: a fourth invariant \(S_k\); Mode-A vs contact one-hot
thresholds \(\tau_A,\tau_C\).

## Scientific question

\[
\boxed{
\text{Can intervention structure explain how evidence semantics transform?}
}
\]

Do not seek \(E\perp\mathcal I\). Construct

\[
p(\text{inadequate}\mid D_0,z_{\mathcal I})
\]

where \(z_{\mathcal I}\) is **learner-visible interaction geometry**, not
a domain ID.

中文：如果显式告诉 evidence model 当前 trajectory 的可辨识性几何，而不是
告诉它“这是 Mode-A 还是 contact”，结构错误的证据语义能否跨干预稳定？

Success is **leave-one-intervention-out calibration**, not in-sample
AUROC.

## Intervention families (three, not two)

| Family | Mechanism | Frozen jobs |
|--------|-----------|-------------|
| `mode_a` | direct hinge torque | \(A/A_0\in\{1.0,1.5\}\) |
| `contact_pull` | unidirectional robot contact | `fast_pull@1`, `pull_release@\(s^\star\)` |
| `contact_push` | bidirectional contact | `pull_push` at scale \(\{1.0,1.5\}\) |

`pull_push` is added only as an RS4A interaction. It is **not** in RS2
`SCRIPTS` and does not reopen C0/Formal.

Probe window: Mode-A full sine; `contact_pull` `phase==2`; `contact_push`
`phase\in\{2,2.5\}\).

## Learner-visible \(z_{\mathcal I}\) (frozen)

From the probe/excitation window only:

1. \(\log(\sigma_{\min}(J_\theta)+\epsilon)\) — min singular value of tangent
2. \(\log(\kappa(J_\theta^\top J_\theta)+\epsilon)\) — tangent condition
3. sign coverage \(\min(\overline{1_{v>0}},\overline{1_{v<0}})\)
4. \(\log n_{\mathrm{probe}}\)
5. contact-active fraction (mean `contact_count>0`; Mode-A \(=0\))
6. \(\log(\mathrm{rms}(v)+\epsilon)\)
7. \(\log(\kappa(G)+\epsilon)\) — observed library Gram condition
8. \(\lvert\mathrm{corr}(|v|v,v^2)\rvert\) — observed basis collinearity
9. \(q\)-span \(\max q-\min q\)

Together with \(\log(D_0+\epsilon)\). **Forbidden in the score:** domain
ID, true \(\alpha\), true \(\phi\), \(X_{\phi,\perp}\).

## Model

Ridge logistic regression (intercept + features, ridge \(1.0\), 25 Newton
steps). Train-only z-scoring. Baseline: intercept + \(\log D_0\) only.

Label \(Y=1\) on \(\{\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\), \(Y=0\)
on C0. C2 excluded.

## Leave-one-intervention-out

Three folds. Each holds out one family; fit on the other two; evaluate
**only** on the held-out family.

\[
\boxed{
\text{train/calibrate on two intervention regimes}
\rightarrow
\text{test on unseen third regime}
}
\]

This is the test that \(z_{\mathcal I}\) learned transport structure
rather than domain labels.

## Matrix

New seeds \(\{14101,14111,14121,14131,14141\}\).
Regimes \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\).
\(N=5\times3\times(2+2+2)=90\).

Smoke: seed `14101` × C0 × {Mode-A \(1.5A_0\), `fast_pull`, `pull_push@1`}.
No GO.

## Hypotheses / GO

Let \(p_z=p(Y=1\mid D_0,z)\) and \(p_0=p(Y=1\mid D_0)\).
On each fold, \(\tau\) is the 95th percentile of **train C0** \(p_z\).

### H1 — held-out C0 specificity (not a second fitted \(\tau\))

Held-out C0 FPR of \(p_z>\tau\) \(\le 0.20\) (allow \(2/10\)).

Pass if **at least 2 of 3** folds satisfy H1.

### H2 — geometry beats \(D_0\)-only on the unseen family

Held-out Brier of \(p_z\) **strictly below** Brier of \(p_0\).

Pass if **at least 2 of 3** folds satisfy H2.

### H3 — Mode-A held-out is not a domain-label cheat

On the fold that **never trains on Mode-A**, held-out median \(p_z\) on
C1 exceeds held-out median \(p_z\) on C0.

This fold is the one where contact_frac cannot be a trivial Mode-A ID
learned from mixed Mode-A/contact training; both train families are
contact.

\[
\boxed{RS4A\_GO = H1 \land H2 \land H3}
\]

`RS4A_GO=true` does not unlock RS4B/C, RS3B/C, or domain thresholds.
It does not set `RS2_GO=true`.

## Failure interpretation

| Pattern | Read as |
|---------|---------|
| all pass | geometry explains how \(D_0\) semantics transform |
| ¬H1 | \(p_z\) still over-fires on unseen C0 |
| H1 ∧ ¬H2 | geometry does not beat amplitude on unseen \(\mathcal I\) |
| ¬H3 | contact-only training cannot read Mode-A geometry; likely a
contact-frac proxy |
| in-sample AUROC high, LOIO fail | memorized two domains, not transport |

If GO is false, do **not** add more \(z\) coordinates ad hoc. The next
honest question would be an explicitly intervention-indexed policy under
a new preregistration (still not RS3B).

## Non-goals

- invariant scalars;
- one-hot \(\tau_A,\tau_C\);
- RS4B typed consequence / RS4C policy;
- C2;
- true operator / domain ID in \(p_z\).
