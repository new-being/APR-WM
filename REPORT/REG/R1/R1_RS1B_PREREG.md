# R1-RS1B Preregistration — Long-Horizon Admissibility of Revise-Worthy Revisions

Date: 2026-08-15  
Status: **FROZEN; formal completed** — see `REPORT/REP/R1/R1_RS1B_REPORT.md` (`RS1B_GO=false`)  
Depends on: `REPORT/REP/R1/R1_RS1A_STAGE_FREEZE.md` (RS1A closed)
Frozen implementation: `aprwm_v0/r1_rs1b.py`

## Scientific question

\[
\boxed{
\text{Among consequential and epistemically justified revisions,
what dynamics-level conditions predict long-horizon stability?}
}
\]

Not the old RS1 question (“why do short-horizon accepts fail at H32?” on an
unfiltered accept set), but:

\[
\mathcal P_{\mathrm{rev}}
\rightarrow
\text{candidate revision}
\rightarrow
\text{long-horizon admissibility}
\rightarrow
\text{install / reject}
\]

## Orthogonality with RS1A

| Stage | Question |
|-------|----------|
| RS1A | Should I spend epistemic effort and consider revision? |
| RS1B | Given revision is justified, is the revised vector field safe long-horizon? |

**Forbidden in RS1B:** retuning detectors, VoI \(\lambda\), \(C_{\mathrm{tol}}\), or
support channels after seeing H32 outcomes.

## Intake population (frozen)

\[
\boxed{
\mathcal P_{\mathrm{rev}}
=
\{C\ge C_{\mathrm{tol}}\}
\cap
\{\mathrm{detect}=1\}
}
\]

Construction protocol (reuse RS1A.4/A.5 machinery):

1. Door Mode-A, Panda frozen, P0 passive, \(f=0.20\) Hz  
2. \(\alpha\in\{-0.12,-0.18,-0.24\}\) (consequential family; exclude tolerate-scale \(\lvert\alpha\rvert\le0.09\))  
3. Excitation at **VoI-optimal** \(A=1.5A_0\) (from RS1A.5), plus \(A=2A_0\) as sensitivity  
4. Keep only episodes with \(C\ge C_{\mathrm{tol}}\) and \(D_0\) detect at cell FPR\(\le1\%\)  
5. On those episodes only: run frozen R0.6 proposal→selection→short validation→passivity→utility accept

Seeds (new held-out): \(\{9901,9911,9921,9931,9941\}\).

Smoke: seed `9011` × \(\alpha=-0.24\) × \(1.5A_0\).

Formal matrix: \(5\) seeds × \(3\) coefficients × \(2\) amplitudes = **30 episodes**.

The intake detector is not recalibrated in RS1B. `C_tol` and the per-amplitude
`D0` thresholds are loaded from the passing frozen
`runs/r1_rs1a5/formal/summary.json`. Both a passing RS0 summary and a passing
RS1A.5 summary are hard prerequisites.

Numerical diagnostics are frozen before smoke:

- 48 deterministic fit-support states per candidate (or all if fewer);
- centered finite differences with \(\epsilon=10^{-4}\);
- damping and power numerical tolerance \(10^{-8}\);
- H32 = `32 × 4` MuJoCo steps on 8 held-out queries;
- revision force is recomputed from current \((q,v)\) at every rollout step.

## Candidate revision object

Frozen R0.6 dissipative library revision of hinge generalized force:

\[
\tau_{\mathrm{rev}}
=
\tau_{\mathrm{nominal}}
+
\hat\alpha\,\phi(\hat k),\quad
\phi\in\mathcal L_{\mathrm{diss}}
\]

(typically \(\phi(v)=|v|v\) when correctly selected). No new MuJoCo-specific operators.

## Dynamics-level diagnostics (primary axes)

For incumbent \(f\) and revised \(f_{\mathrm{rev}}\) on rollout support \(S\):

1. **Effective damping**
   \[
   d_{\mathrm{eff}}(q,v)=-\partial r/\partial v
   \]
   Require \(d_{\mathrm{eff}}\ge0\) a.e. on \(S\) for dissipative candidates (hard filter).

2. **Energy production**
   \[
   P_r=v^\top r
   \]
   Passivity \(P_r\le0\) already hard in R0.6; report integrated excess
   \(\int\max(P_r,0)\,dt\) as soft severity.

3. **Local expansion**
   \[
   \lambda_{\max}\Bigl(\tfrac{J_{\mathrm{rev}}+J_{\mathrm{rev}}^\top}{2}\Bigr)
   -
   \lambda_{\max}\Bigl(\tfrac{J+J^\top}{2}\Bigr)
   \]
   on sampled states in \(S\). Hypothesis: positive excess expansion predicts H32 instability.

4. **Support distance / state excursion**
   Fraction of H32 rollout leaving the observation support used for fitting.

H2/H4/H8 remain acceptance-visible. **H32 is blind** (cannot enter accept).

## Primary endpoints

On \(\mathcal P_{\mathrm{rev}}\) accepted revisions only:

| Endpoint | Criterion |
|----------|-----------|
| H32 stability rate | report; target \(\ge99\%\) as aspirational |
| Seed-mean H32 paired gain vs no-revision | \(>0\) |
| Expansion excess → instability association | AUROC or rank correlation preregistered as diagnostic |
| Passivity violations among accepts | **0** (safety regression if any) |

### GO (frozen before plumbing smoke)

\[
\begin{aligned}
&\text{number of accepted revise-worthy revisions }>0\\
&\text{passivity accepts with }P_r>0:\ 0\\
&\text{seed-mean H32 gain }>0\\
&\text{among accepts, H32-stable rate }\ge 0.90
\quad\text{(diagnostic bar; 0.99 remains stretch)}
\end{aligned}
\]

If expansion excess separates stable/unstable accepts with AUROC \(\ge0.75\),
record as **mechanism hit** for a follow-on hard gate (RS1B.1), not as post-hoc GO edit.

## Explicit non-goals

- Extending validation horizon alone as the fix (R0.4)  
- Re-analyzing raw RS1’s 71 accepts  
- Contact / robot motion (RS2)  
- Changing VoI or tolerate thresholds

## Unlock onward

On RS1B_GO: draft **RS1C** integrated confirmation (detector+VoI+revise-worthy H32 together) before RS2 contact.

On fail: classify by dynamics axis (damping / energy / expansion / support exit); do not reopen RS1A.

Follow-on is **RS1B.1 support-risk calibration**, not a post-hoc
`support-exit < τ` hard filter. See `REPORT/REP/R1/R1_RS1B1_REPORT.md`.
