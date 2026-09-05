# R1-RS2B Preregistration — Contact-Mediated Consequence Transport

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R1/R1_RS2_REPORT.md`, `REPORT/REP/R1/R1_RS2A_REPORT.md`  
Does not change: `RS1C_GO`, `RS1B_GO`, `RS2_GO`, `RS2A_GO`, RS2-C0, \(s^\star\),
\(D_0\), thresholds, VoI, \(C_{\mathrm{tol}}\), RS1C policy

## Placement

\[
\boxed{
\begin{aligned}
RS2A &: \text{Can I see the mismatch? (closed)}\\
RS2B &: \text{If I see it, does my old estimate of “how much it matters” still transport?}
\end{aligned}
}
\]

Stage name:

\[
\boxed{
R1\text{-RS2B — Contact-Mediated Consequence Transport}
}
\]

中文名：**接触介导条件下的后果可迁移性**。

This is a **diagnosis** of the RS2 C2 allocation failure. It does not retune
RS2 and **does not reopen C1 detectability** (RS2A).

## Frozen (do not reopen)

- \(D_0\), cell thresholds, \(s^\star\), scripts, C0 geometry;
- RS1C policy, VoI, passivity, operator library;
- \(C_{\mathrm{tol}}\) as a **frozen Mode-A ruler**, not a quantity to retune;
- no revision install, no accept bit, no new detector.

RS2A remains closed. No \(X_\phi\), \(\kappa_\perp\), or detect gates here.

## Scientific question

\[
\boxed{
\text{Does Mode-A consequence }C
\text{ retain its decision meaning under contact-mediated interaction?}
}
\]

中文：在直接关节激励下测得的“如果不修正模型会造成多大后果”，到了机器人
接触介导的交互方式下，是否仍然是同一个决策量？

RS2 Formal C2 already showed

\[
D_0\text{ large},\quad \mathrm{detect}=1,\qquad C_{\mathrm{ModeA}}<C_{\mathrm{tol}},
\]

so the frozen policy labeled **tolerate**. That is not a visibility miss
(C1 / RS2A). It is: **saw the mismatch, judged it unimportant**.

## Why C1 and C2 stay split

| | C1 (RS2A) | C2 (this stage) |
|--|-----------|-----------------|
| Symptom | \(D_0\) too small | \(D_0\) large, `detect=1` |
| Frozen label | probe | tolerate |
| Failure | identifiability transport | consequence transport |
| Question | can I see it? | if I see it, does “how much it matters” transport? |

## Definitions

Frozen Mode-A consequence (unchanged RS1A.4 / RS2 Formal):

\[
C_{\mathrm{ModeA}}
=
\mathbb E_i\!\left[
\mathrm{RMSE}(\text{nominal},\text{truth})
-
\mathrm{RMSE}(\text{oracle-drag},\text{truth})
\right]
\]

on the intervention query set, hinge-torque `forecast_pair`.
`oracle-drag` is \(\alpha|v|v\) (zero on C0 and C2). For C2, latch is inside
**all** Mode-A rollouts, so \(C_{\mathrm{ModeA}}\) does not isolate latch harm.

Contact-domain quantities use the **same query ICs** \((q,v)\), frozen
`fast_pull` OSC tape recorded on C0 (incumbent interaction), horizon \(H_{32}=32\)
controller steps, and the frozen H32 state scale \((1,2)\):

\[
\begin{aligned}
L_{\mathrm{contact}}
&=
\mathbb E_i\!\left[\mathrm{RMSE}(\text{incumbent},\text{truth})\right],\\
C_{\mathrm{contact}}
&=
\mathbb E_i\!\left[
\mathrm{RMSE}(\text{incumbent},\text{truth})
-
\mathrm{RMSE}(\text{oracle-drag},\text{truth})
\right].
\end{aligned}
\]

Incumbent = C0 contact physics (no hidden drag, no latch).  
Truth = regime physics (C1 hidden drag and/or C2 latch).  
Oracle-drag = incumbent plus frozen \(\alpha|v|v\) (zero on C0/C2).

So \(C_{\mathrm{contact}}\) is the **same functional** as \(C_{\mathrm{ModeA}}\)
under a different intervention mechanism. \(L_{\mathrm{contact}}\) is the
incumbent’s raw forecast loss, which can see out-of-library latch harm
because truth has latch and incumbent does not.

Decision bits (frozen ruler, not retuned):

\[
\mathrm{consequential}_{\mathrm{ModeA}} = (C_{\mathrm{ModeA}}\ge C_{\mathrm{tol}}),
\quad
\mathrm{consequential}_{\mathrm{contact}}^{\Delta}
= (C_{\mathrm{contact}}\ge C_{\mathrm{tol}}).
\]

A protocol IC \((q,v)=(0.12,0)\) (frozen C0 hinge start) is logged in addition
to the eight intervention queries; the GO uses the eight-query mean, matching
Mode-A \(C\). Protocol \(L\) is reported.

## Matrix

Seeds: RS2 Formal \(\{11101,11111,11121,11131,11141\}\).  
Regimes: \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H},\mathrm{C2}\}\).  
One frozen script: `fast_pull` @ scale \(1.0\) (action tape only; not an
excitation sweep).

\[
N=5\times 4=\boxed{20\text{ cells}}.
\]

Smoke: seed `11101` × {C0, C2}. No GO.

## Hypotheses

### H0 — C0 sanity

Both rulers stay below \(C_{\mathrm{tol}}\):

\[
\boxed{
H0:\quad
C_{\mathrm{ModeA}}<C_{\mathrm{tol}}
\text{ and }
C_{\mathrm{contact}}<C_{\mathrm{tol}}
\text{ on all C0 cells}
}
\]

### H1 — in-library drag \(\Delta C\) (reported; not the C2 question)

On C1-L ∪ C1-H:

\[
\boxed{
H1a:\quad
P(\mathrm{consequential}_{ModeA}=\mathrm{consequential}_{contact}^{\Delta})\ge 0.8
}
\]

\[
\boxed{
H1b:\quad
\rho(C_{\mathrm{ModeA}},C_{\mathrm{contact}})>0
}
\]

If H1 passes, drag-consequence as a \(\Delta\)RMSE still transports. That does
**not** rescue C2.

### H2 — C2 primary: Mode-A \(C\) misses contact latch harm

\[
\boxed{
\begin{aligned}
H2a &\colon \text{all C2 have }C_{\mathrm{ModeA}}<C_{\mathrm{tol}}\\
H2b &\colon \mathrm{median}\,L_{\mathrm{contact}}^{\mathrm{C2}}
\ge C_{\mathrm{tol}}\\
H2c &\colon \mathrm{median}\,L_{\mathrm{contact}}^{\mathrm{C2}}
>
\mathrm{median}\,L_{\mathrm{contact}}^{\mathrm{C0}}
\end{aligned}
}
\]

H2 says: the frozen decision ruler calls C2 unimportant, while the incumbent
is actually a bad forecast of latch truth under contact. That is consequence
non-transport, not “detector threshold wrong.”

Report \(C_{\mathrm{contact}}\) on C2 (expected \(\approx 0\), because
oracle-drag \(\equiv\) incumbent). If that holds, \(C\) as a functional cannot
see out-of-library novelty in either domain; contact \(L\) is what reveals
the missing harm.

## GO

\[
\boxed{RS2B\_GO = H0 \land H2a \land H2b \land H2c}
\]

`RS2B_GO=true` does **not** set `RS2_GO=true`, does not retune \(C_{\mathrm{tol}}\),
and does not reopen RS2A. It only supports: Mode-A \(C\) does not retain
decision meaning for contact-mediated out-of-library harm.

H1 is reported. H1 failure means even in-library \(\Delta C\) is not
intervention-invariant; still no threshold edit.

## Failure interpretation

| Pattern | Read as |
|---------|---------|
| H0 ∧ H2 | C2 tolerate was a consequence-transport miss |
| H2b/c fail | latch did not hurt incumbent forecasts under this OSC tape; do not retune \(C\) |
| H1 pass, H2 pass | in-library \(\Delta C\) transports; out-of-library harm does not |
| H1 fail, H2 pass | \(C\) is not intervention-invariant even for drag |
| H0 fail | contact incumbent vs C0 truth is a new interface issue; stop |

None of these outcomes authorizes \(C_{\mathrm{tol}}\) search, \(s^\star\)
changes, or detectability retunes.

## Non-goals

- any \(D_0\) / \(\kappa_\perp\) / exposure analysis;
- VoI, passivity, H32 policy utility, installs;
- contact-specific \(C\) threshold;
- learned latch models or new operators.
