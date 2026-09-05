# R1-RS2 Preregistration — Contact-Mediated Epistemic Revision

Date: 2026-08-16  
Status: **FROZEN; RS2-C0 passed; formal complete; RS2_GO=false**  
Depends on: `REPORT/REP/R1/R1_RS1C_REPORT.md` (`RS1C_GO=true`)  
Does not change: `RS1B_GO=false`; no RS1B.3; no support hard veto  
Frozen implementation: `aprwm_v0/r1_rs2.py`  
Frozen C0 report: `REPORT/REP/R1/R1_RS2_C0_REPORT.md`

## Placement

\[
\boxed{
RS1C\_GO=true
\rightarrow
RS2\text{-C0 contact closure}
\rightarrow
\text{exposure-adapter freeze}
\rightarrow
RS2\text{-Formal}
}
\]

Stage name:

\[
\boxed{
R1\text{-RS2 — Contact-Mediated Epistemic Revision}
}
\]

RS2 adds one scientific variable: **robot contact**. It does not add RGB,
a learned contact estimator, a learned policy, multiple objects/tasks, a new
operator library, or a new support definition.

## Primary scientific question

\[
\boxed{
\text{Does the frozen RS1C policy remain valid when Door dynamics are
excited through robot contact rather than direct hinge torque?}
}
\]

Main hypothesis:

\[
\boxed{
\text{The frozen RS1C epistemic-revision policy remains useful and
physically safe under robot-contact excitation, provided contact
generalized forces are explicitly accounted for.}
}
\]

The deeper question is whether epistemic control survives
interaction-mediated observation: can the system still decide when to
tolerate, probe, and revise when evidence is produced by embodied action?

## Scope and non-goals

Frozen environment:

- `robosuite==1.5.2`, `mujoco==3.11.0`, `Door`, `Panda`;
- oracle state; no cameras;
- deterministic scripted OSC control, not a learned policy;
- one Door asset and one frozen candidate library;
- oracle robot–Door contact generalized force;
- CNEG excluded from primary confirmatory matrix (may be a later diagnostic).

Explicit non-goals:

- contact-force estimation;
- RGB / tactile perception;
- policy learning or task success optimization;
- Wipe / ToolHang / multi-object transfer;
- RS1B.3 or any retuning of \(d_{\mathcal S}\);
- detector, VoI, support, operator, or accept-threshold search.

## Contact-aware force interface

Mode-A used direct hinge torque. RS2 uses the Door-hinge row of the coupled
robot–Door dynamics:

\[
\boxed{
\left[M(q)\ddot q+q_{\mathrm{bias}}(q,\dot q)\right]_{h}
=
\tau_{\mathrm{act},h}
+\tau_{\mathrm{contact},h}
+\tau_{\mathrm{nominal},h}
+r_{\tau,h}.
}
\]

The oracle contact term is

\[
\boxed{
\tau_{\mathrm{contact}}
=
\sum_{c\in\mathcal C_{\mathrm{robot,Door}}}
J_c^\top f_c,
}
\]

where only contacts with one robot geom and one Door geom enter the sum.
The implementation must use MuJoCo contact forces and point Jacobians with
the correct frame/sign. The Door-hinge component is learner-visible as an
**oracle known input**.

The learner residual is:

\[
\boxed{
r_{\tau,h}
=
\left[M\ddot q+q_{\mathrm{bias}}\right]_{h}
-\tau_{\mathrm{act},h}
-\tau_{\mathrm{contact},h}
-\tau_{\mathrm{nominal},h}.
}
\]

`tau_nominal` may include only declared damping and the known Door-hinge
frictionloss constraint row. Contact rows are handled by \(J_c^\top f_c\).
Joint-limit, latch, equality, and other unknown constraint rows must **not**
be silently subtracted.

### Leakage prohibitions

Learner-visible code must not:

- use full `qfrc_constraint` as known force;
- use full `qfrc_applied` as known force;
- read truth `qfrc_passive`;
- subtract hidden \(\alpha |v|v\), latch novelty, limit force, or revision
  force from the label;
- count normal robot contact as residual.

Raw truth may log these fields for audit, but learner-visible artifacts
receive only the projected oracle contact scalar, declared nominal terms,
state, and scripted controller command.

### Independent contact audit

For every C0 frame, compute contact generalized force two ways:

1. \(\tau_{\mathrm{contact}}^{J^\top f}\) from `mj_contactForce` and point
   Jacobians;
2. \(\tau_{\mathrm{contact}}^{\mathrm{efc}}\) from contact-typed MuJoCo
   constraint rows only.

The learner uses method 1. Method 2 is audit-only. Disagreement is an
interface failure, not a scientific no-go.

## Frozen RS1C policy

\[
\boxed{\pi_{\mathrm{RS1C}}\text{ is frozen in RS2}}
\]

The following are immutable:

- \(C_{\mathrm{tol}}\);
- frozen per-exposure \(D_0\) thresholds;
- tolerate / probe / revise-worthy definitions;
- \(\lambda_E=0.0015\), \(c(A)=(A/A_0)^2\);
- \(A^\star=1.5A_0\), represented through the contact adapter below;
- frozen proposal/operator library;
- physical admissibility and passivity;
- H2/H4/H8 short utility;
- install decision before blind H32;
- existing \(d_{\mathcal S}\), \(I_{\mathrm{exit}}\), and support coding;
- support used only for monitoring.

\[
\begin{aligned}
I_{\mathrm{exit}}=0
&\Rightarrow \texttt{monitor\_intensity=normal},\\
I_{\mathrm{exit}}>0
&\Rightarrow \texttt{monitor\_intensity=elevated};
\quad\text{revision retained}.
\end{aligned}
\]

Support must never flip `accepted`.

## Contact excitation adapter

Direct torque amplitude is not identified with an OSC command. RS2 uses
**exposure matching**:

\[
X_\phi(u)=\operatorname{mean}(\dot q_h^4).
\]

Frozen Mode-A exposure references are medians from the already completed
RS1A.5 formal artifact:

| Equivalent \(A/A_0\) | \(X_\phi^{\mathrm{ModeA}}\) |
|----------------------:|---------------------------:|
| 0.5 | \(1.9487441\times10^{-15}\) |
| 1.0 | \(6.3415677\times10^{-7}\) |
| 1.5 | \(7.8029109\times10^{-4}\) |
| 2.0 | \(4.7586901\times10^{-3}\) |

Each fixed baseline script is assigned once to the nearest reference in
log-exposure (with \(10^{-15}\) floor). Its frozen RS1A.5 threshold is then
used; no contact-specific detector threshold is fitted.

The extra VoI action is one standardized deterministic pull-release probe
\(u_s(t)\). Only its pull displacement is scaled:

\[
u_s(t)=u_{\mathrm{contact}}+
s\left(u_0(t)-u_{\mathrm{contact}}\right),
\quad
s\in\{0.75,1.0,1.25,1.5,1.75,2.0\}.
\]

On C0 development seeds, select the **smallest safe** \(s\) minimizing

\[
\left|\log
\frac{\operatorname{median}X_\phi(u_s)}
{7.8029109\times10^{-4}}
\right|.
\]

Safe means finite, no joint-limit violation, robot–Door contact is active,
and the Door moves at least \(0.05\) rad. The selected probe must match the
target within \(\pm20\%\). Otherwise RS2-C0 fails and formal remains locked;
do not extrapolate the grid.

The exact OSC controller configuration, waypoints, timing, gripper command,
selected \(s^\star\), script-to-exposure-bin mapping, and manifest SHA256 are
frozen in the RS2-C0 report **before** any formal seed is run. C0 development
may tune only contact feasibility and this adapter on dev seeds; it may not
change RS1C.

## Stage 1 — RS2-C0 contact-aware closure

Development seeds:

\[
\{11001,11011,11021\}
\times
\{\text{slow pull, fast pull, pull-release}\}
\times
3\text{ deterministic repeats}
=27\text{ episodes}.
\]

The three scripts are deterministic Cartesian OSC waypoint trajectories.
Repeat \(r\in\{0,1,2\}\) uses sub-seed `seed + 100*r`; the policy and
waypoints are unchanged.

C0 contains no hidden operator and no latch. All robot joints move through
the controller; no direct Door-hinge torque is allowed.

Define contact-active frames per episode by

\[
|\tau_{\mathrm{contact},h}|
>
0.05\,P_{95}(|\tau_{\mathrm{contact},h}|),
\]

and require \(P_{95}(|\tau_{\mathrm{contact},h}|)>0.02\) Nm.

### C0 gates

All must pass:

1. all 27 episodes finite;
2. every script/repeat moves the Door by at least \(0.05\) rad;
3. contact-active fraction \(\ge10\%\) in every episode;
4. \(J^\top f\) vs contact-`efc` projection:
   \(\mathrm{NRMSE}<10^{-6}\) in every cell;
5. complete and contact-active residual closure:
   \(\mathrm{NRMSE}_\tau<10^{-3}\) in every cell, normalized by
   \(\mathrm{RMS}(\tau_{\mathrm{contact},h})+10^{-8}\);
6. C0 false revision \(=0/27\) (the preregistered \(\le1\%\) rate is
   operationally zero at this sample size);
7. exposure adapter matches \(X_\phi(1.5A_0)\) within \(\pm20\%\);
8. artifact audit confirms no forbidden truth field in learner-visible data.

`RS2-C0_PASS=false` is `infrastructure/interface_block`, not a test of the
RS1C scientific hypothesis. On failure, repair contact accounting or script
feasibility and rerun C0 with a new manifest/version; never inspect formal
seeds.

### Post-C0 adapter / manifest freeze

RS2-C0 v2 passed on 2026-08-16. Before any formal seed was run, the following
development outputs were frozen:

- implementation: `aprwm_v0/r1_rs2.py`;
- passing artifact: `runs/r1_rs2/c0_v2/summary.json`;
- scripted OSC manifest SHA256:
  `3dee8f6160e041b1a5a1499d3e011c5e62a39f378b9a21039ab6997f7e5ab89e`;
- fixed Panda start posture and fixed Door pose are recorded in
  `script_manifest.json`; robot initialization noise is disabled;
- contact acquisition starts from interior Door hinge state \(q_h=0.12\);
- \(s^\star=2.0\);
- median probe exposure
  \(X_\phi=9.2726248\times10^{-4}\), relative target error \(18.84\%\);
- baseline script exposure bins:
  `slow_pull -> 1.0A0`, `fast_pull -> 1.5A0`,
  `pull_release -> 1.0A0`.

The fixed geometry makes the 27 C0 cells deterministic closure replicates;
their purpose is interface closure, not seed-level scientific inference.
Formal seeds remain held out. Formal may not retune this pose, posture,
controller, waypoint timing, gripper command, adapter scale, or exposure-bin
mapping.

## Stage 2 — RS2 Formal

Formal seeds are new and held out from all C0/RS1 work:

\[
\{11101,11111,11121,11131,11141\}.
\]

Regimes:

| Regime | Dynamics |
|--------|----------|
| C0 | adequate contact-aware model |
| C1-L | frozen \(r_\tau=-0.12|\dot q|\dot q\) |
| C1-H | frozen \(r_\tau=-0.24|\dot q|\dot q\) |
| C2 | existing outside-library latch novelty |

C2 uses the same scripted robot–Door contact as the other regimes. Latch
constraint force is excluded from known contact/nominal accounting so C2
novelty cannot be canceled. Report contact exposure matched by regime to
verify that “contact occurred” and “novel latch physics occurred” are
separable.

Matrix:

\[
5\text{ seeds}
\times
4\text{ regimes}
\times
3\text{ scripts}
\times
3\text{ repeats}
=
\boxed{180\text{ episodes}}.
\]

The statistical unit is seed. Repeats and scripts are nested within seed.

H32 is 32 robosuite controller steps at 20 Hz (1.6 s), with all MuJoCo
substeps retained. Truth, no-revision, and revised counterfactuals share the
same initial state and scripted controller commands; each rollout computes
its own oracle \(J^\top f\) from its own contact state. H32 is blind to
intake/accept thresholds.

## Policy baselines

All are evaluated on the same formal episodes:

| Policy | Rule |
|--------|------|
| \(\pi_{\mathrm{none}}\) | never revise |
| \(\pi_{\mathrm{RS1C}}\) | frozen layered policy |
| \(\pi_{\mathrm{veto}}\) | counterfactually drop an RS1C install when \(I_{\mathrm{exit}}>0\) |
| \(\pi_{\mathrm{always}}\) | bypass consequence/VoI allocation; generate a candidate on every episode, but retain frozen passivity + short-utility install checks |

Only \(\pi_{\mathrm{RS1C}}\) is the tested policy. Baselines cannot change its
accept bit.

For \(\pi_{\mathrm{always}}\), do not compare RMSE alone. Report per seed:

\[
\left(G_s,\ E_s,\ R_s\right),
\]

where \(G_s\) is H32 gain, \(E_s=2.25\,N_{\mathrm{extra\ probe}}/N\), and
\(R_s=N_{\mathrm{install}}/N\). Also report the preregistered sensitivity:

\[
J_s(\lambda_R)
=G_s-0.0015E_s-\lambda_RR_s,
\quad
\lambda_R\in\{0,2.5{\times}10^{-4},5{\times}10^{-4},10^{-3}\}.
\]

Because no revision-cost coefficient was validated before RS2,
\(\pi_{\mathrm{always}}\) net value is a **secondary sensitivity analysis**,
not a GO gate. No \(\lambda_R\) may be selected after outcomes.

## Primary endpoints

### Gate 1 — C0 specificity

\[
P(\mathrm{install}\mid C0)\le2\%.
\]

With 45 C0 formal episodes this operationally requires **0 false installs**.

### Gate 2 — tolerate safety

\[
\#\{\mathrm{install}\cap\mathcal T\}=0.
\]

### Gate 3 — physical admissibility

\[
\#\{\text{passivity violation among installs}\}=0.
\]

### Gate 4 — H32 policy utility

For each seed,

\[
G_s=
\mathrm{RMSE}^{\pi_{\mathrm{none}}}_{H32,s}
-
\mathrm{RMSE}^{\pi_{\mathrm{RS1C}}}_{H32,s}.
\]

Require:

\[
\operatorname{mean}_sG_s>0
\quad\land\quad
\operatorname{mean}_sG_s^{\mathrm{RS1C}}
\ge
\operatorname{mean}_sG_s^{\mathrm{veto}}.
\]

Report paired seed-level 95% CI. CI lower bound \(>0\) is a stronger result,
but is not the frozen GO threshold.

### Gate 5 — VoI path is exercised

Require:

- at least 10 initial probe episodes;
- at least one probe \(\rightarrow\) revise-worthy promotion;
- promotion occurs in at least 3 of 5 formal seeds.

Report promotion fraction and post-promotion install/utility. This gate
prevents a nominal pass in which contact causes almost everything to be
directly detected or tolerated.

### Gate 6 — monitoring invariance

Every installed episode with \(I_{\mathrm{exit}}>0\) must be retained and
marked `elevated`; every installed episode with \(I_{\mathrm{exit}}=0\) must
be `normal`; support-triggered accept flips must equal zero.

## RS2 GO

\[
\boxed{
\begin{aligned}
RS2\_GO
=&\ (\text{RS2-C0 PASS})\\
&\land(\text{C0 specificity})\\
&\land(\text{tolerate false install}=0)\\
&\land(\text{passivity violations}=0)\\
&\land(\text{seed-mean H32 gain}>0)\\
&\land(\text{gain vs veto}\ge0)\\
&\land(\text{VoI path exercised})\\
&\land(\text{monitor consistency}).
\end{aligned}
}
\]

Report, but do not gate on:

- contact-active fraction by regime/script;
- \(I_{\mathrm{exit}}\), \(D_{\mathrm{exit}}\), RMSE, gain, and harmful
  fraction;
- \(P(\mathrm{harmful}\mid I_{\mathrm{exit}}>0)\);
- `always` cost-sensitive baselines;
- C2 rejection/unknown diagnostics;
- CI lower-bound sign.

Harmful retains the frozen definition:

\[
\mathrm{harmful}
=
(\text{not finite})\lor(\Delta\mathrm{RMSE}<0).
\]

## Failure interpretation

- C0 closure or contact audit fail: interface/infrastructure block; do not
  run formal.
- C0 false installs: contact accounting or specificity failure.
- VoI gate fail: epistemic-control path was not exercised under contact.
- passivity fail: physical-admissibility transfer failure.
- H32 gain fail: revision utility did not transfer through interaction.
- monitoring fail: forbidden support leakage into accept.

None of these outcomes authorizes RS1B.3, support-distance retuning, detector
retuning, or post-hoc contact-force estimation.
