# TASK-XL Preregistration — Learned Task / Progress Latents (TASK-X retry)

Date: 2026-08-27  
Status: **FROZEN**; **DESIGN ONLY**; **LOCKED** (do not train). Queue is
**RTWX-X0S** then ID then capacity; X0R `instrument_ready` is false and
is **not** an XL unlock. Hand-phase G2 already failed on a frozen probe;
incomplete RoboTwin demos plus a new latent would add confounders.  
Depends on: `REPORT/REG/TASKX/TASKX_PREREG.md`;
`REPORT/REP/TASKX/TASKX0_REPORT.md` (honest two-class reading);
`REPORT/REG/RTWX/robot_dynamics/RTWX0R_PREREG.md` (queue)  
Does not: hand-rewrite phase tables as the retry; mix \(z_g,z^p\) into
\(F_{\mathrm{physics}}\); RGB as first XL host; PLAN-X2 on `capx_arm3`;
RoboTwin-X0 `physics_predict`; capacity \(R_P\); R10

## Orthogonality

\[
\boxed{
\text{RoboTwin-X0R repairs dynamics I/O.}
\quad
\text{TASK-X0 tested a \emph{hand} phase instrument.}
\quad
\text{TASK-XL tests a \emph{learned} task/progress split.}
\quad
\text{TASK-X1 (diffusion attractors) stays LOCKED.}
}
\]

Priority (unchanged family):

```text
RoboTwin-X0R  >  TASK-X retry (this cell)  >  visual  >  diffusion
```

TASK-X retry **means** learned \(z_g,z^p\), **not** a new hand phase
list. Diffusion remains a **later** cell, only after an XL instrument
**PASS**. This preregistration must not be read as a license to train
\(\epsilon_\phi\).

## Thesis (frozen)

Hand-written phase tables are **not** the intended long-run
representation of task progress. Split **two** latents.

**Task latent** (slow; approximately constant within an episode):

\[
z_g = E_g(g).
\]

It answers “what am I doing?” It is bound to the task: task identifier,
language, goal image, and/or target object–region. It is **not** a
frame-wise pose code.

**Progress latent** (fast):

\[
z^p_t = E_p(h_t, z_g),
\qquad
h_t=(s_{0:t},\,a_{0:t-1})
\text{ or belief }b_t.
\]

It is **not** \(f(s_t)\) alone. **Task–state aliasing:** the same
instantaneous configuration (arm near cup) can be pre-grasp or
post-place; \(s_t\) is not a unique task state.

Both are learned from **successful trajectories** without manual
approach / grasp / transport / … labels. Manifold points need **not**
carry human names.

## Task-progress manifold (what \(z^p\) is for)

A progress coordinate is a useful planning state if the set
\(\{z^p_t\}\) on successful trajectories of the **same** task is:

1. **ordered** along completion (not an unordered cluster);
2. **predictive** of future actions \(A\) and of completion;
3. **consistent** across successful trajectories of that task.

Human phase names are optional annotations, not the representation.

## Supervision menu (document all)

Legal: future quantities as **targets**. Illegal: future as **encoder
inputs** (anti-leakage below).

| id | target | role |
|---|---|---|
| **weak** | \(y_t=t/T\) | cheap causal clock; optional ablation encoder |
| **A** | remaining distance / value \(T-t\) or remaining cost | time-to-go / cost-to-go |
| **B** | future action prediction \(a_{t:t+H}\) | closest to planning |
| **C** | future key events (grasp / contact / done / relation change) | **targets only**, never \(E_p\) inputs |
| **D** | trajectory order contrastive \(i<j\) | ordinal consistency |

**Primary scientific test (B):**

\[
\boxed{
H(A\mid s,z_g,z^p)
\;<\;
H(A\mid s,z_g)
}
\]

That is the planning-relevant compression claim. Weak / A / C / D may
shape \(z^p\); they do not replace B as the first instrument’s PASS
criterion.

### First X0-like instrument (frozen; not run here)

**TASK-XL0** (design freeze only):

- **Encoder training target (first):** **B** — future action chunk
  \(A_t=a_{t:t+H}\) as TARGET for \(E_p\) (and a frozen or jointly
  trained \(q(A\mid s,z_g,z^p)\) probe). Weak \(y_t=t/T\) is a
  documented cheap baseline encoder for ablation, **not** the first
  PASS gate.
- **Instrument gate (X0-like):** held-out action NLL / entropy proxy,
  same spirit as TASK-X0 G2, with **no** diffusion train and **no**
  \(S_{\mathrm{task}}\) claim:
  \(\mathrm{NLL}(A\mid s,z_g,z^p)<\mathrm{NLL}(A\mid s,z_g)\);
  paired \(\Delta\mathrm{NLL}\) 95% CI does not cross 0.
- **Primary baseline:** no-progress \(q(A\mid s,z_g)\) (goal/task
  latent only). The retry is **learned \(z^p\) vs no-progress**, not
  rewrite phases.
- **A, C, D:** later / auxiliary. C events must not enter \(E_p\).
- **Diffusion attractor:** **forbidden in XL0.** Allowed only as a
  **later** cell after instrument PASS. Family order still
  X0R \(>\) TASK-X retry \(>\) visual \(>\) diffusion.

Do not freeze XL1 widths / \(T_{\mathrm{diff}}\) / train budget in this
document. Those knobs stay locked with TASK-X1.

## Anti-leakage (hard)

\[
\boxed{
E_p(s_{\le t},\,a_{<t},\,g)
\text{ only.}
}
\]

Future may be TARGET. **Illegal:** encoder sees \(s_{t+1:T}\), future
actions as inputs, or future success, then the same map is treated as
causal at test. Prefix recoverability is mandatory: \(z^p_t\) from
\((s_{0:t},a_{0:t-1},g)\) must equal \(z^p_t\) from the full episode
restricted to that prefix.

## Action attractor (later cell only)

\[
q(A\mid s_t,z_g,z^p_t).
\]

A matched diffusion \(q\) is **not** opened by this freeze. PLAN-X
`iso_sufficient` on `capx_arm3` is unchanged. Do not convert “need a
better progress coordinate” into “need a stronger generator.”

## Optional hierarchy (document; do not implement)

\[
z_g \;\to\; z^{\mathrm{subgoal}}_t \;\to\; z^p_t \;\to\; a_t.
\]

A subgoal latent is a possible later split. This cell implements
**neither** the hierarchy nor training.

## APR-WM state box (family)

\[
X_t=\bigl(
s^{\mathrm{phy}}_t,\;
\theta_E,\;
b^{\mathrm{obs}}_t,\;
z_g,\;
z^p_t,\;
z^{\mathrm{res}}_t
\bigr)
\]

\[
\boxed{
F_{\mathrm{physics}}
\text{ eats physics-related coordinates only;}
\quad
q(A\mid\cdot)
\text{ eats }
(s^{\mathrm{phy}},z_g,z^p).
}
\]

Task semantics must not rewrite physical law. \(z^{\mathrm{res}}\) is
the dynamics residual slot (CAP / RoboTwin), not a second progress
code. Observation belief \(b^{\mathrm{obs}}\) is not an XL0 input
requirement.

## Link to TASK-X0 (honest reading; do not collapse classes)

TASK-X0 **RAN / family FAIL**. That result is an **instrument /
representation** failure of a **hand** phase table plus a frozen kNN
probe. It is **not** a warrant that task progress has no planning
value, and it is **not** a warrant that a learned \(z^p\) is worthless.

Keep the two X0 classes:

1. **Cabinet `coverage_hole`:** instrument / data chain insufficient
   (missing object–cabinet pose, hollow coverage). Does **not** say
   \(p_t\) has no decision value.
2. **Cup / stamp `p_no_nll`:** this hand \(p_t\) did not further
   compress \(H(A\mid s,g)\) on the frozen kNN. Does **not** prove a
   learned \(z^p\) is worthless; it also does **not** prove the
   opposite. Do not rewrite those two classes.

**Next TASK-X retry test:**

\[
\boxed{
\textbf{learned progress latent vs no-progress baseline}
}
\]

**Not:** rewrite phases. **Still:** if the host is RoboTwin
demonstrations, **first** fix coverage and object-state logging.
Do not run XL0 on the same cabinet hole and call it a \(z^p\) verdict.

Hand phases may have been **misaligned** with the action distribution
(named stages that do not carve action basins). That is a reason to
change the **representation family**, not to open diffusion.

## Host / data (when unlocked)

Official RoboTwin successful demos (`place_empty_cup`,
`put_object_cabinet`, `stamp_seal`) only after object-state logging is
auditable. Do not use RoboTwin-X0 random-wrench near-static traces as
progress data. Do not attach unclosed `physics_predict`.

Unlock conditions (all):

1. RoboTwin-X0R has run its own instrument cell (this family does not
   jump the queue);
2. object-state coverage is sufficient for the chosen task, **or** a
   non-RoboTwin host with closed object state is declared in an XL0
   header **before** first parse;
3. XL0 numeric floors (NLL CI, \(H\), seeds) frozen in that header
   **before** seeing curves.

This document does **not** start XL0.

## One question (XL0, when later run)

\[
\boxed{
\text{On successful demonstrations, does a causally encoded learned
}z^p\text{ reduce action-selection uncertainty beyond }(s,z_g)\text{?}
}
\]

XL0 must **not** claim task success, search \(AUC_S\), or visual
\(\hat z^p\).

## Claims ceiling

**This freeze** may say: the long-run TASK-X representation is
\((z_g,z^p)\); hand phases were an X0 instrument; retry is learned
progress vs no-progress; B is the primary test; leakage is forbidden;
diffusion is later; X0R remains NEXT.

**This freeze may not say:** XL0 PASSed; learned \(z^p\) works;
TASK-X1 is unlocked; R10 is in scope; X0 `p_no_nll` is reversed.

## Route (frozen)

```text
TASK-X0   hand-phase instrument     RAN / FAIL
TASK-X1   oracle p + diffusion      LOCKED  (not the retry)
TASK-XL   learned z_g, z^p family   FROZEN; after instrument_ready
TASK-XL0  X0-like NLL instrument    LOCKED until instrument_ready + object-state
          (B-gate; no diffusion)
TASK-XL1  attractor / planning      LOCKED until XL0 PASS
          (diffusion only if later cell after PASS)
TASK-X2   visual                    LOCKED
```

## Ledger

```text
CAP-X3            = PASS
PLAN-X            = CLOSED at iso_sufficient
PLAN-X2           = LOCKED

RoboTwin-X0       = RAN; capacity claim WITHHELD
RoboTwin-X0R      = FAIL / excitation_failure; G1 PASS
                    bottleneck = structure / force accounting
RTWX-X0S          = RAN / FAIL timing_mismatch
RTWX-X0F          = RAN / FAIL force_channel_unresolved

TASK-X0
├── cup      = FAIL  p_no_nll
├── cabinet  = FAIL  coverage_hole
└── stamp    = FAIL  p_no_nll

TASK-XL           = DESIGN FROZEN ONLY; LOCKED
TASK-XL0          = LOCKED (not run)
TASK-X1           = LOCKED  (diffusion; not opened by XL)
TASK-X2           = LOCKED
RoboTwin-X1       = LOCKED
R10               = LOCKED
```

Stub (API only, no train): `aprwm_v0/task_xl.py`,
`tests/test_task_xl.py`.
