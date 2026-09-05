# TASK-X Preregistration — Task-Progress-Conditioned Action Planning

Date: 2026-08-27
Status: **FROZEN (family)**; **TASK-X0 RAN / FAIL**; **TASK-X1 LOCKED**;
**TASK-XL DESIGN FROZEN / LOCKED**; **X0S/X0F RAN FAIL** (force channel unresolved);
TASK-X2 LOCKED; do not train XL0
Depends on: RoboTwin official demo pipeline + oracle state (not
RoboTwin-X0 physics predictor; not X0R G-pass)
Does not: mix into CAP-X \(R_P\); reopen PLAN-X2 on `capx_arm3`;
RoboTwin-X1 RGB; R10; change \(F_{\mathrm{physics}}\) with \(p_t\) or
\(z^p\); train TASK-XL / diffusion in this freeze
Pointer: learned-progress retry `REPORT/REG/TASKX/TASKXL_PREREG.md`

## Orthogonality

\[
\boxed{
\text{CAP-X / RoboTwin-X0R answer dynamics capacity.}
\quad
\text{PLAN-X answered search shape on } \texttt{capx\_arm3}.
\quad
\text{TASK-X answers task-progress as compressed planning state
(long-run: learned }z_g,z^p\text{; X0 used a hand phase instrument).}
}
\]

PLAN-X2 / diffusion on the unimodal arm host stays **LOCKED**
(`iso_sufficient`). TASK-X uses RoboTwin **successful demonstrations**
and a progress-conditioned action attractor. That is not a silent
rename of PLAN-X2.

Paper symmetry (framing; not an X0 claim):

\[
\boxed{
\begin{array}{ll}
\text{CAP-X3:} &
\text{causal physics coordinates compress world-adaptation freedom}\\[1mm]
\text{TASK-X:} &
\text{task-progress coordinates compress action-planning freedom}
\end{array}
}
\]

Hypothesis TASK-X actually tests:

\[
\boxed{
\textbf{Can task progress, like physics parameters, compress
decision-relevant information that is otherwise implicit in
high-dimensional history into a few structured state variables?}
}
\]

Not: “does a task embedding bump success.”

## One question

\[
\boxed{\textbf{TASK-X — Task-Progress-Conditioned Action Planning}}
\]

\[
\boxed{
\textbf{Does explicit task progress }p_t\textbf{ reduce action-selection
uncertainty and raise long-horizon planning efficiency?}
}
\]

## Architecture (frozen)

World state \(s_t=\{\text{robot},\text{object},\theta,\ldots\}\).
Goal \(g\) (e.g. put object into cabinet). Progress \(p_t\).

\[
\boxed{
q_\phi(A_t\mid s_t,g,p_t),\qquad
A_t=(a_t,\ldots,a_{t+H_a-1})
}
\]

\[
(s_t,g,p_t)
\rightarrow
\text{action attractor}
\rightarrow
A^{(1)},\ldots,A^{(K)}
\rightarrow
\text{WM / simulator rollout}
\rightarrow
\text{task cost}
\rightarrow
A^\star.
\]

Non-negotiable:

\[
\boxed{
p_t,g\rightarrow\text{action selection},
\qquad
p_t,g\nrightarrow F_{\mathrm{physics}}.
}
\]

Task semantics must not rewrite physical law.

Long-run state box (`TASKXL_PREREG.md`):

\[
X_t=(s^{\mathrm{phy}}_t,\theta_E,b^{\mathrm{obs}}_t,z_g,z^p_t,z^{\mathrm{res}}_t)
\]

\(F_{\mathrm{physics}}\) eats physics-related coordinates only;
\(q(A\mid\cdot)\) eats \((s^{\mathrm{phy}},z_g,z^p)\). X0 used explicit
\((s,g,p)\) as the instrument stand-in.

X1 first round: **oracle** \(p_t\), **oracle / native RoboTwin state**,
**simulator** (or frozen env) for candidate scoring. Do **not** attach
the unclosed RoboTwin-X0 `physics_predict`. If X1 fails, the failure
is the progress representation, not dynamics or perception.

## Progress representation — long-run vs X0 instrument

Hand-written phase tables are **not** the intended long-run
representation. The frozen retry is TASK-XL: task latent
\(z_g=E_g(g)\) (slow) and progress latent
\(z^p_t=E_p(h_t,z_g)\) with \(h_t=(s_{0:t},a_{0:t-1})\) or \(b_t\),
**not** \(f(s_t)\) alone (task–state aliasing). See
`TASKXL_PREREG.md`. TASK-X0 below remains the **ran** hand-phase
instrument; do not rewrite it as XL.

### X0 instrument (v1: graph + predicates; historical)

Not an arbitrary latent. For `put_object_cabinet` **X0 only**:

\[
p_t=(k_t,c_t)
\]

\[
k_t\in\{\mathrm{approach},\mathrm{grasp},\mathrm{transport},\mathrm{insert},\mathrm{release},\mathrm{done}\}
\]

\[
c_t=[\mathrm{object\_grasped},\mathrm{object\_lifted},\mathrm{near\_cabinet},\mathrm{inside\_target},\mathrm{released}]
\]

\[
p_{t+1}=G(p_t,s_t,s_{t+1},g)
\]

\(p_t\) is a **compressed task-history sufficient statistic**, not a
semantic embedding stuffed into the policy.

Cup / stamp use task-specific phase sets (same contract):

| task | phases (intent) |
|---|---|
| `place_empty_cup` | approach → grasp → lift → transport → place → release |
| `put_object_cabinet` | as above (object–container / insert) |
| `stamp_seal` | grasp → align → press → complete |

### Anti-leakage

\(p_t\) may use only current/past world state, current \(g\), and
predicates already true. **Forbidden:** future action, future success,
scripted next expert phase, demonstration future labels.

First round:

\[
\boxed{p_t=p_t^{\mathrm{oracle}}=G_{\mathrm{frozen}}(s_{\le t},g)}
\]

Do **not** learn a progress estimator in **X0/X1**. Otherwise those
cells cannot separate “hand \(p\) worthless” from “\(\hat p\) wrong.”
Learned \(z^p\) is **TASK-XL**, after X0R, not a rewrite of this \(G\).

## Route (frozen)

```text
TASK-X0   hand-phase instrument on demos (no diffusion)     RAN / FAIL
TASK-X1   oracle p + matched diffusion attractors           LOCKED
TASK-XL   learned z_g, z^p  (retry; not rewrite phases)     FROZEN
          NEXT-after-X0R; LOCKED until X0R / object-state
TASK-XL0  X0-like NLL: H(A|s,z_g,z^p)<H(A|s,z_g)            LOCKED
TASK-X2   visual                                            LOCKED
```

X0 did **not** PASS, so TASK-X1 widths / diffusion steps / \(H_a\) stay
**unfrozen and unrun**. Do not treat TASK-XL as a silent TASK-X1.
Diffusion remains a **later** cell after an XL instrument PASS.
Priority: X0R \(>\) TASK-X retry \(>\) visual \(>\) diffusion.

## Host / data

Official RoboTwin: `place_empty_cup`, `put_object_cabinet`, `stamp_seal`.
Use the repo’s demonstration collector and saved qpos/endpose (and
built-in Diffusion Policy stack if present). Do **not** rebuild an
action-chunk pipeline from scratch. Do **not** use RoboTwin-X0 random-wrench
trajectories (those are near-static).

If even these three show no value for explicit progress, do **not**
scale to 20–50 tasks.

## Comparisons (X1; freeze after X0 PASS)

| id | condition |
|---|---|
| **B0** | \(q(A\mid s,g)\) — primary baseline |
| **B1** | \(q(A\mid s,g,p)\) — primary model |
| **B2** | \(q(A\mid s_{t-L:t},g)\) — history, no explicit \(p\) |
| **B3** | same as B1 with shuffled \(p_{\pi(t)}\) |

Require **B1 > B3** (else extra channels, not progress).

B2 answers: is explicit progress a better **compression** of task
history than raw history? Not “does \(p\) add bits.”

Matched in X1: action horizon, diffusion steps, backbone, parameter
budget, demonstrations, optimizer, test seeds. Main variable: **whether
\(p_t\) is explicit**.

Attractor (X1 only):

\[
A_\tau=\sqrt{\bar\alpha_\tau}A^\star+\sqrt{1-\bar\alpha_\tau}\epsilon,\quad
\mathcal L=\mathbb E\|\epsilon-\epsilon_\phi(A_\tau,\tau,s,g,p)\|^2.
\]

Same \(s\), different \(p\), different action basins.

## Metrics (X1; not imitation-only)

| id | metric |
|---|---|
| **M1** | episode \(S_{\mathrm{task}}\) — **primary** |
| **M2** | \(\Delta P=P(p_{t+H})-P(p_t)\) progress advancement |
| **M3** | \(R_{\mathrm{wrong-phase}}\) |
| search | \(B\in\{1,2,4,8,16\}\); \(S(B)\); \(AUC_S=\mathrm{AUC}_{\log B}S(B)\); \(B_{80}=\min\{B:S(B)\ge 0.8\}\) |

Ideal: not only \(S\uparrow\), but **fewer candidates for the same success**.

Auxiliary (not core capacity): \(R_{\mathrm{task-state}}=1-d_p/(L d_s)\);
condition-encoder params; inference FLOPs; demo sample efficiency.

## Formal gates (X1)

**G-task:** \(S^{\mathrm{progress}}_{\mathrm{task}}>S^{\mathrm{goal\text{-}only}}\)
paired bootstrap 95% CI \(>0\).

**G-search:** \(AUC_S^{\mathrm{progress}}>AUC_S^{\mathrm{goal\text{-}only}}\), CI \(>0\).

**G-history:** \(S^{\mathrm{progress}}\ge S^{\mathrm{history}}-\delta\) with
strictly smaller condition tokens / history. Progress is an **effective
low-dimensional compression**, not “more information than history.”

\(\delta\) frozen in the X1 header before the first X1 eval.

`task_x1_passed` iff G-task \(\land\) G-search \(\land\) G-history
\(\land\) (B1 > B3).

## Patterns (X1)

| pattern | meaning |
|---|---|
| **A** `progress_planning_advantage` | B1>B0 and B1 \(\ge\) B2: explicit progress is useful compressed planning state |
| **B** `history_sufficient` | B2 \(\approx\) B1 > B0: memory helps; semantic task graph not extra |
| **C** `state_sufficient` | B0 \(\approx\) B1: \(s,g\) already determine \(A\); **STOP** |
| **D** `progress_hurts` | B1 < B0: bad inductive bias; accept |

## X2 (locked)

\(\hat p_t=G_\psi(o_{\le t},g)\): first \(s_{\le t}\to p_t\), then RGB.
Test how much of the oracle-progress gain survives. Full stack later:

observation \(\to (s_t,\theta_t,\hat p_t)\to q_\phi(A\mid s,g,\hat p)
\to\) candidates \(\to\) APR-WM rollout \(\to\) execute.

## Claims ceiling

**X0** may say: demos cover phases where G0 PASSed; \(s\)-aliasing
exists where G1 PASSed; NLL drop with \(p\) **or its absence** under
the frozen kNN probe; \(p\) is causal (no future). Family FAIL is an
**instrument / representation** result, **not** “task progress has no
planning value.” Keep two classes: cabinet `coverage_hole` =
instrument insufficient; cup/stamp `p_no_nll` = this \(p_t\) did not
further compress action uncertainty. **Not** planning success.

**X1** may say: oracle progress changes task success and search
efficiency vs B0/B2/B3. **Not** learned \(\hat p\); not physics
capacity; not PLAN-X2 on the arm host. **X1 stays LOCKED.**

**XL** (design only until unlock): learned \(z_g,z^p\) vs no-progress;
primary test \(H(A\mid s,z_g,z^p)<H(A\mid s,z_g)\). Not diffusion.

## Ledger

```text
CAP-X3            = PASS
PLAN-X            = CLOSED at iso_sufficient
PLAN-X2           = LOCKED

RoboTwin-X0       = RAN; capacity claim WITHHELD
RoboTwin-X0R      = NEXT (highest IG)

TASK-X0
├── cup      = FAIL  p_no_nll
├── cabinet  = FAIL  coverage_hole
└── stamp    = FAIL  p_no_nll

TASK-XL           = FROZEN NEXT-after-X0R (learned z_g, z^p)
TASK-XL0          = LOCKED
TASK-X1           = LOCKED (diffusion; X0 did not PASS)
TASK-X2           = LOCKED
RoboTwin-X1       = LOCKED
R10               = LOCKED
```

Priority (freeze): RoboTwin-X0R dynamics instrument \(>\) TASK-X retry
\(>\) visual \(>\) diffusion. TASK-X retry **=** learned \(z_g,z^p\),
**not** hand phases. PLAN-X showed state+goal can learn a search
center; TASK-X0 did **not** show a hand phase table further improves
the action distribution. Cup/stamp G2 fail on frozen kNN does **not**
prove learned \(z^p\) is worthless. Remaining closed: diffusion,
TASK-X1. Do not rewrite “progress unproven” as “need a stronger
generator.” If TASK-X is retried: first fix coverage and object-state
logging when using RoboTwin demos; the test is learned progress vs
no-progress, not rewrite phases.
