# RoboTwin-X0 Preregistration — Oracle-State Physics vs Latent Capacity

Date: 2026-08-23  
Status: **FROZEN**; formal **RAN** 2026-08-27; **capacity claim WITHHELD**
(honest reading: identity-physics / weak excitation). **X0R = RAN**
(`excitation_failure`; G1 PASS). Next **RTWX-X0S**. TASK-XL DESIGN
FROZEN / LOCKED. Do not open X1. Do not unlock R10.  
Depends on: CAP-X3 PASS (`structured_adaptation_advantage`); PLAN-X
closed on `capx_arm3`; R10 locked  
Does not: RGB / RGB-D; diffusion; ViT / Transformer WM; Efficient-WAM
comparison; CAP-X3B; CAP-X4 visual; unlock R10; demonstration datasets

## One question

\[
\boxed{
\textbf{RoboTwin-X0:
Oracle-state Physics vs Latent Capacity Benchmark}
}
\]

\[
\boxed{
\text{On oracle proprioceptive state, does explicit physics + small
residual reduce the representation capacity needed for the same
one-step and rollout prediction quality?}
}
\]

This **transports CAP-X1/X3**, not perception:

- CAP-X1: matched family, capacity \(R_P\).
- CAP-X3: same \(d\), causal \(\theta\) vs latent \(z\) (few-shot).
- RoboTwin-X0: **manipulation-like** tasks, oracle \(s\), capacity curve
  **and** composition split on physical parameters.

## Forbidden in this cell

RGB, cameras as inputs, diffusion, large models, training perception,
policy learning, planner success as the primary metric.

## Tasks (formal; P0 uses a 1-DoF drawer proxy)

| id | intent | RoboTwin file (when assets exist) |
|---|---|---|
| **A** Pick-and-place (rigid, low contact) | `place_empty_cup` or `place_object_stand` |
| **B** Articulated drawer / cabinet | `put_object_cabinet` (no `open_drawer` in this checkout) |
| **C** Contact / insert-like | `stamp_seal` or `place_object_scale` (peg-insert analogue) |

Official task smoke is **PASS** (`rtwx_x0_smoke_passed=true`;
`unlocks_rtwx_x0_formal_runner=true`). Seed 0 hit a hollow object dir;
smoke used `setup_seed=2` / `047_mouse` without retuning gates. Formal
3-task sweep is **unlocked to write and run**. Do not unlock R10.

## State / action (oracle)

\[
s=(q_{\mathrm{robot}},\dot q_{\mathrm{robot}},x_o,R_o,q_{\mathrm{art}},\dot q_{\mathrm{art}})
\quad\text{(no RGB)}
\]

Action: native RoboTwin `qpos` (or applied force on the P0 drawer proxy).
P0 freezes force on the 1-DoF slider so \(F_{\mathrm{phy}}\) is
identifiable (position-PD arms would make “physics” mostly a controller).

## Data (formal)

Per task: 1000 / 200 / 300 episodes; 100–200 steps; **parameter split**
not random-trajectory split. Example:

- train: \(m\in[0.8,1.2]\), \(\mu\in[0.2,0.4]\)
- test: \(m=1.5\), \(\mu=0.5\) (composition)

Random excitation rollouts, **not** human/policy demos.

## Models

1. Pure latent: \(s\to z\to F(z,a)\to s'\), \(d_z\in\{16,32,64,128\}\).
2. Physics-only: \(\hat s'=F_{\mathrm{phy}}(s,a;\phi)\), learn \(\phi\).
3. Hybrid: \(F_{\mathrm{phy}}+R_\psi\), \(d_r\in\{0,8,16,32\}\).

Primary: **capacity curve** \(E_{\mathrm{roll}}(\log C)\) at matched
\(E_1\). \(\rho_r=\dim(z_r)/\dim(z_{\mathrm{latent}}^{\star})\).

Metrics: M1 one-step \(E_1\); M2 rollout \(H\in\{10,50\}\); M3 task
coordinate (e.g. \(q_{\mathrm{drawer}}\)).

## Patterns

| pattern | meaning |
|---|---|
| `physics_capacity_shift` | hybrid/physics curve left of latent at matched \(E_{\mathrm{roll}}\) |
| `latent_sufficient` | same \(C\), latent matches or beats physics |
| `contact_breaks_physics` | A/B hold, C fails |
| `instrument_failure` | P0-style accounting fails; no capacity claim |

## Ledger

```text
CAP-X3            = PASS
PLAN-X            = CLOSED at iso_sufficient

RoboTwin-X0-P0    = PASS
RoboTwin-X0-smoke = PASS
RoboTwin-X0       = RAN
                    metrics finite
                    capacity claim WITHHELD
                    reason:
                    weak excitation /
                    identity-physics confound /
                    phi not identified /
                    latent rollout instability

RoboTwin-X0R      = FAIL / excitation_failure; G1 PASS; G0/G2/G3 FAIL
                    bottleneck = F_phy structure (Δs≠0; audit OK)
RTWX-X0S          = RAN / FAIL timing_mismatch
RTWX-X0F          = RAN / FAIL force_channel_unresolved
TASK-X0
├── cup      = FAIL  p_no_nll
├── cabinet  = FAIL  coverage_hole
└── stamp    = FAIL  p_no_nll
TASK-X1           = LOCKED
TASK-XL           = DESIGN FROZEN ONLY; LOCKED
PLAN-X2           = LOCKED
RoboTwin-X1       = LOCKED
R10               = LOCKED
```

Do not relabel `runs/rtwx_x0/formal/summary.json` after seeing numbers.
Interpretation: `REPORT/REP/RTWX/RTWX0_REPORT.md`.
