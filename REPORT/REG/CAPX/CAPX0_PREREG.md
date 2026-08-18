# CAP-X0 Preregistration — Benchmark / Accounting / Baseline Fairness

Date: 2026-08-17  
Status: **FROZEN**; **PASS** (`cap_x0_passed=true`);
**CAP-X1 prereg frozen** (`REPORT/REG/CAPX/CAPX1_PREREG.md`)  
Depends on: `REPORT/REG/CAPX/CAPX_PREREG.md`  
Does not: capacity replacement ratios; planning MPC; \(\rho\) sweep;
scene adaptation \(K\); visual front-end; unlock R10; retune after
seeing PureNN curves beyond G3 floor

## One question

\[
\boxed{
\text{Is the CAP-X host, scene generator, excitation, split, and
oracle physics accounting fair and closed enough to support later
capacity claims?}
}
\]

X0 **must not** claim \(R_P\) / \(R_F\).

## Plant

\[
\boxed{\texttt{capx\_arm3.v1}}
\]

| item | value |
|---|---|
| DoF | 3 revolute, serial |
| drive | `qfrc_applied` only (\(\nu=0\) actuators) |
| contact | all geoms `contype=conaffinity=0` |
| gravity | \(9.81\) |
| timestep | \(0.002\,\mathrm{s}\) (500 Hz physics) |
| learner rate | downsample to \(100\,\mathrm{Hz}\) for neural train |

### Scene parameters \(\theta_e\) (frozen dim)

\[
\begin{aligned}
\theta_e &=
\bigl(
s_m^{(1:3)},\,
s_I^{(1:3)},\,
b^{(1:3)},\,
\mu^{(1:3)},\,
m_p
\bigr)
\in\mathbb{R}^{13},
\\
s_m^{(i)} &\sim\mathrm{Unif}[0.7,1.3],\quad
s_I^{(i)}\sim\mathrm{Unif}[0.7,1.3],
\\
b^{(i)} &\sim\mathrm{Unif}[0.5\,b_i^0,\,1.5\,b_i^0],\quad
\mu^{(i)}=0\quad(\textbf{X0 freeze}; Coulomb convention deferred),
\\
m_p &\sim\mathrm{Unif}[0,1]\,\mathrm{kg}.
\end{aligned}
\]

Nominal \(b_i^0=0.05\). Payload is a tip body mass change on a fixed
kinematic tip frame. The \(\mu\) slots stay in the vector for forward
compatibility; X0 does not claim Coulomb identification.

## Dataset (frozen)

| split | \(n_{\mathrm{scene}}\) | scene-id seeds |
|---|---:|---|
| train | 128 | \(70000+i\) |
| val | 32 | \(80000+i\) |
| test | 64 | \(90000+i\) |

Per scene: **8** trajectories × **2.0 s** (X0 accounting scale;
X1 may raise to 32×5 s under a new prereg — not silently).

Excitation kinds (no discontinuous torque impulses):

\[
\{\mathrm{multi\text{-}sine},\,\mathrm{chirp},\,\mathrm{bandlimited},\,\mathrm{piecewise\text{-}smooth}\}.
\]

Constraint: \(\theta^{\mathrm{train}}\cap\theta^{\mathrm{val}}\cap\theta^{\mathrm{test}}=\varnothing\)
(exact vector equality forbidden; generator uses disjoint seed spaces).

## Models in X0

| model | role |
|---|---|
| **Oracle Physics** | rigid-body residual with true \(\theta_e\); G0 only |
| **PureNN \(H=256\)** | MLP \([q,\dot q,u,\theta_e]\to\ddot q\); G3 sanity only |

Physics+Residual and capacity sweeps are **forbidden in X0**.

## Gates

| id | requirement |
|---|---|
| **G0** | Oracle \(\theta\): \(\mathrm{NRMSE}(r_{\mathrm{phy}})<10^{-4}\) on train+val pooled samples |
| **G1** | Each DoF: \(\mathrm{Var}(q_i),\mathrm{Var}(\dot q_i),\mathrm{Var}(\ddot q_i)>\varepsilon\) with \(\varepsilon_q=10^{-4}\), \(\varepsilon_v=10^{-3}\), \(\varepsilon_a=10^{-2}\) on train pool |
| **G2** | No duplicate \(\theta\) across splits; no shared trajectory seeds across splits; inventory JSON present |
| **G3** | PureNN \(H=256\): one-step \(\mathrm{NRMSE}(\hat{\ddot q})\) on **val** \(\le 0.25\) after frozen short train |

If G3 fails: **STOP** — do not interpret as “physics wins.”

## Claims ceiling

X0 may say: benchmark closed; oracle accounting holds; large PureNN
is learnable on this family.

X0 may **not** say: physics replaced \(X\%\) capacity; APR-WM is more
sample-efficient; planning improved.

## Unlock

\[
\text{CAP-X0 PASS}\;\Rightarrow\;\text{preregister CAP-X1 (widths, budgets, }R_P\text{ gates)}
\]

R10 remains locked.
