# SIM-X3 Preregistration — Persistent Validity Lifecycle

Date: 2026-08-17  
Status: **FROZEN**; **FAIL** (`REPORT/REP/SIMX/SIMX3_REPORT.md`);
pattern **`false_revoke`**; **not an R10 gate**  
Depends on: `REPORT/REP/SIMX/SIMX2_REPORT.md`,
`aprwm_v0/simx_plant.py` (`simx_hinge.v1`)  
Does not: unlock R10-C0; train a validity detector; CUSUM / hazard /
periodic probe; NetVoI / task reward / harmful-commit stats; retune
\(\sigma_{\mathrm{obs}}\) or the \(0.5\) license threshold; let
calibration duration enter formal log-odds

## One question

\[
\boxed{
\text{X2 per-sample }I_a\text{ ordering}
\;\Longrightarrow?\;
\text{persistent }b(\mathcal V)\text{ lifecycle ordering}
}
\]

\[
\boxed{\textbf{X2 = channel}\qquad\textbf{X3 = lifecycle}}
\]

## Analytic oracle-family updater (not a detector)

Persistent hypotheses:

\[
H\in\{0.05,\,0.10,\,0.15\},
\qquad
b_t(\mathcal V)=w_t(0.10)=P(H=0.10\mid y_{1:t},\dot q_{1:t},a).
\]

Do **not** resample the invalid mixture each step. \(b_{\mathrm{true}}\)
is fixed inside a block.

\[
\mu_h(t)=-(h-b_0)\dot q_t,\qquad b_0=0.10,
\]

\[
p(\tilde r_t\mid h,\dot q_t)=\mathcal N(\mu_h(t),\sigma_{\mathrm{obs}}^2),
\qquad
\sigma_{\mathrm{obs}}=0.01\,\mathrm{N\cdot m}
\]

(same synthetic surrogate as X2). Log-space Bayes with `logsumexp`.

## Canonical license (no calibration-duration confound)

Formal lifecycle **always** starts from

\[
\boxed{(w_{0.05},w_{0.10},w_{0.15})=(0.025,\,0.95,\,0.025)}
\]

i.e. \(b_0(\mathcal V)=0.95\). Optional sanity may show `info` can
*reach* this license from a flat prior; that accumulation must **not**
enter formal cells.

## Cell protocol

Host: `simx_hinge.v1`. Actions:

\[
a\in\{\mathrm{info},\mathrm{cons},\mathrm{high\_f}\}
\]

(`low_f` omitted: identical to `info` in X2).

1. **Pre-roll** \(2\,\mathrm{s}\): plant \(b=0.10\), action on; **no**
   belief update; reach X2 steady regime.
2. Inject canonical \(w\).
3. **Shift** plant to \(b_{\mathrm{true}}\in\{0.10,0.05,0.15\}\)
   (stay / down / up).
4. **Lifecycle** \(4\,\mathrm{s}\): update \(w_t\) from
   \(\tilde r_t=r_t+\epsilon_t\).

Phase \(\phi\in\{0,\pi/2,\pi,3\pi/2\}\) in the torque sine.
Observation-noise seeds: **100**. Matrix:

\[
3\times 3\times 4\times 100=\boxed{3600\text{ cells}}.
\]

Plant trajectories may be cached per \((a,\phi,b_{\mathrm{true}})\);
noise only hits the updater.

## Shadow permission (deliberately boring)

\[
C_{\mathrm{candidate}}=1,\qquad
P_t=\mathbf{1}[b_t(\mathcal V)\ge 0.5],
\qquad
C_t^{\mathrm{shadow}}=C_{\mathrm{candidate}}P_t.
\]

Threshold \(0.5\) from R9-A0. No NetVoI, no harmful commit, no task
utility. Metric of interest: **stale shadow permission**.

## Metrics (invalid trajectories)

| symbol | definition |
|---|---|
| \(T_{\mathrm{revoke}}\) | \(\inf\{t:b_t(\mathcal V)<0.5\}\); **censored** if none in \(4\,\mathrm{s}\) (do not impute \(4\)) |
| \(S_{\mathrm{stale}}\) | \(\frac1T\int_0^T\mathbf{1}[b_t\ge 0.5]\,dt\) |
| \(F_{\mathrm{revoke}}\) | stay-valid: \(\mathbf{1}[\exists t:\,b_t<0.5]\) |
| \(b_T(\mathcal V)\) | terminal; also log full \(w_T(h)\) |

**Do not** preregister “cons must stay stale.” X2 near-blind ≠
lifecycle-blind; ~2 bit/s may still revoke. `high_f` (~0.019 bit/s)
is the near-closed channel for G4.

## Gates

| id | pass |
|---|---|
| **G0** | X2 carryover: \(r=-\Delta b\dot q\); \(\sigma_{\mathrm{obs}}=0.01\) unchanged |
| **G1** | stay-valid: \(P(F_{\mathrm{revoke}})\le 0.01\) |
| **G2** | informative revocation: \(P(T_{\mathrm{revoke}}\le 4\,\mathrm{s}\mid a_{\mathrm{info}},\mathrm{invalid})\ge 0.95\) |
| **G3** | lifecycle ordering: \(S_{\mathrm{stale}}^{\mathrm{info}}<S_{\mathrm{stale}}^{\mathrm{cons}}<S_{\mathrm{stale}}^{\mathrm{high\_f}}\) on each invalid shift and aggregate |
| **G4** | near-blind persistence: \(P(T_{\mathrm{revoke}}>4\,\mathrm{s}\mid a_{\mathrm{high\_f}},\mathrm{invalid})\ge 0.90\) |

## Registered patterns (interpret before forcing PASS)

- **A `lifecycle_blindness`:** info fast revoke, cons slower, high_f
  stale → policy-conditioned evidence loss propagates into persistent
  lifecycle (cross-engine).
- **B `accumulation_recovers`:** even high_f revokes → per-step
  near-blindness overcome by time; R9-strong stale license not
  reproduced on this MuJoCo scale.
- **C `false_revoke`:** stay-valid licenses drop → updater/model
  failure; do not celebrate invalid revoke.

## PASS may claim (pattern A)

\[
\boxed{\textbf{cross-engine lifecycle robustness}}
\]

channel difference → persistent belief difference → stale shadow
permission. Still **not** real-physics validation. R10-C0 untouched.
