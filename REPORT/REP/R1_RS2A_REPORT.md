# R1-RS2A Report — Contact-Mediated Structural Identifiability

Date: 2026-08-16  
Prereg: `REPORT/REG/R1_RS2A_PREREG.md`  
Depends on: `REPORT/REP/R1_RS2_REPORT.md`  
Artifacts: `runs/r1_rs2a/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `1d040b6b1fc12da729e26c3d5b5cfdf7489069fa1f33db3d17f137a9b9dd0ac8`

## Decision

\[
\boxed{RS2A\_GO=\mathrm{true}}
\]

\[
\boxed{
\text{inadequacy detectability is better predicted by tangent-visible
exposure than by raw operator exposure}
}
\]

This does **not** set `RS2_GO=true`. It does **not** rewrite `RS1C_GO`.
It does **not** unlock threshold, \(s^\star\), VoI, or policy edits.
**RS2B was subsequently opened as its own diagnosis** (`REPORT/REP/R1_RS2B_REPORT.md`); this report does not contain it.

Smoke (`runs/r1_rs2a/smoke/`) is plumbing only. The 90-episode matrix is
the first run allowed to set `RS2A_GO`.

## Question (unchanged)

\[
\boxed{
\text{Is contact transfer governed by tangent-visible exposure rather
than raw operator exposure?}
}
\]

C2 / consequence transport is excluded. No decision bits were computed.

## Matrix

90 episodes: Mode-A \(5\times3\times2=30\) plus contact
\(5\times3\times4=60\). Regimes \(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\).
Primary adapter pair: contact `pull_release` @ \(s^\star=2.0\) vs Mode-A
\(1.5A_0\), C1 only.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 raw \(X_\phi\) on adapter pair | **PASS** | rel. error \(17.06\%<20\%\) |
| H2 \(\kappa_\perp^{contact}<\kappa_\perp^{ModeA}\) | **PASS** | \(0.0251<0.0430\), ratio \(0.584\) |
| H3a Spearman \(D_0\) vs \(\lvert\alpha\rvert\sqrt{X_{\phi,\perp}}\) | **PASS** | \(0.343>0.288\) |
| H3b NRMSE of that predictor | **PASS** | \(0.713<1.815\) |

\[
\boxed{RS2A\_GO=H2\land H3a\land H3b=\mathrm{true}}
\]

## What the numbers actually say

On the **adapter pair** (the only pair that was designed to match
exposure):

| Domain | median \(X_\phi\) | median \(\kappa_\perp\) | median \(D_0\) (C1) |
|--------|------------------:|------------------------:|--------------------:|
| Mode-A \(1.5A_0\) | \(7.93\times10^{-4}\) | \(0.0430\) | \(2.27\times10^{-3}\) |
| contact `pull_release` \(s^\star\) | \(9.28\times10^{-4}\) | \(0.0251\) | \(1.10\times10^{-3}\) |

H1 restates the adapter: raw \(X_\phi\) transports. H2 says the fraction
of \(\Phi=|v|v\) that survives outside \(\mathrm{col}(J_\theta)\) is
lower under contact. H3 says replacing \(X_\phi\) by \(X_{\phi,\perp}\)
improves the cross-domain \(D_0\) predictor, especially in NRMSE.

That is **not** a tight identity \(D_0=|\alpha|\sqrt{X_{\phi,\perp}}\)
(NRMSE \(0.71\)). It is a directional geometry result: detectability
tracks excitation orthogonal to the incumbent parameter tangent more
than it tracks raw \(\mathbb E[v^4]\).

Both domains already have **small** \(\kappa_\perp\) (\(\approx0.04\)
on Mode-A), because the frozen tangent includes \(\dot q\), and
\(|v|v\) is strongly correlated with \(v\) on these windows. Contact
does not create a new kind of invisibility from nothing; it further
collapses the already-thin orthogonal component, most on the long
one-sided \(s^\star\) stroke.

## Formal bin assignment was not exposure matching

Reported, not gated:

| Formal pair | \(X_\phi\) rel. error | \(\kappa_\perp\) contact / Mode-A |
|-------------|----------------------:|----------------------------------:|
| `slow_pull` \(\leftrightarrow 1.0A_0\) | \(5.87\) | \(0.099 / 0.042\) |
| `fast_pull` \(\leftrightarrow 1.5A_0\) | \(0.844\) | \(0.048 / 0.043\) |

Nearest-bin mapping sent Formal `fast_pull` to the \(1.5A_0\) threshold
while leaving \(X_\phi\) about \(6\times\) smaller. That is a second,
independent reason Formal never crossed `detect` on those scripts. RS2A
does not retune the bins.

## Secondary Gram (not a detector)

| | Mode-A C1 | contact C1 |
|--|----------:|-----------:|
| median \(\mathrm{corr}(\lvert v\rvert v,\ v^2)\) | \(-0.041\) | \(\mathbf{0.995}\) |

Mode-A sine is bidirectional, so `abs_v_v` and `signed_v2` are nearly
uncorrelated. Contact pulls are one-sided, so the two operators are
almost collinear. Contact changes **identifiability geometry**, not
only amplitude. This remains a diagnostic, not a new \(D_0\).

## Interpretation

\[
\boxed{
X_\phi^{contact}\approx X_\phi^{ModeA}
\not\Rightarrow
D_0^{contact}\approx D_0^{ModeA}
}
\]

on the adapter pair, and the gap shrinks when exposure is taken
orthogonal to \(\mathrm{col}(J_\theta)\). Combined with RS2 Formal:

\[
\boxed{
\begin{aligned}
\text{contact physics interface} &\quad\text{transfers}\\
\text{epistemic allocation calibration} &\quad\text{does not}\\
\text{operator exposure matching} &\not\Rightarrow\text{structural detectability matching}
\end{aligned}
}
\]

The C1 miss is a **transportability** failure of the detectability
statistic under a different intervention mechanism
(\(\tau_{\mathrm{hinge}}\) vs \(u_{\mathrm{robot}}\to J^\top f\)),
recoverable in part by V3’s parameter-tangent split.

C2 consequence non-transfer is a different problem, opened later as RS2B.

## Status

\[
\boxed{
RS1C\_GO=\mathrm{true},\quad
RS2\text{-C0\_PASS}=\mathrm{true},\quad
RS2\_GO=\mathrm{false},\quad
RS2A\_GO=\mathrm{true},\quad
RS2B\text{ subsequently diagnosed (see R1\_RS2B\_REPORT)}
}
\]
