# R1-RS2 Report — Contact-Mediated Epistemic Revision

Date: 2026-08-16  
Prereg: `REPORT/REG/R1_RS2_PREREG.md`  
C0: `REPORT/REP/R1_RS2_C0_REPORT.md`, `runs/r1_rs2/c0_v2/`  
Artifacts: `runs/r1_rs2/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `aab3e8ba4121b080c70886e1ecc7714bcc8493d40d2a9c7d65aac4b56f5a54a8`

## Decision

\[
\boxed{
RS2\text{-C0\_PASS}=\mathrm{true},\qquad
RS2\_GO=\mathrm{false}
}
\]

\[
\boxed{
\text{Frozen RS1C does not transfer as an epistemic-revision policy
under robot-contact excitation.}
}
\]

This does **not** rewrite `RS1C_GO` (`true` remains on direct hinge
excitation). It does **not** rewrite `RS1B_GO` (`false`). It does **not**
authorize detector, VoI, adapter, threshold, script, or support retuning.

Smoke (`runs/r1_rs2/formal_smoke/`) remains plumbing only. This 180-episode
held-out matrix is the first run allowed to set `RS2_GO`.

## Scientific question (unchanged)

\[
\boxed{
\text{Does the frozen RS1C policy remain valid when Door dynamics are
excited through robot contact rather than direct hinge torque?}
}
\]

C0 already closed the interface claim

\[
\text{normal contact}\not\Rightarrow\text{structural residual},
\]

given explicit \(J_c^\top f_c\) accounting. Formal therefore tests transfer
of four frozen mechanisms together: epistemic partition, VoI probing,
physical admissibility among installs, and long-horizon utility of
\(\pi_{\mathrm{RS1C}}\).

## Matrix

Held-out seeds \(\{11101,11111,11121,11131,11141\}\) (disjoint from C0/RS1).

\[
5\text{ seeds}
\times
\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H},\mathrm{C2}\}
\times
3\text{ scripts}
\times
3\text{ repeats}
=
\boxed{180\text{ episodes}}.
\]

Frozen and untouched:

- C0 manifest SHA256
  `3dee8f6160e041b1a5a1499d3e011c5e62a39f378b9a21039ab6997f7e5ab89e`;
- \(s^\star=2.0\) (exposure adapter, not “use 2× OSC force”);
- script \(\to\) amp bins: `slow_pull→1.0A0`, `fast_pull→1.5A0`,
  `pull_release→1.0A0`;
- RS1A.5 intake: \(C_{\mathrm{tol}}=0.001934\),
  \(A^\star=1.5A_0\), \(\lambda=0.0015\);
- cell thresholds: \(0.5A_0\!:1.002\), \(1.0A_0\!:1.002\),
  \(1.5A_0\!:0.002231\), \(2.0A_0\!:0.003364\);
- support role: monitoring only, never a veto.

Contact geometry is the frozen C0 pose. Seed/repeat therefore vary Mode-A
consequence \(C\) and the revision candidate, not the scripted contact
trajectory. That is a property of the freeze, not a post-hoc choice.

## Gates

| Gate | Result | Detail |
|---|---|---|
| C0 specificity | **PASS** | 0 / 45 C0 installs |
| tolerate false install | **PASS** | 0 / 108 |
| passivity among installs | **PASS** | 0 (vacuous: 0 installs) |
| seed-mean H32 vs none | **FAIL** | \(\Delta=0\) (no \(\pi_{\mathrm{RS1C}}\) install) |
| gain vs veto | **PASS** | \(\Delta=0\) |
| VoI path | **FAIL** | 72 initial probes; **0** promotions; 0/5 seeds |
| monitor consistency | **PASS** | 0 support accept-flips |

\[
\boxed{RS2\_GO=\mathrm{false}}
\]

fails on **VoI path** and **utility vs none**. Per prereg, VoI failure means
the epistemic-control path was not exercised under contact. Utility failure
here is a consequence of allocation: \(\pi_{\mathrm{RS1C}}\) never installed,
so its seed-mean gain is identically zero. It is **not** a C0
interface/accounting failure.

## Allocation (the actual transfer failure)

| Stage | n |
|-------|--:|
| Initial tolerate / probe / revise-worthy | **108 / 72 / 0** |
| VoI extra \(s^\star=2.0\) `pull_release` acquired | 48 |
| Promoted probe \(\to\) revise-worthy | **0** |
| Final tolerate / probe / revise-worthy | 108 / 72 / 0 |
| \(\pi_{\mathrm{RS1C}}\) installs | **0** |

By regime (initial bucket):

| Regime | tolerate | probe | revise-worthy |
|--------|--------:|------:|--------------:|
| C0 | 45 | 0 | 0 |
| C1-L | 18 | 27 | 0 |
| C1-H | 0 | 45 | 0 |
| C2 | 45 | 0 | 0 |

C1-L tolerate (18) is two seeds whose Mode-A \(C<C_{\mathrm{tol}}\)
(11101: \(0.001894\); 11121: \(0.001470\)). The other three C1-L seeds and
all C1-H are consequential but undetected.

C2 is **detected** on every cell (\(D_0\in[3.86,7.06]\)) but **not
consequential** on the frozen Mode-A \(C\) (\(C\le 0.001085<C_{\mathrm{tol}}\)),
so RS1C labels it tolerate. Latch novelty is visible in the contact residual
and is not canceled by \(J_c^\top f_c\); allocation still refuses install.

## Why detect / VoI never fire

Contact-mediated \(D_0\) (phase-2 \(r_\perp\) RMS after the frozen tangent
posterior) stays below every active detect threshold:

| Regime | script (amp bin) | \(D_0\) | threshold |
|--------|------------------|---------:|----------:|
| C1-H | `fast_pull` (\(1.5A_0\)) | \(7.62\times10^{-4}\) | \(2.23\times10^{-3}\) |
| C1-H | `slow_pull` (\(1.0A_0\)) | \(5.35\times10^{-4}\) | \(1.002\) |
| C1-L | `fast_pull` (\(1.5A_0\)) | \(4.87\times10^{-4}\) | \(2.23\times10^{-3}\) |
| VoI extra | `pull_release` @ \(s^\star\) vs \(1.5A_0\) | max \(5.35\times10^{-4}\) | \(2.23\times10^{-3}\) |

C0 \(D_0\) is the same order (\(1.8\times10^{-4}\) to \(5.8\times10^{-4}\)).
Normal contact is not leaking into a large structural residual; the hidden
drag simply does not reach the Mode-A cell thresholds under this frozen
contact excitation.

The \(1.0A_0\) threshold \(\approx 1.002\) is the RS1A.5 noise-floor cell.
Mapping `slow_pull` / `pull_release` onto that bin makes detect essentially
impossible there. Even the \(1.5A_0\) cell — the one RS1A.5 actually uses
for VoI — is not crossed by contact \(D_0\) or by the \(s^\star\) extra
probe. That is the transfer break: **operator exposure was matched in
\(X_\phi\), but residual detectability was not.**

## Physical admissibility and always-baseline (not GO)

Among \(\pi_{\mathrm{RS1C}}\) installs, passivity violations are 0 because
there are no installs.

\(\pi_{\mathrm{always}}\) (candidate every episode; frozen passivity + short
utility still apply; **not a GO gate**) accepted **84 / 180**:

| Regime | always accepts | passivity flags (any candidate) |
|--------|---------------:|-------------------------------:|
| C0 | 3 | 30 |
| C1-L | 36 | 9 |
| C1-H | 45 | 0 |
| C2 | 0 | 27 |

The three C0 always-accepts are seed 11111 `fast_pull` repeats; they do
**not** count as \(\pi_{\mathrm{RS1C}}\) C0 installs.

Seed-mean H32 gain of \(\pi_{\mathrm{always}}\): \(+0.00327\)
(95% CI \([0.00016, 0.00639]\)). Among the 84 always-accepted episodes,
mean query-level gain is \(+0.00702\) (min \(-0.019\), max \(+0.033\));
every one has \(I_{\mathrm{exit}}>0\). Cost-sensitivity \(J(\lambda_R)\)
stays positive on the preregistered grid (not a gate).

So a physically admissible drag candidate **can** improve contact-mediated
H32 if allocation is bypassed. Formal failure is not “revision is useless
under contact.” It is that the frozen detect/VoI layer never promotes
contact evidence to revise-worthy.

## Monitoring

No \(\pi_{\mathrm{RS1C}}\) install, so `monitor_intensity` is `none`
everywhere under that policy. Support never flipped `accepted`.
\(I_{\mathrm{exit}}>0\) on always-accepts is reported only; it does not
veto and does not change `RS1B_GO`.

## Interpretation (frozen; no retune)

\[
\boxed{
\begin{aligned}
\text{interface} &: \text{held (C0 specificity, no contact}\Rightarrow\text{false residual)}\\
\text{allocation} &: \text{did not transfer (0 revise-worthy, 0 VoI promotions)}\\
\text{admissibility / utility} &: \text{not tested for }\pi_{\mathrm{RS1C}}\text{; always H32}>0
\end{aligned}
}
\]

The causal boundary from C0 remains:

\[
\text{normal contact}\not\Rightarrow\text{structural residual}.
\]

The new scientific boundary is:

\[
\boxed{
\text{Mode-A exposure matching}
\not\Rightarrow
\text{Mode-A detectability under contact}
}
\]

on this frozen OSC adapter. Embodied epistemic interaction is therefore
**not** yet shown: the robot can excite the Door, the force interface is
closed, but the frozen RS1C partition does not spend VoI or install.

Authorized next work was diagnosis, not a silent threshold move:
`R1-RS2A` and `R1-RS2B`. The scientific close is
`REPORT/REP/R1_RS2_STAGE_FREEZE.md`. Do not rerun RS2.

## Status

\[
\boxed{
RS1C\_GO=\mathrm{true},\quad
RS2\text{-C0\_PASS}=\mathrm{true},\quad
RS2\_GO=\mathrm{false},\quad
RS2A\_GO=\mathrm{true},\quad
RS2B\_GO=\mathrm{true}
}
\]
