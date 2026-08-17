# R3-V7B.1 Report — Transient-Robust Temporal Epistemic Calibration

Date: 2026-08-16  
Prereg: `REPORT/REG/R3/R3_V7B1_PREREG.md`  
Depends on: `REPORT/REP/R3/R3_V7B_REPORT.md`  
Artifacts: `runs/r3_v7b1/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `1b821f9095eaeb90dcdf1ab3c2a83206340fc44c961a7ca1197416983423bc4c`

## Decision

\[
\boxed{V7B.1\_GO=\mathrm{false}}
\]

\[
\boxed{
\text{same GRU + integrated Brier does not separate
transient surprise from persistent inadequacy}
}
\]

H3 (non-inferior accumulation) passes. H1 and H2 fail.
This does **not** rewrite `V7B_GO` or `V7A_GO`. It does **not** open
V7C/D, RS5B, or a forgetting penalty as an implicit fix.

Smoke (`runs/r3_v7b1/smoke/`) is plumbing only.

## Question (unchanged)

Can the V7B recurrent representation keep its calibration benefit if
trained with episode-balanced IBS instead of time-weighted BCE?

Architecture of B2 and B3 is identical. Only the objective changes.
\(\tau_{\mathrm{dev}}=0.515\) is the 95th percentile of **B2**
development C0 \(p_T\), frozen once. \(\delta=0.02\).

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 \(IBS_{B3}<IBS_{B2}\) | **FAIL** | \(0.143>0.134\) |
| H2 \(B_{\mathrm{C0}}^{B3}<B_{\mathrm{C0}}^{B2}\) and FPR \(\le0.20\) | **FAIL** | \(B_{\mathrm{C0}}\) \(0.192<0.200\); FPR \(0.341>0.20\) |
| H3 mid/final Brier \(\le\) B2 \(+\delta\) | **PASS** | mid \(0.139\le0.133+0.02\); final \(0.132\le0.122+0.02\) |

\[
\boxed{V7B.1\_GO=H1\land H2\land H3=\mathrm{false}}
\]

## What the numbers say

IBS training did **not** improve overall temporal calibration (H1).
It slightly reduced episode-balanced C0 Brier mass
\(B_{\mathrm{C0}}:0.200\to0.192\) and false-belief mass
\(M_{\mathrm{C0}}:0.438\to0.432\), but occupancy vs the frozen B2
threshold got worse (\(0.341\)). So the proper scoring rule did not
deliver the safety line, and the occupancy drop is too small to count
as solving transients.

H3 shows B3 did **not** collapse to a never-suspect policy: midpoint
and final Brier stay within \(\delta=0.02\) of B2. The accumulation
benefit is retained; it is simply not cleaned up.

\(\mathrm{Brier}(t/T)\) is almost flat for both models (B2
\(0.133/0.129/0.133/0.135/0.122\); B3
\(0.130/0.143/0.139/0.141/0.132\)). There is no clean
early-to-late accumulation curve on this mixture.

Event-aligned C0 `switch_cycle` mean \(p\) (B3, evaluator phase only):

| event | mean \(p\) |
|---|---|
| first contact | \(0.393\) |
| release | \(0.416\) |
| re-contact | \(0.299\) |
| quiet | \(0.290\) |

This looks more like **elevated baseline during interaction** (highest
at release, still \(\approx0.29\) in quiet) than a single contact-onset
spike that then forgets. Median recovery remains \(0\); mean FP dwell
\(4.6\) steps.

## Scientific reading

The V7B.1 hypothesis

\[
\text{representation works, objective is misaligned}
\]

is **not supported** as a sufficient fix. A fair test of IBS vs BCE on
the same GRU leaves transient C0 occupancy unsolved and slightly
worsens integrated Brier.

Persistent-belief mainline is still not refuted (V7B H1/H2 stand; V7B.1
H3 shows IBS does not erase them). The remaining gap is representational
time-scale, not “add \(\lambda\sum p_t\) on C0” and not sensors.

## What this does not authorize

- opening V7C/D;
- installing a C0 occupancy penalty as the next default;
- enlarging to Transformer because IBS failed;
- rewriting V7B H1/H2.

If continuing: a **new** prereg for explicit fast/slow \(p=H(u^{slow})\)
was run as R3-V7B.2 (`V7B.2_GO=false`). Not V7C.

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
V7A\_GO &= true\\
V7B\_GO &= false \quad\text{(memory helps; transients not robust)}\\
V7B.1\_GO &= false \quad\text{(IBS \(\neq\) enough)}\\
V7C/D,\ RS5B &= locked
\end{aligned}
}
\]
