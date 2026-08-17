# R3-V7B.2 Report — Fast/Slow Epistemic State Separation

Date: 2026-08-16  
Prereg: `REPORT/REG/R3_V7B2_PREREG.md`  
Depends on: `REPORT/REP/R3_V7B1_REPORT.md`  
Artifacts: `runs/r3_v7b2/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `09549e488400cd99958885c959dc485cdd3183afb479247e1a40ee9708b72f23`

## Decision

\[
\boxed{V7B.2\_GO=\mathrm{false}}
\]

\[
\boxed{
\text{fast/slow + persistence gating keeps (even improves) accumulation
vs B2, but does not reduce C0 false-belief occupancy,
and loses the occupancy comparison to a capacity-matched single GRU}
}
\]

H1 and H3 pass. H2 and H4 fail. This does **not** rewrite V7A/B/B.1 GOs.
It does **not** open V7C.

Smoke (`runs/r3_v7b2/smoke/`) is plumbing only.

## Question (unchanged)

Does explicit separation of transient surprise and persistent
inadequacy retain accumulation while lowering C0 occupancy?
Same BCE as V7B B2. No phase in the model. \(p_t=H(u^{slow})\) only.

Frozen: \(\delta=0.02\);
\(\tau_{\mathrm{dev}}=0.489\) (B2 dev C0 \(p_T\) 95th);
\(\eta=0.120\) (median B2 dev C0 `switch_cycle` max event rise).
Params: B2 \(4449\), B2-wide \(6521\), B4 \(6626\).

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 non-inferior + one strict vs B2 | **PASS** | mid \(0.110<0.123\); final \(0.106<0.121\) |
| H2 \(B_{\mathrm{C0}}^{B4}<B_{\mathrm{C0}}^{B2}\) and FPR \(\le0.20\) | **FAIL** | \(0.204>0.178\); FPR \(0.595\) |
| H3 held-out C0 switch median \(\Delta p^{B4}<\eta\) | **PASS** | \(0.086<0.120\) |
| H4 \(B_{\mathrm{C0}}^{B4}<B_{\mathrm{C0}}^{wide}\) | **FAIL** | \(0.204>0.191\) |

\[
\boxed{V7B.2\_GO=H1\land H2\land H3\land H4=\mathrm{false}}
\]

## What the numbers say

B4 is a **better accumulator** than B2 on held-out mid/final Brier
(and slightly better IBS than B2: \(0.122<0.127\); B2-wide IBS \(0.120\)
is similar). Persistence gating did not destroy V7B’s memory benefit.

It did **not** solve the actual V7B failure. C0 occupancy got worse:
\(B_{\mathrm{C0}}\) and \(M_{\mathrm{C0}}\) both rise vs B2; FPR vs the
frozen B2 \(\tau\) is \(0.595\); mean FP dwell is \(44\) steps (far
above V7B.1’s \(\sim 5\)). H4 says extra **structure** is not better
than extra **GRU width** for C0 Brier mass — the wide single GRU is
lower.

H3 passing with this \(\eta\) means slow \(p\) does not **jump** at
contact/release/recontact more than B2 did on development. The C0
problem remains an **elevated slow baseline**, not an event spike.
Event-aligned held-out C0 `switch_cycle` mean \(p_{\mathrm{slow}}\):

| event | \(p_{\mathrm{slow}}\) | \(\|u^{fast}\|\) |
|---|---|---|
| first contact | \(0.259\) | \(1.97\) |
| release | \(0.430\) | \(1.89\) |
| re-contact | \(0.221\) | \(2.31\) |
| quiet | \(0.194\) | \(2.18\) |

Fast state is active across events (diagnostic). Slow \(p\) is still
highest at release and \(\approx0.19\) in quiet — the same qualitative
shape as V7B.1’s single GRU, not a slow belief that stays down while
fast absorbs transients.

## Scientific reading

The representation hypothesis

\[
\text{epistemic state needs explicit fast/slow dynamics}
\]

is **not supported** as a fix for C0 occupancy on this Door mixture,
under matched BCE and a capacity control. Time-scale separation as
implemented can help **discrimination/accumulation** (H1) without
making slow belief **specific to persistent inadequacy** (H2, H4).

That is consistent with V7B.1: the defect is not “forget faster” or
“score IBS instead,” and not merely “one GRU is too small.”

## What this does not authorize

- V7C / V7D;
- occupancy penalties as the next default;
- treating H1 as a reason to enlarge Transformers;
- feeding phase labels into \(u^{fast}\).

If continuing: a new prereg is required. **R3-V7B.3** (evidence-warranted
supervision) was run (`V7B.3_GO=true`). V7C remains locked until a
sensor-stage prereg.

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
V7A\_GO &= true\\
V7B\_GO &= false \quad\text{(memory helps; transients)}\\
V7B.1\_GO &= false \quad\text{(IBS insufficient)}\\
V7B.2\_GO &= false \quad\text{(fast/slow occupancy not solved)}\\
V7C/D &= locked
\end{aligned}
}
\]
