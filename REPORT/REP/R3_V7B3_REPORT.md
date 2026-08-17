# R3-V7B.3 Report — Causal Evidence-Warranted Epistemic Belief

Date: 2026-08-16  
Prereg: `REPORT/REG/R3_V7B3_PREREG.md`  
Depends on: `REPORT/REP/R3_V7B2_REPORT.md`  
Artifacts: `runs/r3_v7b3/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `a7e81f2cb6c174b5b5b3ca3364eb0811dcec47009b7f8cf906deee61d9527c63`

## Decision

\[
\boxed{V7B.3\_GO=\mathrm{true}}
\]

\[
\boxed{
\text{supervising when counterfactual structural evidence becomes
available reduces C0 occupancy while tracking }w_t^{\mathrm{evid}},
\text{ without destroying B2 mid/final calibration within }\delta
}
\]

Same GRU as B2. Runtime does **not** see the C0 tape or \(w_t\).
This does **not** open V7C, revision, or occupancy penalties.

Smoke (`runs/r3_v7b3/smoke/`): 3 matched triples; C1 \(w_T\approx 1\),
C0 \(w_T=0\).

## Question (unchanged)

Does evidence-availability supervision beat episode-constant \(y\) for
C0 temporal occupancy, while keeping history-based Brier vs \(y\)?

## Design (frozen)

- Seeds \(\{20101,20111\}\) train, \(20121\) val, \(\{20131,20141\}\) held-out.
- Matched C0 record / C1-L / C1-H replay; \(N=105\).
- \(w_t=A_t/(A_T+\varepsilon)\), \(A_t=\sum_{k\le t}(r_k^{C1}-r_k^{C0})^2\).
- B2: time-weighted BCE vs \(y\). B5: BCE vs \(y\cdot w_t\).
- \(\tau_{\mathrm{dev}}=0.620\) from B2 development C0 \(p_T\).

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 mid/final Brier \(\le\) B2 \(+\delta\) | **PASS** | mid \(0.178\le0.179+0.02\); final \(0.184\le0.181+0.02\) |
| H2 \(B_{\mathrm{C0}}\) down and FPR \(\le0.20\) | **PASS** | \(0.219<0.293\); FPR \(0.006\) |
| H3 Spearman\((p,w)>0\) and \(E[p\mid w_{\mathrm{high}}]>E[p\mid w_{\mathrm{low}}]\) | **PASS** | \(\rho=0.210\); \(0.665>0.340\) |
| H4 early \(|\Delta p|\le0.20\) and late gap grows by \(\gamma\) | **PASS** | early \(0.149\); late \(\Delta=0.219>0.149+0.05\) |

\[
\boxed{V7B.3\_GO=H1\land H2\land H3\land H4=\mathrm{true}}
\]

The predeclared Door-oracle recurrent **stop** (if H2 fails) is **not**
triggered.

## What the numbers say

On this paired mixture, constant-\(y\) B2 itself has a high C0 baseline
(\(M_{\mathrm{C0}}=0.534\), \(B_{\mathrm{C0}}=0.293\)). Warranted targets
cut occupancy (\(M_{\mathrm{C0}}=0.440\), \(B_{\mathrm{C0}}=0.219\)) and
almost eliminate steps above B2’s C0 final 95th (\(0.006\)).

H3/H4 are the mechanism: B5’s \(p_t\) on C1 co-moves with offline
\(w_t\), and the matched C1−C0 gap is smaller before evidence
accumulates than after. That is the intended distinction between
“hidden class is wrong” and “the prefix already warrants high
confidence.”

H1 is **non-inferiority**, not a new Brier record: B5 final vs \(y\) is
slightly worse than B2 (\(0.184\) vs \(0.181\)) but inside \(\delta\).
The win is temporal occupancy and evidence timing, not a louder
end-of-episode detector.

## What this does not authorize

- feeding \(w_t\) or the C0 tape into the deployed model;
- feeding \(w_t\) at runtime (V7C later removes only inference-time
  oracle contact; \(w_t\) stays train-only);
- claiming invariant scalars or LOIO transport;
- occupancy penalties as the method (they were not used).

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
V7A\_GO &= true\\
V7B\_GO &= false\\
V7B.1/B.2\_GO &= false\\
V7B.3\_GO &= true\\
V7C/D &= locked
\end{aligned}
}
\]

**V7C** was subsequently run under its own prereg
(`REPORT/REP/R3_V7C_REPORT.md`, `V7C_GO=true`). **V7D** remains locked.
