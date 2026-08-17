# R1-RS1C Preregistration — Layered Epistemic–Physical Policy

Date: 2026-08-16  
Status: **FROZEN; formal completed** — see `REPORT/REP/R1_RS1C_REPORT.md` (`RS1C_GO=true`)  
Depends on: `REPORT/REP/R1_RS1A_STAGE_FREEZE.md`, `REPORT/REP/R1_RS1B_STAGE_FREEZE.md`  
Frozen implementation: `aprwm_v0/r1_rs1c.py`

## Placement

\[
\boxed{
\text{RS1B.2 closes the support-risk branch}
\rightarrow
\text{RS1C = new integrated policy hypothesis}
\rightarrow
\text{RS2 prereg unlocked by RS1C\_GO}
}
\]

RS1C **must not**:

- retune $D_{\mathrm{exit}}$ / add RS1B.3;  
- lower RS1B’s `stable_h32` bar;  
- treat support exit as a hard veto;  
- claim that a passing RS1C retroactively sets `RS1B_GO=true`.

It **must** compose only already-validated pieces:

\[
\boxed{
\begin{aligned}
&\text{tolerate / probe / revise-worthy}\\
+&\text{VoI-optimal probing}\\
+&\text{passivity}\\
+&\text{support as confidence monitoring}
\end{aligned}
}
\]

## Scientific question

\[
\boxed{
\text{Does a layered policy — allocate epistemic effort, then install only
if physically admissible and short-horizon useful, while using support
exit only to raise monitoring — improve long-horizon prediction without
a support veto?}
}
\]

This tests a **policy hypothesis**, not a new detector or a new distance.

## Frozen inputs (do not recalibrate)

| Piece | Source |
|-------|--------|
| $E_{\mathrm{known}}=D_0$, typed unknown/support channel | RS1A freeze |
| $C_{\mathrm{tol}}$, per-amplitude $D_0$ thresholds | passing `runs/r1_rs1a5/formal/summary.json` |
| $A^\star=1.5A_0$, $\lambda=0.0015$, $c(A)=(A/A_0)^2$ | RS1A.5 |
| Dissipative library + passivity + H2/H4/H8 utility | frozen R0.6 / RS1B accept (pre-H32) |
| $d_{\mathcal S}$, $I_{\mathrm{exit}}$ | RS1B.1 formula; **monitoring only** |
| H32 queries | RS1B *intervention* sampler (not RS1B.2 targeted bands) |

## Policy $\pi_{\mathrm{RS1C}}$

### Stage 1 — epistemic allocation (RS1A)

\[
\begin{aligned}
\mathcal T &= \{C<C_{\mathrm{tol}}\} && \text{tolerate: no probe, no revision}\\
\mathcal P &= \{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=0\} && \text{probe candidate}\\
\mathcal R &= \{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=1\} && \text{revise-worthy}
\end{aligned}
\]

On $\mathcal P$, apply frozen VoI. If $\max_a V(a)>0$, acquire evidence at
$A^\star=1.5A_0$ (do **not** use $2A_0$). If the post-probe cell is detect,
promote to $\mathcal R$; else **defer** (still no revision).

### Stage 2 — install (RS1B without support veto)

On $\mathcal R$ (including promotions):

\[
\text{revise-worthy}
\rightarrow
\text{candidate}
\rightarrow
\text{physical admissibility (passivity / }d_{\mathrm{eff}}\ge0\text{)}
\rightarrow
\text{short utility (H2/H4/H8)}
\rightarrow
\text{install or reject}
\]

Accept is frozen **before** blind H32. Support **never** enters this bit.

### Stage 3 — confidence monitoring (not veto)

After install, record $I_{\mathrm{exit}}$ on blind H32:

\[
\begin{aligned}
I_{\mathrm{exit}}=0 &\Rightarrow \text{normal confidence}\\
I_{\mathrm{exit}}>0 &\Rightarrow \text{revision retained; monitor / fallback-ready}
\end{aligned}
\]

The monitor flag is logged (`monitor_intensity ∈ {normal, elevated}`). It
must not flip `accepted`.

## Counterfactual baselines (evaluation only)

Computed on the same episodes; none may change $\pi_{\mathrm{RS1C}}$ accept:

| Policy | Rule |
|--------|------|
| $\pi_{\mathrm{none}}$ | never revise |
| $\pi_{\mathrm{veto}}$ | $\pi_{\mathrm{RS1C}}$ install **and then** drop if $I_{\mathrm{exit}}>0$ |
| $\pi_{\mathrm{always}}$ | revise-worthy $\Rightarrow$ install, skipping passivity/utility (diagnostic; expect safety regression) |

$\pi_{\mathrm{veto}}$ is the empirically anti-utility rule RS1B.1/B.2 rejected.
RS1C must not adopt it; the comparison asks whether keeping those installs
helps.

## Matrix

New held-out seeds: $\{10101,10111,10121,10131,10141\}$.

\[
\alpha\in\{0,-0.09,-0.12,-0.18,-0.24\}
\times
A/A_0\in\{0.5,1.0,1.5,2.0\}
\times
5\text{ seeds}
=
\boxed{100\ \mathrm{episodes}}
\]

This covers tolerate-scale, probe-scale, and revise-worthy cells under the
frozen $C_{\mathrm{tol}}$ / $D_0$ policy. Baseline $A$ is the grid $A$; VoI
may add one extra $1.5A_0$ trajectory only for $\mathcal P$ cells with
$A<1.5A_0$.

Smoke: seed `9041` × $\{\alpha=0,-0.24\}$ × $\{0.5,1.5\}A_0$ (plumbing; no GO).

## Primary endpoints

On held-out seeds ($n=5$), after accept freeze:

| Endpoint | Criterion |
|----------|-----------|
| Nontrivial installs | $\#\{\text{installed}\}>0$ |
| Passivity among installs | $0$ violations |
| False install on $\mathcal T$ | $\le 1\%$ |
| Seed-mean H32 paired gain $\pi_{\mathrm{RS1C}}$ vs $\pi_{\mathrm{none}}$ | $>0$ |
| Utility vs support-veto | seed-mean gain$(\pi_{\mathrm{RS1C}})\ge$ gain$(\pi_{\mathrm{veto}})$ |
| Monitor consistency | every installed episode with $I_{\mathrm{exit}}>0$ has `elevated`; none of those has accept flipped |

Harmful $=($not finite$)\lor(\Delta\mathrm{RMSE}<0)$, same as RS1B.1.
Report $P(\mathrm{harmful}\mid\mathrm{install},I_{\mathrm{exit}}>0)$ as a
**diagnostic**, not a veto threshold.

H32 remains blind.

## GO

\[
\begin{aligned}
RS1C\_GO
&=
(\text{nontrivial installs})\\
&\land
(\text{passivity among installs}=0)\\
&\land
(\text{false install on }\mathcal T\le 1\%)\\
&\land
(\text{seed-mean H32 gain vs none}>0)\\
&\land
(\text{gain vs veto }\ge 0)\\
&\land
(\text{monitor consistency})
\end{aligned}
\]

A passing RS1C **unlocks drafting RS2** (contact / oracle $J^\top f$).  
It does **not** rewrite RS1B_GO.

On fail: classify by layer (allocation / passivity / utility / monitoring
leakage into accept). Do not reopen $D_{\mathrm{exit}}$ formula.

## Explicit non-goals

- RS1B.3 support-metric search  
- Using RS1B.2 targeted boundary queries as the RS1C evaluation distribution  
- Contact, cameras, policies (RS2)  
- Training new networks
