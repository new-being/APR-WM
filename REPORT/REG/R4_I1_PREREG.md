# R4-I1 Preregistration — No-Leak Audit of Full \(h^{S}\)

Date: 2026-08-17  
Status: **FROZEN** (infrastructure; not R4-C0 GO)  
Depends on: R4-I0 **v2** `PASS=true` (`REPORT/REP/R4_I0_V2_REPORT.md`)  
Does not change: I0 v2 protocol (\(\mu\), `solref`, taxel grid, squeeze,
\(F_t(t)\)), `REPORT/REG/R4_C0_PREREG.md` question  
Locks: I0 retuning; encoder/GRU search; RGB; R4-C0 until I1 passes

## Role

I0 v2 constructed

\[
D(h^{S,A},h^{S,B})\approx 0,\qquad D(X^A,X^B)\gg 0.
\]

I1 asks whether that split **survives a strong, over-complete
learner-visible macro baseline** — not whether a weak \(h^{S}\) can be
beaten by stuffing taxels.

\[
\boxed{
R4\text{-I1 — No-leak audit with full }h^{S}
}
\]

## Frozen I0 protocol

Do **not** retune \(\mu\), `solref`, grid, squeeze, or \(F_t(t)\).
A/B remain left/right compliance swap. Additional trajectories vary
only **initial block \(q_0\)** (named trajectories), same ramp.

## \(h^{S}\) (over-complete on purpose)

Per time, include every macro channel a deployed model could see:

\[
q,\;\dot q,\;\ddot q,\;
q_{\mathrm{finger}},\;\dot q_{\mathrm{finger}},\;\ddot q_{\mathrm{finger}},\;
u_{\mathrm{finger}},\;F_t^{\mathrm{cmd}},\;
F_n,\;F_t,\;F_{x,y,z},\;M_{x,y,z}.
\]

History: last \(W=8\) steps concatenated (linear probe, **no GRU**).
Do **not** include per-taxel or per-finger split forces (those are \(X\)
or would smuggle the spatial pattern).

## Split (hard)

\[
\boxed{\text{split by trajectory }q_0\text{, not by random frames}}
\]

Train \(q_0\in\{-8,-4,0,+4,+8\}\times 10^{-4}\).  
Held-out \(q_0\in\{-12,+12\}\times 10^{-4}\).

Probe is fit on train matched frames only; AUROC is **held-out
trajectories**.

## Gates

Linear probes, no encoder.

**H1 (no leak).** Held-out \(\mathrm{AUROC}(h^{S})<0.90\).
If \(\ge 0.95\): fail, do not train tactile nets.

**H2 (conditional gain).** Held-out

\[
\mathrm{AUROC}(h^{S},X)-\mathrm{AUROC}(h^{S})>0.10.
\]

\[
\texttt{R4\_I1\_PASS}=H1\land H2.
\]

## Diagnostic (not GO)

Because A/B is a left/right swap:

- bilateral \(X\mapsto X(-x)\) on both classes is equivariant: a linear
  probe can relearn the swapped pattern, so AUROC need not drop;
- **canonicalize** (flip B onto A's \(x\) frame) should collapse toward
  chance.

The canonicalized drop is the spatial-allocation check. It is not a GO
gate.

## Unlock

`R4_I1_PASS` unlocks implementing `R4_C0_PREREG.md` only.
