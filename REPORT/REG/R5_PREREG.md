# R5 Preregistration — Consequence-Varying Hidden Contact

Date: 2026-08-17  
Status: **FROZEN** (contact **PAUSED**; selfstress I0 `PASS=false`;
I1 `GO=true`; D0-preflight `PASS=true`; no encoder)  
Depends on: `REPORT/REP/R4_FAMILY_STOP.md`,
`REPORT/REP/R5_CONTACT_FAMILY_PAUSE.md`  
Does not change: R4 STOP, Door V7, I0 v1–v3 numbers  
Locks: R4-C3; R5 encoder/GRU; treating I1 A/B as R5 GO; a fourth
contact-surface pattern; `R5_CONTACT_IMPOSSIBLE`

## Placement

\[
\boxed{
R3/V7:\text{ conditional modality value}
\rightarrow
R4:\text{ conditional consequential information}
\rightarrow
R5:\text{ consequence-varying hidden state}
}
\]

R4 is stopped. The contact-family I0 search is **paused**, not proven
impossible, and not replaced by a learner.

## Admission (unchanged)

\[
\boxed{
\exists\lambda\text{ on an interval:}\quad
\frac{\partial X}{\partial\lambda}\neq 0,\quad
\frac{\partial C}{\partial\lambda}\neq 0,\quad
\frac{\partial h^{S}}{\partial\lambda}\approx 0
}
\]

and \(\operatorname{Var}(C\mid h^{S})>0\) with a stable
\(X(\lambda)\)–\(C(\lambda)\) map. Only then is \(I(X;C\mid h^{S})>0\)
a well-posed learner question.

## Contact family (closed as a search, not as a theorem)

I0 v1–v3 (`REPORT/REG/R5_I0_PREREG.md`) exhausted the intended surface
mechanisms (\(\mu\), self-equilibrated \(\tau\), nullspace \(k_t\)).
v3 achieved \(D(h^{S})\approx 0\) and ordered \(X\), not ordered \(C\).

\[
\boxed{
\text{conditional observability}
\neq
\text{conditional consequential observability}
}
\]

inside this family. Ledger: `REPORT/REP/R5_CONTACT_FAMILY_PAUSE.md`.

## If a later R5 family exists (not now)

Not another \(\phi\) on the same pads. Internal-state families need a
new I0. **R5-I0-SELFSTRESS** is that I0
(`REPORT/REG/R5_I0_SELFSTRESS_PREREG.md`); I0 `PASS=false` on scalar
\(C\). **R5-I1-SELFSTRESS** (`REPORT/REG/R5_I1_SELFSTRESS_PREREG.md`)
then tests \(I(X;Y^{\mathrm{future}}\mid h^{S})\) without rewriting
\(C\); first run `GO=true`
(`REPORT/REP/R5_I1_SELFSTRESS_REPORT.md`). **R5-D0-preflight**
(`REPORT/REG/R5_D0_PREFLIGHT_PREREG.md`) found an oracle ranking
crossover (`PASS=true`); that unlocks a later regret D0, not an
encoder. Contact family stays paused.
