# R5-I0 Preregistration — Physical Feasibility of Consequence-Varying \(\lambda\)

Date: 2026-08-17  
Status: **FROZEN** (v1–v3 run; contact family PAUSED)  
Depends on: `REPORT/REG/R5_PREREG.md`, `REPORT/REP/R4_FAMILY_STOP.md`  
Does not change: R4 family STOP; v1 / v2 results  
Locks: neural probes; moving \(t_\star\) into the probe; encoder;
deleting \(f_t\); deleting endpoints; raising the diagnostic action

## Role

I0-style **feasibility search**. No learner. \(t_\star=\) last hold frame.
Admission is a nullspace question for the official learner-visible
operator \(h^{S}=A(x)\):

\[
J_A(x_0)v\approx 0,\qquad J_X(x_0)v\neq 0,\qquad \nabla C(x_0)^\top v\neq 0.
\]

## v1 (run, fail G_local)

Mean-preserving \(\mu(\lambda)\). Future \(C\) ordered, \(h^{S}\) overlap,
but \(X\equiv0\) at hold (\(F_t^{\mathrm{cmd}}=0\)). Lesson:

\[
\text{future-consequential DOF}
\neq
\text{currently observable consequential DOF}.
\]

## v2 (run, fail G_macro / G_local / G_cons)

Self-equilibrated opposing shear. Signed \(F_x,M_z\) cancelled; \(X\) saw
the couple on \([0,0.75]\); official \(h^{S}\) still moved because
\(f_t=\sum|\tau|\). Frozen probe \(C\) unordered. Lesson: macro-null means
the **whole** \(A(x)=h^{S}\), not just signed wrench.

## v3 (run, fail G_cons)

Quadrupole \(k_t(\lambda)\). Official \(h^{S}\) null
(\(\max D_h=4.5\times10^{-4}\)); \(X\) ordered (Spearman \(0.857\));
\(C\) jumps to \(\approx 0.61\)–\(0.66\) and saturates (Spearman
\(-0.214\)). Lesson: \(I(X;\lambda\mid h^{S})>0\) does not give
\(I(X;C\mid h^{S})>0\).

Contact family **PAUSED** (`REPORT/REP/R5_CONTACT_FAMILY_PAUSE.md`).
Not `R5_CONTACT_IMPOSSIBLE`. No fourth surface mechanism. No neural
probe.
