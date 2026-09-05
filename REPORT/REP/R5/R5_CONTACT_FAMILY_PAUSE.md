# R5 Contact-Family Pause

Date: 2026-08-17  
Status: **PAUSED** (not impossible; not a new family)  
Does not change: R4 family STOP; Door V7 close; I0 v1–v3 numbers  
Does not train: R5 encoder / neural probe  
Does not open: a fourth \(k_t/\mu/\tau\) pattern; `R5_CONTACT_IMPOSSIBLE`

Canonical ledger for this pause. Numbers: `REPORT/REP/R5/R5_I0_REPORT.md`,
`REPORT/REP/R5/R5_I0_V2_REPORT.md`, `REPORT/REP/R5/R5_I0_V3_REPORT.md`.

## Frozen stage result

\[
\boxed{
\begin{aligned}
v1 &: C\checkmark,\ X\times\\
v2 &: X\text{ partially }\checkmark,\ h^{S}\times,\ C\times\\
v3 &: h^{S}\checkmark,\ X\checkmark,\ C\times\\
\text{contact family} &: \textbf{PAUSE}
\end{aligned}
}
\]

v3 removed the last major ambiguity **inside this family**: the failure
is no longer macro leak, and no longer “tactile cannot see \(\lambda\)
now”. It is that the continuous DOF tactile tracks does **not**
correspond to a continuous future-consequence DOF.

## Strongest supported boundary

\[
\boxed{
\text{conditional observability}
\neq
\text{conditional consequential observability}
}
\]

The former is only

\[
I(X;\lambda\mid h^{S})>0.
\]

R5 admission needs

\[
\boxed{I(X;C\mid h^{S})>0}
\]

with a continuous coordinate \(\lambda\mapsto X(\lambda)\),
\(\lambda\mapsto C(\lambda)\). v3 showed the first can be arranged in
the official \(h^{S}\) nullspace; the second did not follow.

This is one step past R4: the scarce object is **observing
decision-relevant variation**, not merely observing a hidden state.

## Scope (not an impossibility theorem)

Supported:

> Under the tested local contact mechanisms, the official strong
> \(h^{S}\), and the frozen future probe, no continuous contact DOF
> made the three gates coexist.

Not supported, and **not** written:

\[
\text{strong }h^{S}
\Rightarrow
I(X;C\mid h^{S})=0
\quad\text{for all contact systems.}
\]

Hence **PAUSE**, not `R5_CONTACT_IMPOSSIBLE`.

## What is locked in this family

- fourth / fifth spatial contact trick (\(\mu\), prestress, \(k_t\),
  another \(\phi\))
- neural probe on these traces
- deleting \(f_t\), dropping \(\lambda\) endpoints, moving \(t_\star\)
  into the probe, raising \(a^{\mathrm{diag}}\) to manufacture a
  common interval

## If R5 is reopened later (not now)

Do **not** resume this contact family. Open a **new physics family**
whose causal skeleton already looks like

\[
\lambda
\rightarrow
\begin{cases}
X_{\mathrm{local}}(\lambda)\\
C_{\mathrm{future}}(\lambda)
\end{cases}
,\qquad
\lambda\not\rightarrow h^{S}_{\mathrm{current}}.
\]

That is likelier in systems with **internal state** (internal strain,
preload, hidden constraint, continuum deformation) than in further
nullspace search over surface traction patterns.

The restart question is:

\[
\boxed{
\text{which physical systems naturally have an internal state that is}
\text{ locally visible now, macro-hidden now, and continuously}
\text{ future-consequential?}
}
\]

not “can we design a cleverer tactile pattern”. That would need a new
I0, written before any learner.

A different family, **R5-I0-SELFSTRESS**, was opened as feasibility
only (`REPORT/REG/R5/R5_I0_SELFSTRESS_PREREG.md`). It does **not** unpause
this contact file. I0 `PASS=false` on scalar \(C\) remains frozen.
R5-I1-SELFSTRESS (`REPORT/REP/R5/R5_I1_SELFSTRESS_REPORT.md`) tests
\(I(X;Y^{\mathrm{future}}\mid h^{S})\) without rewriting that \(C\).
