# R5-I0-SELFSTRESS Preregistration — Geometric Stiffness from Hidden Preload

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REG/R5_PREREG.md`, `REPORT/REP/R5_CONTACT_FAMILY_PAUSE.md`  
Does not change: contact-family I0 v1–v3; R4 STOP  
Locks: neural probe; contact-family unpause; \(\lambda\) as an actuator command;
PD-hold that leaks preload into motor effort

## Role

New **physics family**, still I0 feasibility only. No learner.

Contact family stays **PAUSED**. This is not a fourth \(k_t/\mu/\tau\)
pattern.

Causal skeleton (analytic):

\[
\lambda
\rightarrow
\text{internal tension}
\rightarrow
\text{geometric stiffness}
\rightarrow
\text{future lateral displacement}
\]

with current equal-and-opposite tendon forces so net wrench on the
slider is zero.

## Mechanism

Central slider, lateral joint \(y\), two symmetric spatial tendons
along \(\pm x\) with the same stiffness \(k\). Preload is **passive
rest length**

\[
L_0(\lambda)=a-\lambda/k,
\]

not a control channel. External `ctrl` is identical across \(\lambda\):
hold \(u=0\), then frozen \(F^{\mathrm{diag}}\).

A small joint stiffness \(k_0>0\) keeps the linear law

\[
k_{\mathrm{eff}}(\lambda)\approx k_0+2\lambda/a
\]

away from the \(\lambda=0\) geometric singularity, so \(C\) can vary
continuously instead of jumping once and saturating.

\(\lambda\in\{0,2,\ldots,12\}\) (newtons of pretension at \(y=0\)).

\(t_\star=\) last hold frame. \(h^{S}\) is the slider macro
\((q,v,a,u,\tau_{\mathrm{motor}},u^{\mathrm{cmd}})\). Tendon tension
is **not** in \(h^{S}\).

\(X= (T_L+T_R)/2\) from tendon stretch. \(C=\|Y_\lambda-Y_0\|_2\) on
the future macro trajectory (relative \(\ell_2\)).

## Gates

Unchanged overlap / order thresholds, plus a range gate and an
\(X\)–\(C\) coupling gate (v3 showed Spearman\((\lambda,C)\) can fail
by saturation even when \(\mathrm{range}(C)\) is large, and a tiny
monotone \(C\) can pass Spearman alone):

- **G_macro:** \(\max_\lambda D(h^{S}_\lambda,h^{S}_0)<0.05\)
- **G_local:** Spearman\((\lambda,X)>0.80\)
- **G_cons:** Spearman\((\lambda,C)>0.80\)
- **G_range:** \(\max C-\min C>0.10\)
- **G_xc:** Spearman\((X,C)>0.80\)

\[
\texttt{R5\_I0\_SELFSTRESS\_PASS}
=G_{\mathrm{macro}}\land G_{\mathrm{local}}\land G_{\mathrm{cons}}
\land G_{\mathrm{range}}\land G_{\mathrm{xc}}.
\]

No neural probe if any gate fails. Do not pass by putting tension into
\(h^{S}\), by PD-holding \(y\) against preload, or by retuning
\(F^{\mathrm{diag}}\) after seeing \(C\).
