# R5 Self-Stress Family Stop

Date: 2026-08-17  
Status: **STOPPED**  
Does not change: contact-family PAUSE; I0 `PASS=false`; I1 `GO=true`;
D0-preflight `PASS=true`; R4 STOP  
Does not open: \(X\to a^\star\) classifier; retuning \(J\) / ridge;
encoder; claiming oracle \(\mathrm{VoI}(X)\le 0\)

## Three-layer split (plus I0)

\[
\boxed{
\begin{aligned}
\textbf{Predictive relevance:}
&\quad I(X;Y^{\mathrm{future}}\mid h^{S})>0
&&\checkmark\\
\textbf{Oracle decision relevance:}
&\quad \operatorname{Var}_\lambda(a^\star)>0
&&\checkmark\\
\textbf{Finite-planner realization:}
&\quad \bar R_X<\bar R_0
&&\times
\end{aligned}
}
\]

Ledger:

\[
\boxed{
\begin{aligned}
\text{I0}&:\ C=\|Y-Y_0\|\text{ as global coordinate}\quad\times\\
\text{I1}&:\ X\rightarrow Y^{\mathrm{future}}\text{ conditional predictive gain}\quad\checkmark\\
\text{D0-preflight}&:\ Y\text{ differences induce oracle ranking flip}\quad\checkmark\\
\text{D0}&:\ X\text{ via frozen }\Pi\text{ yields lower executed regret}\quad\times\\
\text{self-stress family}&:\textbf{STOP}
\end{aligned}
}
\]

## What D0 falsifies

\[
\boxed{
I(X;Y^{\mathrm{future}}\mid h^{S})>0
\ \land\
\operatorname{Var}_\lambda(a^\star)>0
\ \not\Rightarrow\
\mathbb{E}[R(\pi_X)]<\mathbb{E}[R(\pi_0)]
}
\]

**for the registered ridge-\(\hat Y\)-then-\(J\) planner \(\Pi\).**

Oracle VoI, in the sense “an optimal decisioner may use \(X\)”, is
positive: \(h^{S}\) cannot distinguish \(\lambda\), while \(a^\star(0)
\neq a^\star(\lambda\ge 2)\). Realized

\[
\mathrm{VoI}_{\Pi}(X)=\mathbb{E}[R(\pi_0)]-\mathbb{E}[R(\pi_X)]
\]

is **negative**. Do not write oracle \(\mathrm{VoI}(X)\le 0\).

\[
\boxed{
\text{conditional predictive value}
\neq
\text{realized decision value}
}
\]

Acquisition and realization are separate:

\[
\text{epistemic acquisition}
\rightarrow
\text{decision realization}.
\]

Trajectory MSE down does not imply correct \(\arg\min_a J(\hat Y_a)\).
That split is audited offline in **R6-A0**
(`REPORT/REP/R6/R6_A0_REPORT.md`): ranking errors are \(\rho_b<-1\)
events; global \(L_Y\downarrow\) is not decision-sensitive improvement.
R6-A0 does **not** reopen this family, retune ridge / \(J\), or train
a classifier. **R6-B0** closed the abstention layer algebraically
(`B0_POLICY_COLLAPSE=true`; `REPORT/REP/R6/R6_B0_REPORT.md`). Do not
sweep \(z_{\min}\) here. R6 realization is frozen
(`REPORT/REP/R6/R6_REALIZATION_FREEZE.md`). **R7-P0** passed on a new
family (`REPORT/REP/R7/R7_P0_REPORT.md`); R7-A0 is licensed, not executed.

Report: `REPORT/REP/R5/R5_D0_REPORT.md`.
