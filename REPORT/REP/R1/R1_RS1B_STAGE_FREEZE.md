# R1-RS1B Stage Freeze — Support-Risk Branch Closed

Date: 2026-08-16  
Status: **STAGE FROZEN**  
Does not change `RS1B_GO=false`. Does not install a support hard filter.

## One-line verdict

\[
\boxed{
\text{support is a confidence-state variable, not an accept/reject safety variable}
}
\]

**No RS1B.3.** Further $D_{\mathrm{exit}}$ metric tuning is closed unless a
separate paper-sized question is opened later (“which variable predicts harm
*after* extrapolation”). Marginal return on the support-distance line has
dropped.

## Frozen chain

\[
\boxed{
\begin{aligned}
RS1B &: \text{passive + useful revisions can still leave modeled support}\\
RS1B.1 &: \text{support exit raises absolute difficulty, not harm}\\
RS1B.2 &: \text{intermediate }D\text{ is reachable; risk is mostly a }D=0\text{ crossing}
\end{aligned}
}
\]

## Permanently frozen (do not reopen in RS1C as veto)

- $I_{\mathrm{exit}}>0 \Rightarrow$ reject revision  
- $D_{\mathrm{exit}}$ as a proportional risk score for acceptance  
- Lowering RS1B’s $0.90$ `stable_h32` bar after seeing H32  
- Treating RS1C as a repair of `RS1B_GO`

RS1B_GO remains **false** on the original protocol. The scientific reading is:
frozen binary modeled-support *retention* is a bad harm surrogate, not that
long-horizon prediction failed.

## Symmetry with RS1A

\[
\boxed{
\begin{aligned}
RS1A &: \text{error magnitude}\not\Rightarrow\text{worth revising}\\
RS1A.5 &: \text{more excitation}\notRightarrow\text{more net epistemic value}\\
RS1B.1 &: \text{support exit}\not\Rightarrow\text{revision harmful}\\
RS1B.2 &: \text{larger support distance}\not\Rightarrow\text{proportionally larger risk}
\end{aligned}
}
\]

General principle:

\[
\boxed{
\text{raw diagnostic magnitude should not be used directly as a decision rule}
}
\]

Calibrate through consequence / value / confidence first.

## Handoff

Next stage is a **new** preregistered policy: `REPORT/REG/R1/R1_RS1C_PREREG.md`.  
RS2 stays locked until RS1C.
