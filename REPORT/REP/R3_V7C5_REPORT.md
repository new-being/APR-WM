# R3-V7C.5 Report — Belief Integration of \(Z_{NSG}\)

Date: 2026-08-16  
Prereg: `REPORT/REG/R3_V7C5_PREREG.md`  
Depends on: `REPORT/REP/R3_V7C4_REPORT.md` (`verdict=SUFFICIENT`)  
Artifacts: `runs/r3_v7c5/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `2f022ad403ca3cceeb4a8e6099ac75b5a1d272341e87c6355f46e7b244fa9630`

## Decision

\[
\boxed{V7C.5\_GO=\mathrm{false}}
\]

\[
\boxed{\text{epistemic usefulness: FAIL}}
\]

Representation sufficiency (V7C.4) stays **PASS**. This stage does
**not** train a new encoder family, does **not** rewrite FIELD, and
does **not** open V7D.

Smoke did not evaluate GO.

## Question (unchanged)

Can \(Z_{NSG}\) improve evidence-warranted \(p_t\) when **added** to
frozen B5-S, via time-locked contact evidence?

## Design (frozen)

- Seeds \(\{26101,26111\}\) train, \(26121\) val, \(\{26131,26141\}\)
  held-out.
- Three trained arms: **B5-S**, **B5-S+\(Z\)**, **B5-S+\(Z_{\mathrm{shuf}}\)**.
- Encoder architecture frozen as V7C.4 (2-channel CNN + geometry MLP).
- \(z_t\) concatenated to the 12-D B5-S step. Shuffle permutes \(z\)
  in time inside each episode; B5-S steps stay aligned.
- \(L\) = held-out BCE vs \(e_t=y\,w_t\).
- \(\varepsilon_{\mathrm{temporal}}=0.005\).

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1 \(S{+}Z\) Brier \(\le S+\delta\) | **FAIL** | mid \(0.184\le0.166+0.02\); final \(0.183>0.160+0.02\) |
| H2 C0 step-FPR \(\le 0.20\) | **PASS** | FPR \(0.067\) |
| H3 Spearman\((p,w)>0\) and high \(w>\) low \(w\) | **PASS** | \(\rho=0.028\); \(0.636>0.365\) |
| H4 \(S{+}Z\) useful vs B5-S | **FAIL** | \(B_{\mathrm{C0}}^{SZ}=0.217>0.189\); C1-final \(0.175>0.145\); pooled Brier also worse |
| H5 \(\Delta_{\mathrm{real}}-\Delta_{\mathrm{shuf}}>\varepsilon_{\mathrm{temporal}}\) | **FAIL** | \(\Delta_{\mathrm{real}}=-0.046\); \(\Delta_{\mathrm{shuf}}=-0.041\); gap \(-0.005\not>0.005\) |

\[
\boxed{V7C.5\_GO=H1\land H2\land H3\land H4\land H5=\mathrm{false}}
\]

## How to read the failure

Adding \(Z_{NSG}\) **hurt** the warranted-evidence objective relative
to B5-S alone:

\[
L(S)=0.442,\qquad L(S{+}Z)=0.488,\qquad L(S{+}Z_{\mathrm{shuf}})=0.483.
\]

So this is not “Brier improved from a static fingerprint.” The fused
channel is worse than proprioceptive B5-S on \(e_t\) **and** on V7C
belief metrics. Shuffle is about as bad as real alignment; the
temporal gate therefore fails, as it should.

\[
\boxed{
I(Z_{NSG};e_t\mid h^{-c})>0
\not\Rightarrow
I(Z_{NSG};e_t\mid h^{S})>0
}
\]

C.4’s probe was \(h^{-c}\) (contact slots zeroed). B5-S already
carries noisy wrist/joint contact. On this Door mixture, \(Z_{NSG}\)
is **conditionally redundant or harmful given B5-S**, even though it
was probe-sufficient given \(h^{-c}\).

That also tightens V7C.1: the earlier tactile-for-belief failure is
not explained by “the encoder must have dropped \(X_{NSG}\).” After a
SUFFICIENT representation, **belief integration still fails**.

H3’s Spearman is positive but small (\(0.028\)); occupancy stays
legal (H2). Those do not rescue H1/H4/H5.

## What this does not authorize

- encoder architecture search
- stuffing \(Z\) into V7D / RGB
- rewriting V7C.2 FIELD or V7C.4 SUFFICIENT
- occupancy penalties / GRU-gate edits
- treating shuffle-null C.4 as a reason to skip this stage (this
  stage was the right test)

## What follows

\[
\boxed{
\begin{aligned}
\text{representation sufficiency} &\quad\text{PASS (C.4)}\\
\text{epistemic usefulness vs B5-S} &\quad\text{FAIL (C.5)}\\
V7D &\quad\text{locked}
\end{aligned}
}
\]

Do **not** continue \(X\to Z\). A later tactile step, if any, would
need a new prereg on a distribution where patch geometry is not
recoverable from wrist/joint proprioception (slip / friction /
distributed compliant contact)—not a wider CNN on this Door mixture.
