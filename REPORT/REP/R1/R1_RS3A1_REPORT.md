# R1-RS3A.1 Report — Cross-Fitted Structural Excess-Risk Evidence

Date: 2026-08-16  
Prereg: `REPORT/REG/R1/R1_RS3A1_PREREG.md`  
Depends on: `REPORT/REP/R1/R1_RS3A_REPORT.md`  
Artifacts: `runs/r1_rs3a1/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `30d500dc2e212491eb6d8aa8167d3cfaf2b5fa59afcb038aa5868cfba2ac2140`

## Decision

\[
\boxed{RS3A.1\_GO=\mathrm{false}}
\]

\[
\boxed{
\text{cross-fitted excess structural risk }E_{\mathrm{XR}}
\text{ does not transport evidential meaning across Mode-A and contact}
}
\]

This was the last strong attempt at an intervention-free scalar
structural-evidence ruler at this representation. It does **not** rewrite
`RS3A_GO=false` or `RS2_GO=false`. It does **not** unlock RS3B/C, a
domain-specific \(\tau_E\), or another \(S_k\) scalar.

**The intervention-free scalar programme is closed.**

\[
\boxed{
\text{No intervention-free scalar tested so far preserves
structural-evidence semantics.}
}
\]

The failure is not only scale drift. On C1 the constructions **rank
regimes differently** (\(D_0^{\mathrm{ModeA}}>D_0^{\mathrm{contact}}\)
while \(E_{\mathrm{XR}}^{\mathrm{ModeA}}<E_{\mathrm{XR}}^{\mathrm{contact}}\)):

\[
\boxed{
\text{different evidence constructions rank intervention regimes
differently}
}
\]

\[
\boxed{
\text{extra-capacity benefit is itself intervention-dependent}
}
\]

Do not open RS3A.2. The next main stage, if any, is **RS4A**
(`REPORT/REG/R1/R1_RS4A_PREREG.md`): geometry-conditioned evidence, not a
fourth invariant scalar. RS3B/C stay locked.

Smoke (`runs/r1_rs3a1/smoke/`) is plumbing only.

## Question (unchanged)

\[
\boxed{
\text{Does extra frozen structural capacity reduce held-out predictive
risk by a similar amount under Mode-A and contact, without domain ID?}
}
\]

Not contact AUROC. Not a quieter local null.

## Statistic

Probe window only. Common budget \(K_f=K_h=100\), two-fold cross-fit.
\(M_{\mathrm{par}}\): incumbent \(\hat\theta\) only.
\(M_{\mathrm{flex}}\): same \(\hat\theta\) plus frozen Door-box 12-RBF
(\(q\in\{-0.4,0,0.4,0.8\}\), \(v\in\{-0.8,0,0.8\}\), ridge \(2\times10^{-3}\),
clamp \(\pm0.4\)).

\[
E_{\mathrm{XR}}
=
\frac{L_{\mathrm{par}}-L_{\mathrm{flex}}}{L_{\mathrm{par}}+L_{\mathrm{flex}}+\epsilon}.
\]

No quiet H0, no domain ID, no true \(\phi\). \(n_{\mathrm{used}}=200\) on
every scientific episode.

## Matrix

Seeds \(\{13101,13111,13121,13131,13141\}\).
\(\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H}\}\), no C2. \(N=90\).

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H1a C0 medians \(\le 0.15\) | **PASS** | Mode-A \(-0.030\); contact \(0.0009\) |
| H1b shared \(\tau_E\) FPR | **FAIL** | \(\tau_E=-0.0071\); Mode-A \(0/10\); contact \(12/20=0.60>0.05\) |
| H2 \(g(E_{\mathrm{XR}})<g(D_0)\) | **FAIL** | adapter C1: \(g(E)=1.32\not< g(D_0)=0.691\) |

\[
\boxed{RS3A.1\_GO=H1a\land H1b\land H2=\mathrm{false}}
\]

## What the numbers actually say

H1a is the important negative control: the frozen RBF is **not** a
universal memorizer. Extra capacity does not buy held-out risk on C0 in
either domain.

H1b fails for the same logical reason RS3A H1 was over-readable. Mode-A
C0 \(E_{\mathrm{XR}}\) is entirely negative (RBF slightly hurts:
\([-0.102,-0.007]\)), so \(\tau_E=\max S_{\mathrm{ModeA,C0}}\) sits at
\(-0.007\). Contact C0 is near zero but often just positive
(\(\mathrm{median}\,0.0009\), range \([-0.079,0.027]\)), hence
\(12/20\) exceed that Mode-A ceiling. Two clouds can both be “about zero”
and still not share a scale.

Adapter pair, C1:

| Domain | median \(D_0\) | median \(E_{\mathrm{XR}}\) |
|--------|---------------:|---------------------------:|
| Mode-A \(1.5A_0\) | \(2.26\times10^{-3}\) | \(0.051\) |
| contact `pull_release@\(s^\star\)` | \(1.10\times10^{-3}\) | \(0.249\) |

\(D_0\) is larger on Mode-A; \(E_{\mathrm{XR}}\) is larger on contact.
The operating-point gap **widens** relative to raw \(D_0\), and the
**direction of the domain split reverses**. That is not a transported
inadequacy ruler.

A consistent reading: the 12-RBF leftover capacity barely explains
Mode-A \(|v|v\) on a sine probe (\(E_{\mathrm{XR}}\approx0.05\)), while
contact probe occupancy lets the same extra capacity reduce held-out
force error more (\(E_{\mathrm{XR}}\approx0.25\)). Capacity help is still
\(\mathcal I\)-dependent.

## What this closes

Together with RS3A:

\[
\boxed{
\begin{aligned}
D_0 &\text{ amplitude does not transport}\\
S_\perp &\text{ local-null whitening does not transport}\\
E_{\mathrm{XR}} &\text{ held-out excess structural risk does not transport}
\end{aligned}
}
\]

\[
\boxed{
\text{a domain-free scalar structural-evidence ruler may not exist
at this level of representation}
}
\]

**Do not** open RS3A.2 / another scalar. **Do not** repair H1b by moving
\(\tau_E\). **Do not** enlarge the RBF per domain.

The scientifically honest next question, under a **new** preregistration
if opened at all, is intervention-conditioned epistemics

\[
\pi(E,C,V\mid\mathcal I),
\]

not a better invariant score. RS3B/C stay locked until that
preregistration exists; they are not unlocked by this failure.

## Frozen GO status after this report

\[
\boxed{
\begin{aligned}
RS1C\_GO &= true\\
RS2\text{-C0\_PASS} &= true\\
RS2\_GO &= false\\
RS2A\_GO &= true\\
RS2B\_GO &= true\\
RS3A\_GO &= false\\
RS3A.1\_GO &= false\\
RS3B/RS3C &= locked
\end{aligned}
}
\]
