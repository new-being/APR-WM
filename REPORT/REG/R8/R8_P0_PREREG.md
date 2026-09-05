# R8-P0 Preregistration — Active Certificate-Validity Feasibility

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R7/R7_B1_REPORT.md` (`shift_valid=false`)  
Does not: train a validity model; \(\tau_{\mathrm{valid}}\); \(\pi\);
WM update; conformal recalibration; neural detector; \(J\)/ \(A\) in
\(S_{\mathrm{epi}}\)

## Question

Can a cheap frozen diagnostic action, applied **before** mid/cons,
produce evidence that the frozen B0 certificate is still applicable
— not evidence that \(F_{\max}\) equals 2?

\[
\boxed{
\mathcal V=\mathbf 1\{\text{frozen WM + certificate remains decision-safe here}\}.
}
\]

Passive \((h^S,X)\) cannot work: at \(t_\star\),
\(u=y=\dot y=0\) so \(I(h^S,X;F_{\max})=0\).

## Frozen objects

B0 \(f_{\mathrm{WM}}\) and \(q\). \(\delta=10^{-3}\). \(X=0.25\alpha\).
Held \(\alpha\): B0 test grid (certificate domain).

## Locked \(F_{\max}\) classes (oracle audit only)

| class | \(F_{\max}\) | oracle \(\mathcal V\) on B0 test (coverage \(\ge 0.90\), no harmful commit) |
|---|---|---|
| ID | \(2.0\) | valid |
| benign shift | \(2.2\) | valid (\(F_{\max}\neq 2\), certificate still safe) |
| invalidating shift | \(1.5\) | invalid (B1) |

Gates that compute \(S_{\mathrm{epi}}\) **do not** take \(F_{\max}\) or
\(\mathcal V\) as inputs. Oracle labels are used only to check
alignment after the fact.

## Frozen probe

\[
a^{\mathrm{epi}}=u_{\mathrm{default}}=1.0,\qquad T_{\mathrm{epi}}=0.10\,\mathrm{s}
\]

(\(n=50\) steps; matches B0 first WM knot index 49). After hold-at-rest,
apply \(a^{\mathrm{epi}}\), observe \(y(T_{\mathrm{epi}})\). WM predicts
that knot: \(\hat y=f_{\mathrm{WM}}(X,a^{\mathrm{epi}})_{k=0}\).

\[
S_{\mathrm{epi}}=\lvert y(T_{\mathrm{epi}})-\hat y\rvert.
\]

No \(A\), \(J\), \(a^\star\). Evidence is obtained before any mid/cons
commit (same trial prefix).

Alignment is scored on the **frozen commit-set**
\(\{L_{\mathrm{WM}}>\delta\}\), which depends only on \(X\) (seven B0
held \(\alpha\)). Low-\(\alpha\) points never commit; \(F_{\max}\) is
decision-irrelevant there.

## Costs

- \(C_{\mathrm{prefix}}=(y_{T_{\mathrm{epi}}})^2+\beta u_{\mathrm{epi}}^2\)
- \(C_{\mathrm{down}}=J(\text{epi then default }0.50\,\mathrm{s})-J(\text{default from rest})\)

G3 uses \(C_{\mathrm{down}}\) vs mean \(\lvert A\rvert\) on the ID
commit-set: the probe must not eat typical task advantage in the
**same episode**.

## Gates (all required)

- **G0** passive: \(\max D(h^S)=0\) across classes (\(\varepsilon_h=0.05\))
- **G1** validity-aligned separation on commit-set:
  \(\min S_{\mathrm{invalid}}-\max(S_{\mathrm{ID}},S_{\mathrm{benign}})\ge 10^{-3}\)
- **G2** timing: probe prefix before mid/cons (structural)
- **G3** \(\mathrm{mean}\,C_{\mathrm{down}}/\mathrm{mean}\lvert A\rvert_{\mathrm{commit,ID}}\le 1\)

If G1 fails: do not train a stronger detector. If only G3 fails:
identifiable but too expensive; do not proceed to R8-A0 with this
pulse. No \(\tau\), no \(\pi\), no NetVoI policy yet.
