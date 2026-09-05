# R5-D0 Preregistration — Conditional Decision Value of Hidden Preload

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REG/R5/R5_D0_PREFLIGHT_PREREG.md`,
`REPORT/REP/R5/R5_D0_PREFLIGHT_REPORT.md`  
Does not change: I0 `PASS=false`; I1 `GO=true`; preflight \(\mathcal A,J\);
contact PAUSE  
Locks: \(X\mapsto a^\star\) lookup/classifier; Adam / neural planner;
retuning \(\mathcal A\) or \(J\); using accuracy as the sole endpoint;
new scalar \(C\)

## Question

Does using \(X\) reduce **executed decision regret** beyond \(h^{S}\),
through a world-model chain rather than an action classifier?

\[
(h^{S},X,a)\rightarrow\hat Y^{\mathrm{future}}(a)\rightarrow J(\hat Y)\rightarrow\hat a.
\]

## Locked objects

Physics, \(\lambda\) grid, \(\mathcal A=\{0.4,1.0,2.5\}\), and \(J\)
are those of D0-preflight. Every train \(\lambda\) has counterfactual
rollouts of **all three** actions.

## Planners (ridge, closed form, no Adam)

Per action \(a\), a ridge map to the physical \(Y\) trajectory
(same keys as I0 `y_future`). Features z-scored on the fold's train
rows; targets remain in physical units so \(J\) is not applied in
z-space.

- \(\pi_0\): \(h^{S}\mapsto\hat Y(a)\) then \(\hat a_0=\arg\min_a J(\hat Y(a))\)
- \(\pi_X\): \((h^{S},X)\mapsto\hat Y(a)\) then \(\hat a_X=\arg\min_a J(\hat Y(a))\)
- \(\pi_{X_{\mathrm{shuf}}}\): same as \(\pi_X\) after a frozen permutation
  of \(X\) across the seven \(\lambda\) (seed \(0\))

Executed cost is the **true** simulator \(J(a_\pi,\lambda)\), not
\(J(\hat Y)\).

## Split

Leave-one-\(\lambda\)-out over \(\{0,2,\ldots,12\}\). This is required
because the only oracle flip is \(\lambda=0\) vs \(\lambda\ge 2\); a
held set contained in \(\lambda\ge 2\) would let a constant-`mid`
policy look strong on accuracy. Accuracy is auxiliary.

## Endpoint

\[
R(\pi,\lambda)=J(a_\pi,\lambda)-J(a^\star(\lambda),\lambda).
\]

Means are uniform over the seven LOO folds.

## Gates

- **H1 (primary):** \(\bar R_X<\bar R_0\) and
  \((\bar R_0-\bar R_X)/(\bar R_0+10^{-12})>0.05\)
- **H2:** \(\mathrm{Acc}_X>\mathrm{Acc}_0\) (also report \(\lambda=0\)
  and \(\lambda\ge 2\) separately)
- **H3:** \(P(\hat a_X\neq\hat a_0)>0\)
- **H4:** \(\bar R_X<\bar R_{X_{\mathrm{shuf}}}\) and
  \((\bar R_{\mathrm{shuf}}-\bar R_X)/(\bar R_{\mathrm{shuf}}+10^{-12})>0.05\)

\[
\texttt{R5\_D0\_GO}=H_1\land H_2\land H_3\land H_4.
\]

If GO is false, stop the family at **realized** \(\mathrm{VoI}_{\Pi}<0\)
for this \(\Pi\). That does not assert oracle \(\mathrm{VoI}\le 0\).
Do not retune the planner class to rescue a near miss.
