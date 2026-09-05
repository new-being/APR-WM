# APR-WM V6R0.6: Passivity-feasible, Utility-accepted Revision

## Executive conclusion

R0.6 passes its preregistered confirmatory gates on five untouched SAPIEN
seeds. The two-layer architecture is:

\[
\boxed{
\text{physical admissibility}
\rightarrow
\text{predictive utility}
\rightarrow
\text{persistent revision}
}
\]

For the native-dissipation R0-N regime, passivity + utility achieves:

- H16 RMSE `0.23468` versus `0.32833` for no revision;
- `100%` H16 stability among 102 accepted revisions;
- `85%` nontrivial acceptance;
- `93.68%` useful-revision recall;
- `87.25%` useful-revision precision.

The original R0.4 short-rollout rule again fails catastrophically on fresh
seeds: H16 RMSE is `0.99785`, including seed-level failures of `2.116` and
`1.876`. Passivity-only and passivity + utility remain stable on every accepted
branch.

C0 false revision is `0%`. C1 exact drag recovery is `100%`, with mean
coefficient `-0.12010` versus truth `-0.12`.

The main hypothesis is supported: constraining the admissible dissipative
model class removes anti-damping failures without collapsing acceptance or
useful-revision recall. A secondary result is negative: the H2/H4/H8 utility
gate does not improve aggregate H16 over passivity-only (`0.23468` versus
`0.23449`). It rejects seven harmful but also six oracle-useful revisions.

The SAPIEN realism gate is therefore eligible to advance. RoboTwin assets were
not downloaded in this experiment.

## 1. Cohort separation and implementation audit

The initial `6001–6041` run is retained only as an implementation-audit cohort.
It exposed two semantic mismatches:

1. the parameter regularizer was anchored to a posterior already contaminated
   by structural residual, shrinking C1 drag from `-0.11985` to `-0.08178`;
2. the passivity path still contained the old V6 force-BF hard threshold,
   creating an unregistered third acceptance gate and reducing useful recall to
   `16.67%`.

Neither issue was a numeric-threshold failure. They contradicted the declared
R0.6 architecture. The implementation was corrected to:

- anchor weak parameter regularization to the pre-revision passive posterior;
- use proposal/selection as force evidence, followed by exactly two R0.6
  layers: physical feasibility and rollout utility.

The fixed implementation was smoke-tested on seed 6001, then frozen. The sole
confirmatory cohort is:

```text
7001, 7011, 7021, 7031, 7041
```

No threshold, utility weight, operator annotation, or prior weight was changed
after observing these seeds.

## 2. Method

### Layer 1: physical admissibility

The following frozen operator families are declared dissipative over the
current positive-q validity support:

```text
xv, abs_v_v, position_damping, velocity_coupled
```

For these families, the structural coefficient is fitted in the constrained
space:

\[
r_{\tau,M}(q,\dot q,0)\dot q\le0.
\]

For the current bases this is equivalent to:

\[
\alpha\le0.
\]

If the unconstrained optimum has positive alpha, constrained MAP fitting places
it on the `alpha=0` boundary and refits theta. Position-only/potential-like
operators are not mislabeled as dissipative; in the native-dissipation R0-N
scope they are outside the admissible revision family rather than being forced
through the power inequality.

### Weak parameter-posterior regularization

Parameter refit is retained with fixed weight `lambda_theta=0.05`:

\[
L_{fit}
+0.05(\theta'-\theta_{passive})^\top
\Sigma_{passive}^{-1}
(\theta'-\theta_{passive}).
\]

The anchor uses only pre-revision passive evidence, avoiding the parameter
compensation already present after structural discovery.

### Layer 2: predictive utility

The candidate is compared with the unresolved fallback on independent short
branches:

\[
U(M')
=\frac13\sum_{h\in\{2,4,8\}}
\left[L_h(M_{old})-L_h(M')\right].
\]

Acceptance requires `U > 0`. Passivity is not added to this score; it has
already defined the feasible model space. Mahalanobis displacement, effective
damping, and overdamping are diagnostics only.

## 3. Baselines

Five baselines share proposals, selected operators, observations, noise, and
independent final rollouts:

1. `no_revision`: unresolved residual fallback;
2. `original_short`: frozen R0.4 force + H2/H4/H8 acceptance;
3. `passivity_only`: every triggered candidate in the admissible constrained
   family;
4. `passivity_utility`: constrained candidate with positive multiscale utility;
5. `oracle_safe_useful`: constrained candidate iff independent H16 is stable
   and improves over no revision.

The oracle is used only for recall/precision measurement.

## 4. Confirmatory gate summary

| Gate | Result | Status |
|---|---:|---|
| C0 false revision | `0% <= 1%` point estimate | Pass |
| C1 exact recovery | `100% >= 95%` | Pass |
| Native H16 gain | `+0.09365 >= 0` | Pass |
| Stability among accepted | `100% >= 99%` | Pass |
| Native acceptance | `85% > 0` | Pass |
| Useful-revision recall | `93.68% >= 80%` | Pass |
| Passivity + utility <= original short H16 | `0.23468 <= 0.99785` | Pass |

Thus `overall_go=true` under the frozen R0.6 gates.

Zero observed C0 failures over 120 episodes remains a point estimate, not a
population proof of `<1%` false revision.

## 5. H16 results

| Regime | No revision | Original short | Passivity only | Passivity + utility | Oracle |
|---|---:|---:|---:|---:|---:|
| C0 | `0.01123` | `0.01123` | `0.01123` | `0.01123` | `0.01123` |
| C1 | `0.22381` | `0.00519` | `0.00518` | `0.00518` | `0.00518` |
| R0-N | `0.32833` | `0.99785` | `0.23449` | `0.23468` | `0.21826` |

Passivity + utility improves native H16 by `0.09365`, or `28.52%`, relative to
no revision. Every confirmatory seed improves individually:

| Seed | No revision | Original short | Passivity + utility |
|---:|---:|---:|---:|
| 7001 | `0.3285` | `0.3183` | `0.2510` |
| 7011 | `0.3383` | `0.3327` | `0.2449` |
| 7021 | `0.3430` | `2.1160` | `0.2313` |
| 7031 | `0.2877` | `1.8765` | `0.2245` |
| 7041 | `0.3442` | `0.3458` | `0.2216` |

The failure is therefore not removed by a favorable seed average. The physical
feasibility constraint specifically prevents the catastrophic branches that
survive original short validation.

## 6. Stability and nontriviality

Native acceptance rates are:

| Strategy | Acceptance rate | Accepted H16 stability |
|---|---:|---:|
| Original short | `12.5%` | not used as a gate; aggregate instability remains |
| Passivity only | `95.83%` | `100%` |
| Passivity + utility | `85.00%` | `100%` |
| Oracle safe/useful | `79.17%` | `100%` by definition |

The method does not achieve safety by rejecting everything. It accepts 102 of
120 native revisions while eliminating all observed accepted-branch
instability.

## 7. Useful-revision retention

The oracle labels 95 constrained native revisions safe and H16-beneficial.
Passivity + utility accepts 89 of them:

\[
\text{recall}=89/95=93.68\%.
\]

It accepts 102 revisions total:

\[
\text{precision}=89/102=87.25\%.
\]

There are 13 accepted revisions with non-positive independent H16 gain and six
missed oracle-useful revisions. Thus the utility gate is high-recall but not an
oracle approximation.

By operator:

- `position_damping`: 77 accepted, 67 oracle-useful accepted;
- `abs_v_v`: 23 accepted, 20 oracle-useful accepted; every positive optimum is
  projected to the passive boundary;
- `velocity_coupled`: 1/1 useful accepted;
- `xv`: 1/1 useful accepted.

## 8. Negative secondary result: utility does not beat passivity-only

Passivity-only H16 is `0.23449`; passivity + utility is `0.23468`. The utility
layer's incremental aggregate gain is therefore `-0.00018`.

It rejects 13 passivity-feasible candidates:

- seven are not H16-beneficial;
- six are oracle-useful.

Meanwhile, 13 accepted candidates have positive H2/H4/H8 utility but
non-positive H16 gain. Their mean H16 gain is `-0.0898`.

This repeats the horizon-transfer boundary in a milder form:

\[
\boxed{
U_{H2,H4,H8}>0
\not\Rightarrow
\Delta L_{H16}>0
}
\]

The crucial difference from R0.4 is that these false-positive utility decisions
remain physically stable. Passivity has separated catastrophic safety failure
from ordinary model-selection error.

## 9. Overdamping and parameter diagnostics

Accepted native revisions have:

- mean structural overdamping ratio: `1.98` relative to declared physical
  damping;
- mean parameter Mahalanobis displacement: `1484.1`.

Neither is gated. The large values show that R0.6 has not solved calibration or
optimal damping magnitude. Yet unlike R0.4, these deviations remain stable and
aggregate utility is positive. Future work may improve precision, but should
not retroactively tune a damping or Mahalanobis threshold on this cohort.

## 10. Scientific conclusion and realism decision

R0.6 supports the primary hypothesis:

\[
\boxed{
\text{passivity-constrained revision space}
\Rightarrow
\text{anti-damping failures removed while useful revisions are retained}
}
\]

and supports the secondary comparison:

\[
\boxed{
\text{passivity + utility}
>
\text{short-rollout acceptance alone}
}
\]

It does not support:

\[
\boxed{
\text{short multiscale utility}
>
\text{passivity-only in aggregate H16 accuracy}
}
\]

The mechanism now satisfies the declared preconditions for advancing beyond
the lightweight SAPIEN bridge: C0 protection, C1 exact dissipative recovery,
native positive H16 utility, `>99%` accepted stability, nonzero acceptance, and
high useful-revision recall all pass on fresh seeds.

This makes a RoboTwin task experiment justified, not already validated.
R0.6 still uses oracle state, a one-DOF constraint-free hinge support, and known
dissipative-family annotations. No claim is made yet for contacts, conservative
revision families, noisy state, or vision.

## 11. Reproduction

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r06 \
  --output runs/v6r06/confirmatory \
  --device cpu \
  --seeds 7001 7011 7021 7031 7041 \
  --episodes 24
```

RoboTwin assets were not downloaded. The complete regression suite passes:
`76 passed`.
