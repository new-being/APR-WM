# APR-WM V6R0.5P: Operator-controlled Passivity Falsification

## Executive conclusion

R0.5P removes the operator-identity confound and supports a narrower, more
mechanistic conclusion:

\[
\boxed{
\text{within }r_\tau=\alpha|\dot q|\dot q,
\quad
\alpha>0
\text{ strongly predicts H16 instability}
}
\]

It does **not** support the stronger claim that dissipativity guarantees
positive rollout utility.

With the same `abs_v_v` operator, symmetric coefficient magnitudes, identical
states/actions/noise/posterior, and five new seeds:

| Parameter handling | Dissipative H16 instability | Active H16 instability | Power-violation AUROC for instability |
|---|---:|---:|---:|
| Frozen theta | `0/360` | `345/360 = 95.83%` | `0.990` |
| Conditional theta refit | `0/360` | `316/360 = 87.78%` | `0.967` |

The effect survives the exact R0.4 failure condition. After conditional refit,
76 active and 80 dissipative candidates pass all H2/H4/H8 checks. At H16:

- active short-accepted revisions: `44/76 = 57.89%` unstable;
- dissipative short-accepted revisions: `0/80 = 0%` unstable.

Among 37 exact positive/negative coefficient pairs for which both members pass
the short gate, the median H8-gain gap is only `0.0224`; nevertheless, 20 pairs
have an unstable active member and a stable dissipative member.

This is strong operator-controlled evidence for a passivity/stability mechanism.
R0.6 is still not implemented: excessive negative damping remains stable but is
often inaccurate, so structure preservation is a necessary safety restriction,
not an acceptance or utility criterion.

## 1. Preregistered scope

The tested hypothesis is deliberately limited to a dissipative operator family:

\[
H_P:
\text{For dissipative residual families, zero-input positive power predicts
long-horizon unsafe revision.}
\]

No claim is made for conservative operators such as missing spring/potential
terms. For those models, temporary positive `r_tau * qdot` may represent valid
potential-to-kinetic energy exchange.

The experiment does not train an acceptance scorer, combine diagnostics, or
fit any threshold from H16 labels.

## 2. Controlled design

The operator is fixed to:

\[
r_\tau=\alpha|\dot q|\dot q.
\]

The symmetric coefficient grid is fixed before evaluation:

\[
\alpha\in
\{-0.45,-0.30,-0.15,+0.15,+0.30,+0.45\}.
\]

Five seeds unused by R0.3–R0.5 are evaluated:

```text
5001, 5011, 5021, 5031, 5041
```

Each seed contains 24 native SAPIEN episodes. Every positive/negative pair
shares:

- episode and physical parameters;
- force observations and noise realization;
- initial parameter posterior;
- validation states and sample count;
- short and H16 rollout branches;
- coefficient magnitude and model dimension.

Two parameter treatments are reported:

1. `fixed_theta`: freeze the same base posterior mean and change only the sign
   of alpha. This is the cleanest energy-direction intervention.
2. `refit_theta`: conditionally refit the same two parameter-tangent
   coefficients for each fixed alpha on identical force data. This tests
   whether parameter compensation removes the passivity effect.

There are 360 dissipative and 360 active candidates per treatment, or 720
exact symmetric pairs across both treatments.

## 3. Power metric and noise floor

Zero-input structural power is:

\[
P_r(t)=r_{\tau,M}(q_t,\dot q_t,0)\dot q_t
=\alpha|\dot q_t|^3.
\]

The violation threshold is fixed solely from the declared C0 generalized-force
noise floor and preregistered velocity support:

\[
\epsilon_{noise}
=3\sigma_{force}v_{max}
=3(0.002)(1.4)
=0.0084.
\]

The primary diagnostic is:

\[
V_P=\frac1T\sum_t\mathbf 1[P_r(t)>0.0084].
\]

`P_max` is also recorded. Jacobian spectral radius, negative effective damping,
Mahalanobis displacement, force gain, and short-rollout gain are secondary
diagnostics only.

## 4. Direction-level outcomes

| Parameter handling | Direction | N | H16 negative utility | H16 unstable | Combined unsafe | Mean H16 gain |
|---|---|---:|---:|---:|---:|---:|
| Fixed theta | Dissipative | 360 | `88.89%` | `0%` | `88.89%` | `-0.164` |
| Fixed theta | Active | 360 | `99.44%` | `95.83%` | `99.44%` | `-12.848` |
| Refit theta | Dissipative | 360 | `77.22%` | `0%` | `77.22%` | `-0.096` |
| Refit theta | Active | 360 | `93.33%` | `87.78%` | `95.00%` | `-9.188` |

The active/dissipative separation is absolute for stability: no negative-alpha
candidate becomes unstable at H16. It remains strong after theta is allowed to
compensate.

The large negative-utility rate for dissipative candidates is equally
important. The coefficient grid is a causal sign/magnitude intervention, not a
utility-optimized fit; excessive dissipation can remain perfectly stable while
predicting the native simulator poorly. Therefore:

\[
\boxed{
\text{dissipative}
\not\Rightarrow
\text{accurate or useful}
}
\]

## 5. AUROC decomposition

### Frozen theta

| Diagnostic | Combined unsafe | H16 instability | Negative utility |
|---|---:|---:|---:|
| Power violation fraction | `0.749` | `0.990` | `0.749` |
| Maximum power | `0.703` | `0.996` | `0.703` |
| Jacobian spectral radius | `0.757` | `0.997` | `0.757` |
| Negative damping violation | `0.757` | `0.997` | `0.757` |
| Lower short-rollout gain | `0.850` | `0.933` | `0.850` |
| Lower force-validation gain | `0.446` | `0.225` | `0.446` |

### Conditional theta refit

| Diagnostic | Combined unsafe | H16 instability | Negative utility |
|---|---:|---:|---:|
| Power violation fraction | `0.707` | `0.967` | `0.680` |
| Maximum power | `0.701` | `0.990` | `0.683` |
| Jacobian spectral radius | `0.750` | `0.993` | `0.730` |
| Negative damping violation | `0.757` | `0.991` | `0.738` |
| Lower short-rollout gain | `0.765` | `0.815` | `0.775` |
| Lower force-validation gain | `0.352` | `0.077` | `0.377` |

Power-violation AUROC for instability is stable across every new seed:

- fixed theta: `0.978, 0.995, 0.992, 0.992, 0.996`;
- refit theta: `0.967, 0.956, 0.979, 0.972, 0.965`.

This resolves the R0.5 operator confound for the fixed dissipative family.
Jacobian remains highly predictive but adds no demonstrated independent value
yet; it stays secondary as preregistered.

## 6. Short-horizon matched analysis

Short acceptance still requires positive gain over both physics and fallback at
H2/H4/H8 plus the frozen short-window stability test.

| Parameter handling | Direction | Short accepted | H16 unstable | H16 negative utility |
|---|---|---:|---:|---:|
| Fixed theta | Dissipative | 36 | `0%` | `72.22%` |
| Fixed theta | Active | 11 | `72.73%` | `90.91%` |
| Refit theta | Dissipative | 80 | `0%` | `62.50%` |
| Refit theta | Active | 76 | `57.89%` | `76.32%` |

Conditional refit produces 37 exact sign-pairs in which both candidates pass
the short gate. Their mean/median absolute H8-gain gaps are `0.0331/0.0224`.
Within these closely matched pairs:

- 20 have active H16 instability with a stable dissipative counterpart;
- 15 have an active combined-unsafe outcome with a safe dissipative
  counterpart;
- the reverse active-stable/dissipative-unstable stability pattern never
  occurs, because every dissipative candidate is stable.

Thus the passivity effect is not explained only by active candidates failing
the short gate. Parameter refitting can make active candidates locally useful,
but it does not remove their long-horizon anti-damping instability.

## 7. Hypothesis decision

The preregistered wording used “H16 unsafe,” which combines negative utility and
instability. On that combined label, R0.5P gives above-chance but moderate
ranking (`0.749` fixed, `0.707` refit) because many overly dissipative candidates
are stable yet inaccurate.

The proposed physical mechanism is specifically about energy injection and
long-horizon instability. That mechanism is strongly supported:

\[
\boxed{
\text{zero-input positive power}
\Rightarrow
\text{high risk of H16 instability}
}
\]

within the controlled `abs_v_v` family and tested support.

The correct interpretation is therefore:

1. zero-input dissipativity is a strong structure-preserving **stability
   condition** for this dissipative operator family;
2. it is not a sufficient condition for model accuracy, revision utility, or
   acceptance;
3. it must not be imposed on conservative operator families without an energy
   storage model;
4. R0.6 must still separate “physically admissible parameterization” from
   “evidence that this admissible revision improves prediction.”

No R0.6 method or threshold has been created.

## 8. Reproduction

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r05p \
  --output runs/v6r05p/formal \
  --device cpu \
  --seeds 5001 5011 5021 5031 5041 \
  --episodes 24
```

RoboTwin assets were not downloaded. The complete regression suite passes:
`73 passed`.
