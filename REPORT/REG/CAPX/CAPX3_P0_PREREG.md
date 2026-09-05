# CAP-X3-P0 Preregistration — Identifiability Preflight

Date: 2026-08-21  
Status: **FROZEN**; **PASS** (`cap_x3_p0_passed=true`); formal unlocked  
Depends on: `REPORT/REG/CAPX/CAPX3_PREREG.md`  
Does not: formal adaptation comparison; \(\rho>0\); visual; R10

## One question

\[
\boxed{
\text{Is the active }\theta_E\text{ identifiable from passive calibration
trajectories, and can a same-dim latent }z_E\text{ be optimized at all?}
}
\]

If P0 fails, **do not** run CAP-X3 formal. Allowed P0-only action:
shrink the active parameter set **before** seeing formal \(K\) curves.

## Active \(\theta_E\) (proposed freeze, confirmed by P0)

Host vector is \(\mathbb R^{13}\) including Coulomb slots. CAP-X0 froze
\(\mu_i=0\). Those three coordinates are **not** in the adaptation set
(identically zero; not a degree of freedom).

\[
\theta_E^{\mathrm{active}}
=
(s_m^{(1:3)},\,s_I^{(1:3)},\,b^{(1:3)},\,m_p)
\in\mathbb R^{10},
\qquad
d_z=10.
\]

P0 may drop a coordinate only if G-id-pred fails and a pre-declared
leave-one-group diagnostic shows that group is sloppy. Groups:
`mass_scale` (3), `inertia_scale` (3), `damping` (3), `payload` (1).
Do not invent new physical parameters.

## Data (P0 only; not the formal 128/32/64)

| role | \(n\) | scene seed0 | traj |
|---|---:|---:|---|
| latent meta-train | 32 | 100000 | 8 × \(1.0\,\mathrm{s}\) |
| ident / adapt probe | 8 | 140000 | 16 cal + 4 query × \(1.0\,\mathrm{s}\) |

Seeds disjoint from CAP-X0 \(\{70,80,90\}\times 10^3\). \(\rho=0\).
Same excitation family as X0 (multi-sine / chirp / bandlimited /
piecewise-smooth). Physics and latent **see identical** \(D_{\mathrm{cal}}\)
on probe scenes (\(K=16\)).

## Gates

| id | requirement |
|---|---|
| **G-active** | \(d_\theta=10\), \(\mu\) unused; `runs/cap_x3/p0/`; no X0 test reuse |
| **G-id-pred** | median query \(E_1\) after physics fit on \(K=16\) \(\le 0.05\) |
| **G-id-param** | median \(E_\theta=\|\hat\theta-\theta\|_2/\|\theta\|_2\le 0.35\) (**diagnostic**; FAIL here with G-id-pred PASS → log `identification_slack`, still P0 PASS) |
| **G-latent** | median query \(E_1(z_{\mathrm{fit}})\le 0.80\,E_1(z=0)\) (optimizer moves) |
| **G-rho** | no CAP-X2 \(\tau_\perp\) |
| **G-label** | no \(R_K\) in P0 summary as a claim |

`cap_x3_p0_passed` iff G-active \(\land\) G-id-pred \(\land\) G-latent
\(\land\) G-rho \(\land\) G-label.

If G-id-pred FAIL: STOP formal; optionally shrink one group and **re-run P0**
(new summary), never peek at formal \(K_{90}\).

## Claims ceiling

P0 may say: passive \(K=16\), \(T=1\,\mathrm{s}\) identifies (or does not)
the 10-D scene vector for prediction; latent embeddings are trainable.

P0 may **not** say: physics adapts more sample-efficiently than latent.

## Unlock

\[
\text{CAP-X3-P0 PASS}
\;\Rightarrow\;
\text{CAP-X3 formal }K\text{ grid may start}
\]
