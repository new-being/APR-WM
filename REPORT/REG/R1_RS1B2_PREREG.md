# R1-RS1B.2 Preregistration — Support-Band Identification

Date: 2026-08-16  
Status: **FROZEN; formal completed** — see `REPORT/REP/R1_RS1B2_REPORT.md` (`RS1B.2_GO=true`)  
Depends on: `REPORT/REP/R1_RS1B1_REPORT.md` (`RS1B.1_GO=false`; intermediate band empty)  
Frozen implementation: `aprwm_v0/r1_rs1b2.py`

## Placement

\[
\boxed{
RS1B.1 \rightarrow RS1B.2 \rightarrow \text{branch closed (no RS1B.3)} \rightarrow RS1C\ \mathrm{prereg} \rightarrow RS2\ (\mathrm{locked})
}
\]

RS1B.1 established:

\[
I_{\mathrm{exit}}>0\not\Rightarrow\mathrm{harmful}
\]

and a weak positive Spearman\((D_{\mathrm{exit}},\mathrm{RMSE}^{\mathrm{rev}})\), but
could not tell continuous risk from a binary regime jump because

\[
p\bigl(D_{\mathrm{exit}}\in\text{intermediate}\bigr)\approx 0.
\]

This stage does **not** lengthen H32, retune \(d_{\mathcal S}\), add a support
hard filter, or reopen detector / VoI / passivity / \(\mathcal P_{\mathrm{rev}}\).

## Scientific question

\[
\boxed{
\text{Can targeted queries populate intermediate support-departure bands
well enough to distinguish continuous risk from a genuine regime jump?}
}
\]

Not: “is a longer horizon more stable?”  
Not: “should we reject on \(I_{\mathrm{exit}}>0\)?”  
(That reject rule would have discarded 8/8 positive-gain queries in RS1B.1.)

## Frozen (do not touch)

- RS1A.5 intake policy and \(\mathcal P_{\mathrm{rev}}\)  
- Frozen R0.6 revision / passivity / H2–H8 accept; accept still frozen before H32  
- \(d_{\mathcal S}\) formula from RS1B.1 (`modeled_distance`)  
- `harmful` \(=\) (not finite) \(\lor\) (\(\Delta\mathrm{RMSE}<0\))  
- RS1B `stable_h32` definition (diagnostic only)

Only **H32 query initialization / exposure** changes.

## Frozen realized bins (a priori)

Bins are **not** cut from this run’s quantiles. They are fixed from the RS1B.1
empty-band diagnosis (\(D_{\mathrm{tol}}\approx0.04\), jump cluster \(\ge0.048\)):

| Bin | \(D_{\mathrm{exit}}\) |
|-----|------------------------|
| B0 supported | \(=0\) |
| B1 mild | \((0,0.04]\) |
| B2 mid | \((0.04,0.08]\) |
| B3 far | \(>0.08\) |

Occupancy uses **realized** \(D_{\mathrm{exit}}\), not the intended sampling label.

## Targeted query design

Door modeled interior: \([q_{\min}+m,q_{\max}-m]\), \(m=0.05\).

Per accepted episode, **16** H32 queries = 4 intended strata × 4 replicates
(lower/upper boundary alternating).

| Intended | Initial \(d_{\mathcal S}\) target | \(|v_0|\), \(|\tau|\) |
|----------|-----------------------------------|------------------------|
| B0 | deep interior (pad \(0.08(q_{\max}-q_{\min})\) from both modeled walls) | small |
| B1 | uniform in \((0,0.04]\) just outside the nearer wall | small |
| B2 | uniform in \((0.04,0.08]\) | small |
| B3 | uniform in \((0.08,0.16]\) | small |

Small excitation is frozen at amplitude \(0.02\) so H32 (\(\approx0.256\,\mathrm{s}\))
does not immediately wash initial distance into the far joint limit. This is
query exposure, not a change to the revision operator.

Smoke uses 4 queries (one per intended stratum).

## Matrix

New held-out seeds \(\{10001,10011,10021,10031,10041\}\).

Same \(\alpha\in\{-0.12,-0.18,-0.24\}\) and \(A\in\{1.5,2.0\}A_0\) as RS1B
(revision population unchanged): **30 episodes**.

Smoke: seed `9031` × \(\alpha=-0.24\) × \(1.5A_0\).

## Endpoints

### Identification (primary / GO)

Each realized bin B0–B3 has

\[
n_b \ge 12
\]

accepted queries. That is the only occupancy bar.

If B1 (mild) remains empty under this targeted sampler, that is evidence the
intermediate band is **dynamically unfillable** on this hinge, not an \(n\)
problem.

### Mechanism (reported; hard only if occupancy passes)

Among accepted queries with finite RMSE:

- **continuous candidate:** Spearman\((D_{\mathrm{exit}},\mathrm{RMSE}^{\mathrm{rev}}\mid D_{\mathrm{exit}}>0)>0\)
  and \(\overline{\mathrm{RMSE}}(B1)<\overline{\mathrm{RMSE}}(B3)\) (both \(n\ge12\)).
- **jump candidate:** occupancy holds, continuous candidate fails, and
  \(\overline{\mathrm{RMSE}}(B0)<\min_k\overline{\mathrm{RMSE}}(B_k)\) for nonempty
  exited bins \(k\in\{1,2,3\}\).

These classify; they do not rewrite RS1B.1 gates.

### Diagnostics (not GO)

- intended \(\rightarrow\) realized confusion  
- \(P(\mathrm{gain}>0\mid I_{\mathrm{exit}}>0)\) (expect still \(\not\Rightarrow\) harm)  
- never install support-exit veto

## GO

\[
RS1B.2\_GO
=
\bigwedge_{b\in\{B0,B1,B2,B3\}}
(n_b\ge 12).
\]

On GO: report continuous vs jump; **close** the support-risk branch (no
RS1B.3). Still **no** support hard filter and **no** RS2 unlock. RS1C may
be drafted as a *new* policy hypothesis (`REPORT/REG/R1_RS1C_PREREG.md`),
not as a repair of `RS1B_GO`.

On fail: freeze “binary support event may be more natural than Euclidean
\(d_{\mathcal S}\) on this Door”; do not keep retuning the distance formula.
