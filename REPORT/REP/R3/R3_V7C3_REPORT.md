# R3-V7C.3 Report — Epistemically Informative Tactile Observation

Date: 2026-08-16  
Prereg: `REPORT/REG/R3/R3_V7C3_PREREG.md`  
Depends on: `REPORT/REP/R3/R3_V7C2_REPORT.md` (locus FIELD)  
Artifacts: `runs/r3_v7c3/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `8a0a4b8cb6c4801320f7e424e23c7a0a8858b031703004d820941ab31b466584`

## Decision (observation sufficiency, not a method GO)

\[
\boxed{\mathrm{stop\_tactile\_branch}=\mathrm{false}}
\]

\[
\boxed{
\max_j\Delta_j=\Delta_{\mathrm{NSG}}=0.057>\varepsilon=0.005
}
\]

This stage **did not** train a CNN, **did not** re-run frozen B5, and
**does not** open V7D. The predeclared stop rule (all candidates
conditionally redundant given \(h^{-c}\)) did **not** fire.

The nested ablation is **not** the hoped-for shear story:

\[
\Delta_N=0.041,\quad
\Delta_{NS}=0.007,\quad
\Delta_{NSG}=0.057,\quad
\Delta_{NSGM}=0.019.
\]

\[
\boxed{
\text{patch geometry, not shear, is the only nested step that
clearly improves on }X_N
}
\]

Do **not** treat “all four names are `sufficient`” as four equally
trustworthy observations. See the V7C.2 seed caveat below.

## Question (unchanged)

\[
\text{Can a tactile observation expose incremental warranted-evidence
information that is absent from }h^{-c}?
\]

Target \(e_t=y\,w_t\). Linear readout only.

## Design (frozen)

- Seeds \(\{24101,24111\}\) train, \(24121\) val, \(\{24131,24141\}\)
  held-out (new split; V7C.2 used \(231xx\)).
- Same GRU32 / 12-D / \(e_t\) BCE as V7C.2 P0/PX.
- \(X^{(j)}\) linearly mapped \(d_j\to 2\) into contact slots.
- \(\Delta_j=L(P_0)-L(P_j)\), \(\varepsilon=0.005\).
- Matched C0/C1 \(L_2\) is mechanism-only.

## Results

| \(j\) | \(L(P_j)\) | \(\Delta_j\) | \(\Delta_j>\varepsilon\) | late matched \(L_2\) |
|---|---|---|---|---|
| \(P_0\) (\(h^{-c}\)) | \(0.486\) | — | — | — |
| \(N\) | \(0.445\) | \(0.041\) | yes | \(5.43\) |
| \(NS\) | \(0.479\) | \(0.007\) | yes (fragile) | \(3.52\) |
| \(NSG\) | \(0.429\) | \(0.057\) | yes (**max**) | \(3.76\) |
| \(NSGM\) | \(0.467\) | \(0.019\) | yes | \(4.44\) |

Early matched \(L_2=0\) for every candidate. Late maps still move with
hidden dynamics. That remains **not** a substitute for \(\Delta_j\).

\[
\boxed{
\text{pairwise difference}\neq\text{incremental epistemic information}
}
\]

still holds: \(N\) has the largest late \(L_2\) but not the largest
\(\Delta_j\).

## How to read the nested steps

The predeclared diagnostic pattern was
\(\Delta_N\approx 0\) and \(\Delta_{NS}>0\)
\(\Rightarrow\) shear carries incremental information.

That pattern **did not occur**.

- Adding shear (\(N\to NS\)) **reduced** \(\Delta\) from \(0.041\) to
  \(0.007\). Same epoch budget, more dimensions: the linear readout
  did not extract a shear increment beyond \(h^{-c}\).
- Adding patch geometry (\(NS\to NSG\)) is the only nested increase
  (\(\Delta=0.057\)).
- Adding multi-surface / local moments (\(NSG\to NSGM\)) **reduced**
  \(\Delta\) again.

So this Door mixture does **not** support “shear is what \(h^{-c}\)
was missing.” It **does** support a weaker claim: a compact contact-patch
geometry vector, stacked with the pressure/shear fields, can expose
incremental \(e_t\) information in this linear-readout test.

## V7C.2 is not overturned

V7C.2 on seeds \(231xx\): \(\Delta_X=0.001\le\varepsilon\) for the
same 8×8 magnitude field. V7C.3 \(N\) on seeds \(241xx\):
\(\Delta_N=0.041\).

\[
\boxed{
\Delta_N\text{ is seed-unstable; V7C.2 FIELD locus for that map
is not rewritten}
}
\]

Reasons this is not a C.2 reversal:

1. Different held-out seeds by design.
2. Linear GRU+SGD with 60 epochs is a noisy estimator of
   incremental information.
3. \(N\) passing \(\varepsilon\) here is therefore **not** a license
   to reopen encoder search on the old 8×8 pressure map.

The observation to carry forward, if any, is **\(X_{NSG}\)** (largest
\(\Delta\), and the only nested improvement), under a **new**
representation-sufficiency prereg. Not \(N\). Not “stack everything.”

## What this does not authorize

- CNN / encoder architecture search in this stage
- stuffing any \(X^{(j)}\) into frozen B5
- V7D, RGB, fusion studies
- claiming shear or multi-surface as the informative channel
- expanding taxel resolution as the only change

## What follows

Stop rule: **not** triggered.

\[
\boxed{
\text{observation sufficiency (NSG)}
\rightarrow
\text{representation sufficiency (new prereg)}
\rightarrow
\text{epistemic integration (still later)}
}
\]

Until that prereg exists: **V7D locked**. Do not train
\(x^{\mathrm{tactile,new}}\to z^{\mathrm{tactile}}\) in an
unregistered follow-up.

If a later representation stage still cannot keep
\(\Delta_Z\approx\Delta_{NSG}\), that is an encoder problem on an
already-informative observation — the question V7C.2 showed was
**not** the right first question for the old 8×8 map.
