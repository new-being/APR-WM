# R3-V7 Evidence Chain and Causal Boundaries

Date: 2026-08-17  
Status: **FROZEN** (Door / R3-V7 closed; no V7E; R4-C0 `GO=false`)  
Does not change: any RS* GO, C0, \(s^\star\), RS1C, B5-S, \(w_t\), GRU32  
Does not open: visuo-tactile fusion, encoder search, **R3-V7E**, Door capacity shopping  
Next domain: **R5-D0-preflight `PASS=true`** (I1 GO; I0 scalar C still false)
(`REPORT/REP/R5_D0_PREFLIGHT_REPORT.md`); no planner / encoder yet

Companion visual: Cursor canvas `r3-v7-evidence-chain.canvas.tsx`.

## What this freeze is

A map of **which hypotheses were tested**, **which survived**, and
**which experimental knobs are now scientifically closed** on the
current Door mixture. It exists so later work cannot reopen a closed
branch by changing architecture details.

## Evidence chain (verified / failed)

| Stage | Hypothesis | Verdict |
|---|---|---|
| V7A | Context \(h\) calibrates \(p_{\mathrm{struct}}\) in-domain | **PASS** |
| V7B | Persistent history improves calibration vs static \(\tilde h\) | **PASS** on mid/final; **FAIL** C0 occupancy |
| V7B.1 | IBS / objective-only fix solves occupancy | **FAIL** |
| V7B.2 | Fast/slow state solves occupancy | **FAIL** |
| V7B.3 | Evidence-warranted \(e_t=y w_t\) solves occupancy + tracks \(w\) | **PASS** |
| V7C | Noisy proprioceptive contact can replace oracle contact | **PASS** (metric-dependent H4) |
| V7C.1 | Learned tactile \(z\) is a useful epistemic channel vs B5-N/S | **FAIL** |
| V7C.2 | Loss locus is FIELD / encoder / fusion | **FIELD** (old 8×8; seed-unstable \(\Delta_N\) retained) |
| V7C.3 | Some tactile \(X\) has \(I(e_t;X\mid h^{-c})>0\) | **PASS** for \(X_{NSG}\); shear not the nested carrier |
| V7C.4 | \(Z_{NSG}\) preserves that increment (\(\Delta_Z\approx\Delta_X\)) | **SUFFICIENT** (shuffle-null diagnostic) |
| V7C.5 | \(Z_{NSG}\) improves \(p_t\) given B5-S, time-locked | **FAIL** (redundant/harmful; H5 fails) |
| V7D | \(I(Z_{\mathrm{RGB}};e_t\mid h^{S})>0\) with shuffle gate | **FAIL** (redundant/harmful; H5 fails) |
| V7E | Door capacity / extra fusion | **does not exist** |

Strongest surviving scientific claim:

\[
\boxed{
\text{persistent contextual epistemic belief is feasible without oracle
contact, if trained on evidence-warranted temporal supervision}
}
\]

Modality-selection claim (C.5 + V7D):

\[
\boxed{
\text{representation sufficiency}\neq\text{conditional usefulness}
}
\]

\[
\boxed{
I(Z;e_t\mid h^{-c})>0
\not\Rightarrow
I(Z;e_t\mid h^{S})>0
}
\]

\[
\boxed{
I(Z;e)>0
\notRightarrow
I(Z;e\mid h)>0
}
\]

\[
\boxed{
\text{modality availability}
\neq
\text{conditional modality value}
}
\]

Door saturation:

\[
\boxed{
h^{S}\ \text{already contains Door's current structural-error information}
}
\]

\[
\boxed{
\text{additional modality can be actively harmful when redundant
with the existing belief history}
}
\]

Belief update, if later mechanized, is not “use all sensors”:

\[
\boxed{
\text{belief update}
=
\text{evidence accumulation}
+
\text{conditional information allocation}
}
\]

## Causal boundaries (do not reopen on this Door mixture)

Closed because C.5 located **conditional redundancy given B5-S**, not
an encoder defect:

- larger / temporal / attention CNN
- tactile–proprioception fusion search
- tactile gate, occupancy penalty, GRU-gate edits
- \(\tau\) or \(w_t\) refit to “make tactile look useful”
- rewriting V7C.2 FIELD for the old 8×8 map
- treating V7C.4 SUFFICIENT as “encoder is good for belief”
- RGB encoder / resolution / visuo-tactile fusion search after V7D FAIL
- **R3-V7E** / DINO / CLIP / larger GRU / feature-normalization search
  on this Door mixture

Still true, still frozen:

\[
\text{the world being wrong}
\neq
\text{the current history warranting belief that it is wrong}
\]

Warranted belief does not require perfect access to the causal
variable (V7C). A new sensor is not entered because it exists; it is
entered only if

\[
I(\text{new modality};e_t\mid \text{existing history})
\]

is large enough to pay fusion cost.

## Door tactile branch (closed)

```text
V7C proprioceptive contact          ✓ useful vs no-contact (metric-dependent)
        │
        ▼
V7C.1 learned tactile for belief    ✗
        │
        ├── C.2 FIELD (old 8×8)
        ├── C.3 patch geometry incremental given h^{-c}
        ├── C.4 Z_NSG representation-SUFFICIENT
        ▼
V7C.5 Z + B5-S                      ✗ conditionally redundant / harmful
V7D  Z_RGB + B5-S                   ✗ same class; aligned RGB worse than shuffle
```

## Door / R3-V7 is closed

There is **no V7E**. The limiting object is no longer representation
capacity. It is

\[
\boxed{\text{task/distribution does not require the additional modality}}
\]

## Next stage (chosen: former option A)

**R4 family STOP** (`REPORT/REP/R4_FAMILY_STOP.md`). No C3.
R5 prereg is admission only (`REPORT/REG/R5_PREREG.md`, not run):
continuous \(\lambda\mapsto(X,C)\). Next work is a physical mechanism,
not model code.
