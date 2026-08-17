# R1-RS1C Report — Layered Epistemic–Physical Policy

Date: 2026-08-16  
Prereg: `REPORT/REG/R1/R1_RS1C_PREREG.md`  
Artifacts: `runs/r1_rs1c/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `5b35dc6856f2e671e76b42facd23f6525746d855d7c201370947be217fc4a56b`

## Decision

\[
\boxed{RS1C\_GO = \mathrm{true}}
\]

\[
\boxed{
\text{Layered allocation + physical admissibility + short utility, with
support used only as monitoring, improves H32 vs none and vs a support veto.}
}
\]

This does **not** rewrite `RS1B_GO` (`false` remains on the original protocol).
It does **not** install a support-exit hard filter. RS2 prereg may now be
*drafted*; RS2 is not implemented here.

Smoke remains plumbing only. This 100-episode held-out matrix is the first
run allowed to set `RS1C_GO`.

## Scientific question (unchanged)

\[
\boxed{
\text{Does a layered policy — allocate epistemic effort, then install only
if physically admissible and short-horizon useful, while using support
exit only to raise monitoring — improve long-horizon prediction without
a support veto?}
}
\]

## Matrix

New seeds \(\{10101,10111,10121,10131,10141\}\),
\(\alpha\in\{0,-0.09,-0.12,-0.18,-0.24\}\),
\(A/A_0\in\{0.5,1.0,1.5,2.0\}\): **100** episodes.

Frozen RS1A.5 intake: \(C_{\mathrm{tol}}=0.001934\), \(A^\star=1.5A_0\),
\(\lambda=0.0015\). H32 queries are the RS1B *intervention* sampler.

## Allocation

| Stage | n |
|-------|--:|
| Initial tolerate / probe / revise-worthy | 56 / 26 / 18 |
| VoI extra \(A^\star\) acquired | 22 |
| Promoted probe \(\to\) revise-worthy | **18** |
| Final tolerate / probe / revise-worthy | 56 / 8 / 36 |

Remaining deferred probes are all \(\alpha=-0.12\) (8 cells). No install on
\(\mathcal T\) (0 / 56). No install on \(\alpha\in\{0,-0.09,-0.12\}\).

## Install

| Quantity | Value |
|----------|------:|
| Installed \(\pi_{\mathrm{RS1C}}\) | **30** |
| \(\pi_{\mathrm{always}}\) (skip passivity/utility) | 36 |
| \(\pi_{\mathrm{veto}}\) (drop if \(I_{\mathrm{exit}}>0\)) | 14 |
| Passivity violations among installs | **0** |
| Operator among installs | `abs_v_v` (30/30) |
| Rejected revise-worthy | 6 (seed 10121; `x2`; negative short utility + passivity) |

Accept was frozen before blind H32. Support did not enter the accept bit
(`accepted_flipped_by_support=0`).

## Monitoring (not veto)

| Installed subset | n | `monitor_intensity` |
|------------------|---:|---------------------|
| \(I_{\mathrm{exit}}=0\) | 14 | `normal` |
| \(I_{\mathrm{exit}}>0\) | 16 | `elevated` |

Monitor consistency: **pass**. Revisions with exit were **retained**.

Diagnostic (not a gate): \(P(\mathrm{harmful}\mid\mathrm{install},I_{\mathrm{exit}}>0)=0\)
on 128 queries. One harmful query occurred on an *installed interior*
episode (\(I_{\mathrm{exit}}=0\)); it does not license a support veto.

## Primary endpoints

| Gate | Value | Pass |
|------|-------|:----:|
| Nontrivial installs | 30 | ✓ |
| Passivity among installs | 0 | ✓ |
| False install on \(\mathcal T\) | 0 / 56 | ✓ |
| Seed-mean H32 gain vs \(\pi_{\mathrm{none}}\) | **+0.00160** | ✓ |
| Seed-mean gain vs \(\pi_{\mathrm{veto}}\) | \(0.00160 \ge 0.00052\) (\(\Delta=+0.00108\)) | ✓ |
| Monitor consistency | no accept flip; exit \(\Rightarrow\) elevated | ✓ |

Seed-mean \(\pi_{\mathrm{RS1C}}\) 95% CI is \([-4.4\times10^{-5},\,0.00325]\).
The frozen GO uses mean \(>0\), not CI exclusion of zero.

\(\pi_{\mathrm{always}}\) seed-mean \(0.00161\) is only marginally above
\(\pi_{\mathrm{RS1C}}\); the six passivity rejects were not a large H32
pool. The veto arm *does* lose utility, matching RS1B.1/B.2.

## What this does not authorize

- rewriting `RS1B_GO`;
- treating binary modeled-support retention as a restored RS1B gate;
- opening RS1B.3 / retuning \(D_{\mathrm{exit}}\);
- installing \(I_{\mathrm{exit}}>0\Rightarrow\mathrm{reject}\);
- running RS2 formal before the preregistered contact closure and adapter
  manifest are frozen.

## Unlock status

```text
RS1B_GO      false   (unchanged)
RS1B.1_GO    false
RS1B.2_GO    true    (support-risk branch closed)
RS1C_GO      true
RS2          prereg frozen; C0 implementation pending; formal locked
hard filter  not installed
```
