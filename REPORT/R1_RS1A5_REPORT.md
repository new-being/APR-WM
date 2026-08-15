# R1-RS1A.5 Report — Value-of-Epistemic-Excitation

Date: 2026-08-15  
Prereg: `R1_RS1A5_PREREG.md`  
Artifacts: `runs/r1_rs1a5/formal/`

## Decision

\[
\boxed{RS1A.5\_GO = \mathrm{true}}
\]

\[
\boxed{
\text{Epistemic excitation can have positive VoI — but max amplitude is not optimal}
}
\]

RS1B remains **locked for formal run** until `R1_RS1B_PREREG.md` numbers are
frozen; **preregistration / development is unlocked**. Intake = \(\mathcal P_{\mathrm{rev}}\) only.


## Frozen policy (permanent)

\[
\begin{aligned}
\mathrm{tolerate} &= \{C<C_{\mathrm{tol}}\}\\
\mathrm{probe} &= \{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=0\}\\
\mathcal P_{\mathrm{rev}} &= \{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=1\}
\end{aligned}
\]

## Setup

- Only probe instances from \(\alpha\in\{-0.18,-0.24\}\), \(A_b\in\{0.5,1.0\}A_0\)
- \(c(A)=(A/A_0)^2\), \(\lambda=0.0015\)
- \(V=\mathrm{flip}\cdot C-\lambda c(A_a)\)
- \(n_{\mathrm{probe}}=20\), \(n_{\mathrm{VoI\ evals}}=50\)
- \(C_{\mathrm{tol}}=0.00193\)

## Hard gates

| Gate | Value | Pass |
|------|------:|:----:|
| \(\exists A_a:\overline V(A_a\mid 0.5A_0)>0\) | yes (\(A_a=1.5\)) | ✓ |
| \(A^\star\) well-defined and \(\overline V(A^\star)>0\) | \(A^\star=1.5A_0\), \(V=0.00120\) | ✓ |
| Is \(A^\star=\max A\)? | **No** (descriptive) | — |

## VoI table (seed-mean)

| \(A_b\) | \(A_a\) | flip rate | \(\overline V\) | \(\overline V_\Delta\) |
|------:|------:|----------:|----------------:|----------------------:|
| 0.5 | 1.0 | 0.00 | **−0.00150** | −0.00113 |
| 0.5 | **1.5** | **1.00** | **+0.00120** | +0.00157 |
| 0.5 | 2.0 | 1.00 | **−0.00143** | −0.00105 |
| 1.0 | 1.5 | 1.00 | +0.00120 | +0.00270 |
| 1.0 | 2.0 | 1.00 | −0.00143 | ≈0 |

## Scientific reading

1. **Probe → revise-worthy is achievable.**  
   From \(0.5A_0\), both \(1.5A_0\) and \(2A_0\) flip with rate 1.0 on this probe set.

2. **Cost matters.**  
   \(2A_0\) always flips but \(\lambda c(2)=0.006\) exceeds typical \(C\), so net VoI is **negative**.  
   \(1.5A_0\) flips equally often at lower cost → **unique positive optimum**.

3. **Useless mild boosts.**  
   \(0.5\to1.0\) never flips here: pay \(\lambda c\) for no decision change.

4. Therefore epistemic control is not “always excite harder”:

\[
\boxed{
\text{probe}
\rightarrow
\text{choose }A^\star\text{ with }V>0
\rightarrow
\begin{cases}
\text{revise-worthy},\\
\text{defer if all }V\le0
\end{cases}
}
\]

## Decision chain (now complete through VoI)

\[
\boxed{
\text{mismatch}
\rightarrow
\{\text{tolerate},\ \text{probe},\ \text{revise-worthy}\}
\]

\[
\boxed{
\text{probe}
\rightarrow
\text{VoI}(a)
\rightarrow
\{\text{acquire evidence},\ \text{defer}\}
\]

## Unlock status

| Stage | Status |
|-------|--------|
| RS1A.5 | **passed** |
| RS1B | **locked** (intake = \(\mathcal P_{\mathrm{rev}}\) only, when opened) |
| RS2 | locked |

## Next

**RS1A stage frozen** — see `R1_RS1A_STAGE_FREEZE.md`.

**RS1B unlocked for development** — `R1_RS1B_PREREG.md`:

> Among consequential and epistemically justified revisions, what dynamics-level
> conditions predict long-horizon stability?

Population: revise-worthy only. Do not reopen detector/VoI tuning after H32.

