# R1-RS1A.3 Preregistration — Excitation-Limited Inadequacy Identifiability

Date: 2026-08-15  
Status: **FROZEN** — formal completed; see `R1_RS1A3_REPORT.md` (`RS1A.3_GO=true`; excitation-limited confirmed)

## Placement

\[
\boxed{
\begin{aligned}
RS1A &\colon \text{support channel repairs C2}\\
RS1A.1 &\colon \text{support cannot repair C1-L}\\
RS1A.2 &\colon \text{alternative residual statistics do not beat }D_0\\
RS1A.3 &\colon \textbf{is C1-L detectability excitation-limited?}
\end{aligned}
}
\]

**Frozen from prior stages (do not reopen):**

- Typed architecture: \(E_{\mathrm{known}}=\|r_\perp\|\), \(E_{\mathrm{unknown}}=\mathrm{support}\)
- Support branch offline (not optimized here)
- Detector family fixed to **\(D_0\)** only
- No revision / acceptance / passivity / H32
- RS1B locked
- No \(\gamma\) fusion; no new residual statistics

## Scientific question

\[
\boxed{
\text{Is the residual C1-L miss primarily excitation-limited
(trajectory never enters an informative operator region)?}
}
\]

Operational hypothesis:

\[
\boxed{
P(\mathrm{detect}\mid C1\text{-}L)
\text{ increases with operator exposure }
X_\phi=\tfrac1T\sum_t v_t^4
}
\]

where \(\phi(v)=|v|v\) and \(X_\phi=\mathbb E[\phi(v)^2]=\mathbb E[v^4]\).

## Why probe-trajectory \(D_0\) (not independent discovery sampler)

RS1A–A.2 computed \(D_0\) on an independent discovery sampler. That design cannot
test excitation. RS1A.3 therefore evaluates **frozen \(D_0\) on the excited probe
trajectory** after the same passive-context base fit:

```text
passive fit (unchanged)
    → excited sine probe (A, f)
    → r_⊥ along probe
    → D0 = RMS(r_⊥)
```

## Excitation grid

Base amplitude \(A_0=0.12\) (RS0-safe scale).

| Factor | Levels |
|--------|--------|
| Amplitude scale | \(\{0.5,1.0,1.5,2.0\}\times A_0\) |
| Frequency | \(\{0.20,0.40,0.80\}\) Hz sine, phase \(0\) |
| Regimes | C0, C1-L only |
| Seeds | \(\{9601,9611,9621,9631,9641\}\) |

\[
5\times 2\times 4\times 3 = \boxed{120\ \mathrm{trajectories}}
\]

Smoke: seed `8981` × {C0,C1-L} × \((A_0,0.40\,\mathrm{Hz})\).

## Exposure metrics (logged every episode)

\[
\max|v|,\;
\mathrm{RMS}(v),\;
\mathrm{RMS}(|v|v),\;
X_\phi=\mathrm{mean}(v^4)
\]

on the probe velocity series.

## Endpoints

Within each cell \((A,f)\):

1. Calibrate \(\tau\) on that cell’s C0 at FPR \(\le 1\%\)
2. Report C1-L recall at \(\tau\)
3. Report AUROC(C0, C1-L)

### Primary confirmatory claims

\[
\boxed{
\overline{R}(2A_0) > \overline{R}(0.5A_0)
}
\]

(seed-mean recall, averaging frequencies)

\[
\boxed{
\mathrm{Spearman}\bigl(X_\phi,\; \mathbf{1}[D_0>\tau_{\mathrm{cell}}]\bigr)_{\mathrm{C1\text{-}L}} > 0
}
\]

report coefficient and seed-level mean.

Optional descriptive: logistic fit \(P(\mathrm{detect})\approx\sigma(aX_\phi+b)\).

### Interpretation

| Result | Meaning | Next |
|--------|---------|------|
| Recall rises with \(A\) / \(X_\phi\) | excitation-limited | RS1A.4 detectability↔consequence; then active probe design |
| Flat in \(A\) / \(X_\phi\) | not excitation; revisit window or constrained match | RS1A.3b window length |
| C0 residuals also explode with \(A\) | excitation unsafe / closure stress | cap amplitude; do not claim detectability win |

## Explicit non-goals

- improving C2 (already solved by support channel)
- beating \(D_0\) with new scalars
- unlocking RS1B / RS2
- \(\alpha\)-consequence curves (reserved for **RS1A.4**)
