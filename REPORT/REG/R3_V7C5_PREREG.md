# R3-V7C.5 Preregistration — Belief Integration of \(Z_{NSG}\)

Date: 2026-08-16  
Status: **FROZEN** (belief integration; encoder architecture frozen; no V7D)  
Depends on: `REPORT/REP/R3_V7C4_REPORT.md` (`verdict=SUFFICIENT`)  
Does not change: V7A–V7C.4 results, V7C.2 FIELD, B5-S recipe, \(y\,w_t\), GRU32 hidden  
Locks: **V7D**, RGB, encoder search, fusion studies, revision, \(C,V\)

## Placement

\[
\boxed{
V7C.4\ (\texttt{SUFFICIENT})
\rightarrow
R3\text{-V7C.5 (this stage)}
\qquad
V7D\text{ locked}
}
\]

Stage name:

\[
\boxed{
R3\text{-V7C.5 — Belief Integration of }Z_{NSG}
}
\]

中文名：**把已保留的接触斑几何表示接入证据正当化信念**。

## Frozen reading of V7C.4

\[
\boxed{\text{representation sufficiency: PASS}}
\]

does **not** imply

\[
\boxed{\text{epistemic usefulness: PASS}}
\]

C.4 showed \(Z_{NSG}\) is not an information bottleneck relative to
\(X_{NSG}\) on a linear/\(e_t\) probe. It did **not** show that \(Z\)
improves \(p_t\). Shuffle-null on that probe remains a warning:
retained probe information need not be time-locked contact evidence.

This stage does **not** reopen \(X\to Z\). Encoder **architecture** is
frozen as V7C.4’s 2-channel CNN + geometry MLP \(\to z\in\mathbb{R}^{2}\).
No architecture search.

## Question

\[
\boxed{
\text{Can }Z_{NSG}\text{ improve evidence-warranted belief }p_t
\text{ when added to frozen B5-S, via time-locked contact evidence?}
}
\]

Causal chain under test (only the last arrow):

\[
h_t^{S}+Z_{NSG,t}\rightarrow p_t
\]

## Design

- Fresh seeds \(\{26101,26111\}\) train, \(26121\) val,
  \(\{26131,26141\}\) held-out.
- Same B5 GRU32, Door mixture, matched C0/C1, target \(e_t=y\,w_t\),
  manual SGD. Runtime never sees \(w_t\).
- **B5-S:** frozen 12-D proprioceptive-contact recipe (V7C).
- **B5-S+\(Z\):** concatenate \(z_t\in\mathbb{R}^{2}\) to the B5-S
  step; encoder trained **only** with the warranted belief objective
  (no reconstruction loss; no encoder search).
- **B5-S+\(Z_{\mathrm{shuf}}\):** same architecture; \(z\) time-shuffled
  **inside each episode** at train and eval, B5-S steps unshuffled.

\(\delta=0.02\), C0 FPR \(\le 0.20\), \(\varepsilon=0.005\) as in V7C /
V7C.3. New temporal slack:

\[
\varepsilon_{\mathrm{temporal}}=0.005.
\]

\[
\Delta_{\mathrm{real}}=L(S)-L(S{+}Z),\qquad
\Delta_{\mathrm{shuf}}=L(S)-L(S{+}Z_{\mathrm{shuf}}),
\]

with \(L\) = held-out BCE vs \(e_t\).

## Gates (GO)

**H1.** Mid/final Brier vs \(y\):
\(\mathrm{Brier}^{S{+}Z}\le\mathrm{Brier}^{S}+\delta\).

**H2.** \(P(p_t>\tau_{\mathrm{dev}}\mid C0)\le 0.20\) for \(S{+}Z\),
\(\tau_{\mathrm{dev}}\) from development C0 finals of \(S{+}Z\).

**H3.** Held-out C1: mean Spearman\((p_t,w_t)>0\) and
\(\mathbb E[p\mid w_{\mathrm{high}}]>\mathbb E[p\mid w_{\mathrm{low}}]\)
for \(S{+}Z\).

**H4.** \(S{+}Z\) is useful vs B5-S by the V7C metric-dependent rule
(`h4_sensor_useful` with \(S{+}Z\) vs \(S\)).

**H5 (hard temporal gate).**

\[
\boxed{
\Delta_{\mathrm{real}}-\Delta_{\mathrm{shuf}}
>\varepsilon_{\mathrm{temporal}}
}
\]

\[
\texttt{V7C.5\_GO}=H1\land H2\land H3\land H4\land H5.
\]

If H4 holds and H5 fails: Brier may improve from a **static contact
signature**; that is **not** epistemic usefulness of time-varying
patch geometry. GO is false. Do not open V7D.

If H5 holds and H4 fails: time-locked \(e_t\) increment did not
translate into V7C belief metrics vs B5-S. GO is false.

Success of this stage does **not** unlock V7D or RGB.

## What this does not authorize

Encoder search, replacing B5-S with tactile-only slots (that was
V7C.1), rewriting V7C.2 FIELD, V7D, RGB, occupancy-penalty / GRU-gate
changes.
