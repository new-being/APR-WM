# R3-V7C Preregistration — Sensorized Contact Epistemic Belief

Date: 2026-08-16  
Status: **FROZEN**  
Depends on: `REPORT/REP/R3_V7B3_REPORT.md` (`V7B.3_GO=true`)  
Does not change: V7A/B/B.1/B.2/B.3 GOs, RS* GOs, C0, \(s^\star\), RS1C policy  
Locks: **V7D**, RS5B, revision, VoI, \(C,V\), RGB, tactile images, Transformer,
occupancy penalties, GRU gate changes

## Placement

\[
\boxed{
V7B.3\text{ (evidence-warranted supervision)}
\rightarrow
R3\text{-V7C (this stage)}
\qquad
V7D\text{ locked}
}
\]

Frozen principle from V7B.3:

\[
\boxed{
\text{the world being wrong}
\neq
\text{the current history warranting belief that it is wrong}
}
\]

This stage does **not** change that supervision. It removes **one**
inference-time oracle: contact.

Stage name:

\[
\boxed{
R3\text{-V7C — Sensorized Contact Epistemic Belief}
}
\]

中文名：**传感化接触上的认识信念**。

## Question

\[
\boxed{
\text{Does evidence-warranted persistent belief survive when oracle contact
information is replaced by learner-visible noisy contact sensing?}
}
\]

The primary comparison is **B5-S vs B5-N**, not beating B5-O.

## Single manipulated variable

\[
\boxed{
\text{oracle contact}
\rightarrow
\text{learner-visible noisy contact sensing}
}
\]

Still frozen: Door mixture; oracle \(q,\dot q\) / world residual used
for \(\theta\) and for **train-only** \(w_t\); B5 single GRU32;
evidence-warranted targets; no phase/family input; no \(C,V\); no
revision; no RGB; no taxel/tactile image.

\(w_t\) remains a simulator **label generator**. Runtime never sees the
matched C0 tape, \(w_t\), phase, or family. This stage does **not**
answer how to obtain \(w_t\) on a real robot.

## Contact channels (three arms, same GRU)

Step vector is the V7B 12-D series. Only columns 5–6 (instant contact
and local contact statistic) change.

**B5-O (oracle).** V7B.3 contact: `contact_count` \(\to\) binary /
window fraction. Upper bound.

**B5-S (sensorized).** Learner-visible proxies, **not** MuJoCo contact
pairs and **not** exact hinge \(J^\top f\):

- wrist force/torque from a 6-D proprioceptive wrench observer
  (\(J_{\mathrm{eef}}^\top \hat f \approx \tau_{\mathrm{ext}}\) on Panda
  arm joints);
- arm joint-torque residual RMS;
- \(\log(1+\|\hat f\|)\) as a pressure proxy.

Additive Gaussian noise, frozen:

\[
\sigma_f=0.5\,\mathrm{N},\quad
\sigma_\tau=0.05\,\mathrm{N\cdot m},\quad
\sigma_r=0.1.
\]

**B5-N (none).** Contact columns zeroed. Ablation.

Mode-A episodes have no Panda–Door contact; all three channels are
zero there.

## Data / supervision

Same matched C0 record / C1 replay design as V7B.3. New seeds
\(\{21101,21111\}\) train, \(21121\) val, \(\{21131,21141\}\) held-out.
\(N=105\). All three arms train with B5 targets \(y\cdot w_t\).

\[
\delta=0.02,\quad
w_{\mathrm{low}}=0.25,\quad
w_{\mathrm{high}}=0.75.
\]

\(\tau_{\mathrm{dev}}=\) 95th percentile of **B5-S** development C0
\(p_T\), frozen once. H2 FPR uses this \(\tau\) on held-out C0 steps.

## Hypotheses (GO)

**H1.** Held-out Brier vs episode \(y\):
\(\mathrm{Brier}^{S}_{\mathrm{mid/final}}
\le
\mathrm{Brier}^{O}_{\mathrm{mid/final}}+\delta\).
Sensor need not beat oracle.

**H2.** \(P(p_t^{S}>\tau_{\mathrm{dev}}\mid C0)\le 0.20\).
Report \(B_{\mathrm{C0}}^{S}\) vs O/N; occupancy must not explode.

**H3.** Held-out C1, B5-S:
mean Spearman\((p_t,w_t)>0\) and
\(\mathbb E[p\mid w>w_{\mathrm{high}}]
>
\mathbb E[p\mid w<w_{\mathrm{low}}]\).

**H4.** Sensor channel is useful:
\(\mathrm{Brier}^{S}_{\mathrm{mid}}<\mathrm{Brier}^{N}_{\mathrm{mid}}\)
and
\(\mathrm{Brier}^{S}_{\mathrm{final}}<\mathrm{Brier}^{N}_{\mathrm{final}}\),
**or**
\(B_{\mathrm{C0}}^{S}<B_{\mathrm{C0}}^{N}\)
and C1-final Brier \(^{S}<^{N}\).

\[
\boxed{V7C\_GO=H1\land H2\land H3\land H4}
\]

Smoke must not set the GO.

## Stopping / what success does not authorize

If H2 fails, do **not** climb to tactile images or V7D on this Door
mixture. Reassess the contact observation model.

Success does not authorize RGB, taxel arrays, revision/VoI, or feeding
\(w_t\) at runtime.
