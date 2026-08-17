# R5-D0 Report — Conditional Decision Value

Date: 2026-08-17  
Status: **`R5_D0_GO=false`** (self-stress family **STOP**)  
Prereg: `REPORT/REG/R5/R5_D0_PREREG.md`  
Artifacts: `runs/r5_d0/formal/summary.json`  
Does not change: I0 `PASS=false`; I1 `GO=true`; preflight `PASS=true`;
\(\mathcal A\); \(J\)  
Does not train: action classifier or neural planner

## Question

Leave-one-\(\lambda\)-out, ridge world models, then \(J(\hat Y)\):

does \(\pi_X\) lower **executed** regret versus \(\pi_0\), beat shuffled
\(X\), agree more with oracle, and actually select different actions?

## Result

| Gate | Value | Pass |
|---|---|---|
| H1 \(\bar R_X<\bar R_0\) rel\({>}0.05\) | \(\bar R_0=1.72\times10^{-5}\), \(\bar R_X=2.18\times10^{-5}\), rel \(=-0.26\) | false |
| H2 \(\mathrm{Acc}_X>\mathrm{Acc}_0\) | **0.57** vs **0.86** | false |
| H3 \(P(\hat a_X\neq\hat a_0)>0\) | **0.29** | true |
| H4 \(\bar R_X<\bar R_{\mathrm{shuf}}\) rel\({>}0.05\) | \(\bar R_{\mathrm{shuf}}=2.10\times10^{-5}\), rel \(=-0.04\) | false |

\[
\boxed{\texttt{R5\_D0\_GO}=\text{false}}
\]

LOO actions (oracle \(a^\star\): cons at \(0\), mid else):

| \(\lambda\) | \(a^\star\) | \(\hat a_0\) | \(\hat a_X\) | \(\hat a_{\mathrm{shuf}}\) |
|---|---|---|---|---|
| 0 | cons | mid | mid | mid |
| 2 | mid | mid | **cons** | mid |
| 4–10 | mid | mid | mid | mid (cons at 10 shuf) |
| 12 | mid | mid | **cons** | mid |

\(\pi_0\) is a constant-`mid` rule and is already optimal on six of seven
\(\lambda\). \(\pi_X\) never recovers the \(\lambda=0\) cons choice, and
it **injects** two extra cons errors on the mid regime. Regime acc:
\(\lambda=0\) both \(0\); \(\lambda\ge 2\) acc\(_0=1\), acc\(_X=0.67\).

## Interpretation

Not proven: “\(X\) has no value of information.” Oracle ranking already
flips with \(\lambda\), and \(h^{S}\) cannot see \(\lambda\), so an
**optimal** decisioner that may use \(X\) would strictly improve
expected \(J\) (switch to cons only at \(\lambda=0\)).

Proven, for this frozen ridge-\(\hat Y\)-then-\(J\) class \(\Pi\):

\[
\boxed{
I(X;Y^{\mathrm{future}}\mid h^{S})>0
\ \land\
\operatorname{Var}_\lambda(a^\star)>0
\ \not\Rightarrow\
\mathbb{E}[R(\pi_X)]<\mathbb{E}[R(\pi_0)]
}
\]

Define realized / algorithmic VoI

\[
\mathrm{VoI}_{\Pi}(X)
=
\mathbb{E}[R(\pi_0)]-\mathbb{E}[R(\pi_X)].
\]

Here \(\mathrm{VoI}_{\Pi}(X)<0\). That does **not** imply
information-theoretic or oracle \(\mathrm{VoI}(X)\le 0\).

I1 lowers trajectory MSE; D0 needs \(\arg\min_a J(\hat Y_a)\) to stay
correct. Those are not equivalent:

\[
L_Y\downarrow
\not\Rightarrow
\Pr[\arg\min J(\hat Y)=\arg\min J(Y)]\uparrow.
\]

\(X\) **did** change some decisions (H3) but not the \(\lambda=0\)
switch, and it added wrong cons choices. Information entered the
planner; it did not enter **correctly**. Shuffle is not worse, so a
true \(X\)–future pairing does not by itself yield decision gain under
\(\Pi\).

\[
\boxed{
\text{predictive information}
+
\text{oracle decision relevance}
\neq
\text{realized decision value under a finite planner}
}
\]

Do not rescue by an \(X\to a^\star\) classifier, by retuning \(J\), or
by dropping \(\lambda=0\) from LOO.

Self-stress family **stops** here. Contact family stays paused. No
encoder.
