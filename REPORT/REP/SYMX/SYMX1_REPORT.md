# SYM-X1 报告 — Quotient Representation Utility

日期：2026-08-30  
状态：**正式冻结** `quotient_utility_supported`  
预注册：`REPORT/REG/SYMX/SYMX1_PREREG.md`  
产物：`runs/symx_x1/{summary.json,run.json}`  
依赖：SYM-X0=`causal_symmetry_supported`；\(\hat G\) 冻结自 X0 accept 集  
允许预注册 **SYM-X2**。X3 / O0G6R / O1 LOCKED。

## 一句话

发现的 \(\hat G\) 做成 \(s/\hat G\) 后，**预测相对 full state 非劣**（C0 +1%，C4 不靠错误 yaw-quotient），并且 **section 特征不含 yaw、仍含轴**。  
未约束 latent **没有**自发商掉 yaw。本格网格上 **没有** \(P_{90}\) 下降——压缩的是表示内容，不是 MLP 参数。

\[
\boxed{\texttt{quotient\_utility\_supported}}
\]

## 仪器

Teacher-forced 单步 \(E\)（\(d_{G^*}\)）；H-step AR 只作诊断（接触刚体上 AR 贴 persist、不 identifiability）。  
B1/B2：lex section + \(d_G\)；B3：\(z\in\mathbb{R}^{16}\) 不知 \(G\)。\(W^*=32\)。

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-fit | C0：\(E^{B0}<0.8\,E_{\mathrm{mean}}\) | **PASS**（0.647 / 0.851 = **0.759**） |
| G1-noninferior | C0：\(E^{B2}\le1.05\,E^{B0}\) | **PASS**（**1.006**） |
| G2-oracle-gap | C0：\(E^{B2}\le1.05\,E^{B1}\) | **PASS**（**1.000**） |
| G3-C4 | C4：\(E^{B2}\le1.05\,E^{B0}\) | **PASS**（**0.998**） |
| G4-leakage | C0：B1/B2 yaw \(R^2\le0.10\)，axis \(\ge0.80\) | **PASS**（yaw **−0.033**；axis **0.829**） |

## 主表（\(W^*\)，\(d_{G^*}\)）

| 条件 | \(E^{B0}\) | \(E^{B1}\) | \(E^{B2}\) | \(E^{B3}\) | \(E_{\mathrm{mean}}\) | B2 yaw \(R^2\) | B3 yaw \(R^2\) |
|------|----------:|----------:|----------:|----------:|---------------------:|---------------:|---------------:|
| **C0** | 0.647 | 0.651 | **0.650** | 0.659 | 0.851 | **−0.033** | **0.390** |
| C1 | 0.621 | 0.624 | 0.632 | 0.632 | 0.792 | 0.868 | 0.368 |
| C2 | 0.827 | 0.833 | 0.826 | 0.837 | 0.998 | 0.107 | 0.452 |
| C3 | 0.307 | 0.312 | 0.313 | 0.319 | 0.447 | 0.087 | 0.468 |
| **C4** | 0.901 | 0.895 | **0.899** | 0.912 | 1.111 | **0.139** | 0.495 |
| C5 | 0.637 | 0.629 | 0.641 | 0.651 | 0.838 | **−0.029** | 0.443 |

C0 上 B0 特征 yaw \(R^2=0.86\)（full \(R\) 必含 yaw）。

## 容量

\(P_{90}\) 取达到 \(1.05\,E^{B0}(W^*)\) 的最小网格点。C0：\(P_{90}^{B0}=220\)（\(W=4\)），\(P_{90}^{B2}=460\)（\(W=4\) 略超阈）⇒ \(R_P=-1.09\)。  
各物体 \(R_P\in\{0,-1.09\}\)，**不随 \(|\hat G|\) 上升**。Few-shot C0：K=32 已有 \(E^{B2}=0.660\)（满数据 0.650）。

## 读数

1. **非劣预测成立**：B2≈B1≈B0。X0 的 0 false quotient / recall 1 在表示层兑现——discovery 误差没有伤害预测。
2. **C4 没有偷 yaw-quotient**：同外形、惯量不对称时 B2 仍非劣，且 section 仍线性可解码连续 yaw（0.14）。复杂度跟的是因果 \(G\)，不是杯子外形。
3. **C5 不被 logo 拖回 yaw**：与 C0 一样 yaw \(R^2\approx0\)。
4. **B3 未自发 quotient**：C0 上 \(z\to\theta_{\mathrm{yaw}}\) 的 \(R^2=0.39\)（门 0.10）；axis 探针仅 0.47。是“存了一部分 nuisance、也没把相关状态存满”，不是最小等价类。`latent_already_quotients=false`。
5. **\(R_P\) 不是正结果**：该残差-MLP 网格上，忽略 yaw 几乎不减参数——传播姿态仍廉价。实用压缩若存在，更可能在样本/注意力/状态带宽，而不是这一层宽度。

## Pattern

```text
pattern = quotient_utility_supported
G0_fit = PASS (0.759)
G1_noninferior = PASS (1.006)
G2_oracle_gap = PASS (1.000)
G3_C4 = PASS (0.998)
G4_leakage = PASS (yaw=-0.033, axis=0.829)
latent_already_quotients = false
R_P(C0) = -1.09   # not a gate
unlocks_symx2_prereg = true
unlocks_symx3 = false
unlocks_o0g6r = false
unlocks_o1 = false
```
