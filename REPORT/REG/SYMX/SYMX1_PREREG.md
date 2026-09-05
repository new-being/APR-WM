# SYM-X1 预注册 — Quotient Representation Utility

日期：2026-08-30  
状态：**已冻结；本格 RAN / `quotient_utility_supported`**  
依赖：SYM-X0 = `causal_symmetry_supported`（`runs/symx_x0/summary.json` 的 \(\hat G\)）  
**禁止**：RGB；改 X0 门；开 X2/X3；O0G6R / O1；R10。

## 科学问题

> 发现 \(G\) 之后，把状态压到等价类，能否用更少自由度/容量/样本达到同等世界预测——还是只是数学上漂亮？

\(\hat G\) **冻结自 X0 accept 集**（按物体），本格不再搜索候选、不再改 \(\tau\)。

## 表示（四种）

| ID | 表示 | 说明 |
|----|------|------|
| **B0** | full \(s=(p,R,v,\omega)\) | 显式刚体；不 quotient |
| **B1** | oracle quotient \(s/G^*\) | ceiling；C0/C5 用 \(d_{G^*}\)；C1 退化为 B0 |
| **B2** | discovered \(s/\hat G\) | 只用 X0 的 \(\hat G\) |
| **B3** | unconstrained latent | 同等或更大参数预算的普通 WM；**不知** \(G\) |

本格 **不** 为每种物体手写 \(SO(3)/SO(2)\) 坐标。对称感知距离：

\[
d_G(R_1,R_2)=\min_{g\in G}\,d_{SO(3)}(R_1,R_2g).
\]

模型预测的是 \([s]\in S/G\)，不是任意 gauge 代表元。  
B1/B2 的训练损失与 rollout 误差均用对应 \(d_G\)（位置/速度仍欧氏；\(\omega\) 在 \(G\) 下按伴随变换对齐后再比）。

**不永久删 gauge**（那是 X2）：本格只比较“主预测是否需要该坐标”。

## 任务与数据

同一 `symx_rigid.v1`；物体 **C0–C5** 分报，主声明在 **C0**（轴对称正例）与 **C4**（不得靠错误 yaw-quotient 获利）。  
Train/val/test probe 与 X0 **不同 seed**：`(41101, 41102, 41103)`（train / val / test）；每 seed 256 条；\(H=40\)；\(\Delta t=5\,\mathrm{ms}\)。  
B3 不得偷看 X0 accept 标签。

## 仪器细节（开跑前冻结）

同一 MLP 残差族：输入 \((p,R,v,\omega,a)\to\mathbb{R}^{12}\)（\(\Delta p,\delta\theta,\Delta v,\Delta\omega\)），\(R\leftarrow R\exp(\delta\theta)\)。  
\(W\in\{4,8,16,32,64\}\)；**匹配宽度** \(W^*=32\)（G0–G4）。\(P_{90}\)：该网格上最小参数使 \(E\le 1.05\,E^{B0}(W^*)\)。epochs=60。  
B1/B2：输入用离散 section（lex-least \(\{Rg:g\in G\}\)，即 \(s/\hat G\) / \(s/G^*\)），损失为 \(d_G\)；**不**手写 \(SO(3)/SO(2)\) 坐标。  
B3：encoder \(s\to z\in\mathbb{R}^{16}\)、latent dynamics、decode 到全状态；损失为 \(d\)（不知 \(G\)）；参数预算 ≥ B0。  
\(E\)：teacher-forced 单步，对 test probe 的 \(t=1..H\) 时间平均（\(d_{G^*}\)）。AR H-step 另报为诊断（接触刚体长自回归不稳定）。epochs=60。  
G4 探针：B1/B2 用 section 特征 \(s/G\)；B3 用 \(z\)。线性 ridge；yaw 为 \((\cos\theta,\sin\theta)\)。  
Few-shot \(K_{90}\) 仅 C0、不挡门。

## 三个预注册效果

### A. Prediction sufficiency（主门）

同一模型类、**匹配参数** 下：

\[
E_{\mathrm{rollout}}^{\mathrm{disc}}\le(1+\varepsilon)\,E_{\mathrm{rollout}}^{\mathrm{full}},\qquad \varepsilon=0.05.
\]

\(E\) = test 上 \(d\) 或 \(d_{\hat G}\) 的时间平均（B0 用 \(d\)；B1/B2 用各自 \(d_G\)；跨表示比较时 **额外**报一份共用 \(d_{G^*}\) 以免 B2 靠过宽 \(\hat G\) 刷分）。

| 条件 | 期望 |
|------|------|
| C0/C5 | B2 非劣于 B0；B2 ≈ B1 |
| C1 | B2 ≈ B0（\(\hat G=\{I\}\)） |
| C4 | B2 不得靠接受 \(R_y(\theta\notin\{0,180\})\)；X0 已拒，本格只验证压缩后仍能预测各向异性自旋 |

### B. Capacity / sample

容量扫描同一架构族，定义达到 B0 匹配预算 90-分位误差的最小参数 \(P_{90}\)：

\[
R_P=1-P_{90}^{\mathrm{disc}}/P_{90}^{\mathrm{full}}.
\]

主声明要求 C0 上 \(R_P>0\)（发现对称减少学习自由度）。  
另报 few-shot \(K_{90}\)（可选，不挡 A）。

### C. Gauge leakage（对 B3）

探针 \(z\mapsto\theta_{\mathrm{yaw}}\)（杯轴）与 \(z\mapsto n_{\mathrm{axis}}\)。

| | 期望 |
|--|------|
| B2 / B1 | \(I(s/G;\theta_{\mathrm{yaw}})\approx0\)；\(n_{\mathrm{axis}}\) 探针保持高 \(R^2\) |
| B3 | 若 \(R^2_{\mathrm{yaw}}\gg0\) 而 A 仍好：说明 latent **存了不用的 nuisance** |

C4：yaw **不是** nuisance；B1/B2 的 yaw 探针应保持信息（180° 离散除外）。

## Gates

| Gate | 条件 |
|------|------|
| G0-fit | B0 在 C0 test 上明显优于 mean predictor（\(E_{\mathrm{full}}<0.8\,E_{\mathrm{mean}}\)） |
| G1-noninferior | C0：\(E^{B2}\le1.05\,E^{B0}\)（\(d_{G^*}\)） |
| G2-oracle-gap | C0：\(E^{B2}\le1.05\,E^{B1}\) |
| G3-no-false-compress | C4：\(E^{B2}\le1.05\,E^{B0}\)（不得用过宽 \(G\)） |
| G4-leakage | C0：B1/B2 的 yaw 探针 \(R^2\le0.10\)；axis 探针 \(R^2\ge0.80\) |

B3 只描述、不挡 G1。若 B3 非劣且 yaw \(R^2\le0.10\)，记 `latent_already_quotients`（仍不跳过显式 \(\hat G\) 的 X2 安全问题）。

## Patterns

| Pattern | 条件 |
|--------|------|
| `instrument_failure` | ¬G0 |
| `quotient_hurts_prediction` | G0 ∧ ¬G1 |
| `discovered_ne_oracle` | G0∧G1 ∧ ¬G2 |
| `false_compress_C4` | G0 ∧ ¬G3 |
| `gauge_still_in_quotient` | G0∧G1∧G3 ∧ ¬G4 |
| `quotient_utility_supported` | G0–G4 |

## 解锁

`quotient_utility_supported` → 允许预注册 **SYM-X2**（已跑：[`SYMX2_PREREG.md`](SYMX2_PREREG.md) / `gauge_reactivation_supported`）。  
X3 / O0G6R / O1 仍 LOCKED。
