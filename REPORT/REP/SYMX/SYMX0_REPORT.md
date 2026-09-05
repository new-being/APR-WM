# SYM-X0 报告 — Causal Symmetry Discovery

日期：2026-08-30  
状态：**正式冻结** `causal_symmetry_supported`  
预注册：`REPORT/REG/SYMX/SYMX0_PREREG.md`  
产物：`runs/symx_x0/{summary.json,run.json}`  
Host：`symx_rigid.v1`；**无 RGB**；seeds 40101/02/03；96 probe × 3；\(H=40\)；\(\tau=0.04\)  
允许预注册 **SYM-X1**。X2 / X3 / O0G6R / O1 LOCKED。

## 一句话

冻结候选上，\(D_H(g)\) 能恢复各物体正确的因果对称，并且 **C4 不把外形当动力学、C5 不被 logo 欺骗**。  
这与 O0G6A 的 shape matching 是不同命题：这里测的是干预后是否还是同一物理机制。

\[
\boxed{\texttt{causal\_symmetry\_supported}}
\]

## 仪器

Oracle \(s=(p,R,v,\omega)\)；桌面 penalty 接触；球环近似。  
\(\omega\) 数值帽改为 **各向同性** \(|\omega|\le40\)（分量 `clip` 的立方体只对 90° 轴置换不变，会把 \(R_y(15^\circ)\) 判成不对称）。**未改** \(\tau\) / \(G^*\) / probe 合同。

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-instrument | 全部 \(D_H(I)<0.01\)；\(P_{\mathrm{excite}}\ge0.50\) | **PASS**（\(D_H(I)\sim10^{-9}\)；excite=**1.00**） |
| G1-false-quotient | \(P(\mathrm{accept}\mid g\notin G^*)<0.01\) | **PASS**（**0 / 289**） |
| G2-recall | \(P(\mathrm{accept}\mid g\in G^*)\ge0.90\) | **PASS**（**1.000**，131/131） |
| G3-C4 | 不接受非 \(\{0,180\}\) 的 \(R_y(\theta)\) | **PASS**（0/22） |
| G4-C5 | C5 与 C0 的 \(R_y\) accept 集相同 | **PASS** |

AUROC(\(-D_H\), \(g\in G^*\))=**1.000**（六条件）。

## 主表

| 条件 | \(G^*\) 规模 | 接受 | \(D_H(R_y90)\) | \(D_H(R_x90)\) | 读数 |
|------|-------------:|-----:|----------------:|---------------:|------|
| **C0** 轴对称杯 | 26 | 26 | \(\sim5\times10^{-9}\) | **6.11** | 全 \(R_y(\theta)\) + \(R_{x,z}(180^\circ)\) |
| **C1** 有柄 | 1 | **1**（仅 \(I\)） | 4.00 | 3.83 | 拒绝 yaw / 180° |
| **C2** 长方体 | 4 | **4** Klein | 6.77 | 3.35 | 仅三轴 180° |
| **C3** 球 | 70 | **70** | \(\sim10^{-8}\) | \(\sim10^{-8}\) | 全部候选 |
| **C4** 各向异性惯量 | 4 | **4** | **8.20** | 7.66 | **拒非平凡 yaw** |
| **C5** 纯视觉 logo | 26 | 26 | \(\sim10^{-9}\) | 6.11 | 与 C0 同 \(\hat G\) |

C0：\(D_H(R_y90)\ll D_H(R_x90)\)，与 O0G6A 的几何退化轴一致，但判定来自 **counterfactual rollout**，不是 Chamfer。

## 读数

1. **不是 shape matching**：C0 与 C4 几何相同；C4 的 \(I_x\neq I_z\) 让 \(D_H(R_y90)=8.20\)（拒绝），C0 为数值零。发现的是因果对称。
2. **不是 appearance**：C5 的 logo 不进 \(d\)；accept 集与 C0 相同。若把 RGB 放进 \(D_H\)，这条对照会失效。
3. **false quotient = 0**：在 289 个负例上零误接受。漏对称只是多存状态；本格没有删错自由度。
4. **有限群也能分**：C1 只有 \(I\)；C2 恰好 Klein 四元群；C3 全接受。不是“杯子特化阈值”。
5. **解锁 SYM-X1**：下一步问的是 \(s/\hat G\) 能否以更少容量达到同等预测，以及 unconstrained latent 是否仍泄漏 yaw。**不**开 X2/X3，**不**开 O0G6R。

## Pattern

```text
pattern = causal_symmetry_supported
G0_instrument = PASS
G1_false_quotient = 0.000
G2_recall = 1.000
G3_C4 = PASS
G4_C5 = PASS
unlocks_symx1_prereg = true
unlocks_symx2 = false
unlocks_symx3 = false
unlocks_o0g6r = false
unlocks_o1 = false
```
