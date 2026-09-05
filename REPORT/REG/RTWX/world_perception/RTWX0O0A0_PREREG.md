# RTWX-O0A0 预注册 — Appearance-Conditioned Yaw Observability

日期：2026-08-30  
状态：**已冻结 / RAN** `appearance_yaw_generalization_failure`  
依赖：O0G6A=`global_geometry_ambiguous`（几何钉住倾倒、杯轴 yaw 退化）；O0G2R position PASS；O0C `orientation_failure`（~90° 重尾）  
**旁支**：O0Q* = PAUSED / infrastructure-blocked（无因果 quotient 结论；**不**改 O0 target）  
**禁止**：改 O0C/O0R orientation 门（15°/30°）；重训 full-\(R\)；PointNet / CAD / Kabsch / RANSAC / ICP / correspondence；混测 segmentation（O0G2R 已过）；跨 base snapshot 泄漏 split；开 O0A1 / O0T0 / O1 除非本格 pattern 解锁。

## 科学问题

几何已不能确定杯轴 yaw。问：

\[
\boxed{
I(\theta_{\mathrm{yaw}};RGB\mid p,n_{\mathrm{cup}},\mathrm{camera})>0
\quad\text{且能否 fresh 泛化？}
}
\]

分解：

\[
R=R_{\mathrm{axis}}(n)\,R_y(\theta_{\mathrm{yaw}}).
\]

本格**不是**重新训练 full-\(R\)；是受控 ceiling：oracle 条件掉 nuisance，只测 appearance 是否稳定编码 canonical yaw。

## 冻结合同

| 项 | 值 |
|----|------|
| 数据 | fresh natural **base snapshots**；仅 render，**不** step physics |
| 冻结 | \(p,\,n_{\mathrm{cup}},\,q,\,\mathrm{camera},\,\mathrm{lighting}\) |
| 唯一变化 | \(R\leftarrow R\,R_y(\theta_k)\)，\(\theta_k=15^\circ k\)，\(k=0,\ldots,23\) |
| Split | **按 base snapshot 分组**：同一 base 的 24 yaw **同属** train/val/test；例 **48/24/24** |
| \(\mathcal O_t\) | `head` + `observer` |
| 输入 | **oracle GT mask** → bbox crop + fixed padding → \(128\times128\)；背景置零 |
| Mask | GT（不重测 segmentation） |
| seeds | **35601**（formal）；smoke 缩规模 |

### Branches（同数据合同）

| ID | 输入 | 角色 |
|----|------|------|
| **B0** | \((D^{\mathrm{head}},D^{\mathrm{obs}},\mathrm{mask})\) | depth-only **负对照**（预期差） |
| **B1** | \((RGB^{\mathrm{head}},RGB^{\mathrm{obs}},\mathrm{mask})\) | **primary** |
| **B2** | RGB-D 双视角 | 稳定性 / P90；claim 仍由 B1 |

### 模型（固定，非 arch 竞赛）

共享小 encoder \(f_\theta(I_{\mathrm{head}}),f_\theta(I_{\mathrm{obs}})\) → concat → linear/2-layer → **24-bin CE**。  
**空间 flatten，不用 GAP**（与已冻结的 O0D3/O0R orientation 编码器合同一致；GAP 会抹掉 crop 内外观的空间位移）。  
不用 MSE 角度回归（周期边界）。**训满预注册 epoch**（ceiling；不用 val early-stop 把随机初始化冻成 checkpoint）。circular error：

\[
e_\theta=\min_{m\in\mathbb Z}|\hat\theta-\theta+360^\circ m|.
\]

量化天花板 \(\le 7.5^\circ\)。

## Gates

| Gate | 条件 |
|------|------|
| **G0** render instrument | (1) 同 yaw 重复 render \(\Delta RGB\approx0\)；(2) yaw 后 \(p,n_{\mathrm{cup}}\) 不变；(3) 24 labels 与 sim \(R\) 一致。失败 → STOP |
| **G1** fresh RGB yaw（B1） | \(\mathrm{median}(e_\theta)\le 15^\circ\) 且 \(\mathrm{P90}\le 30^\circ\)；另报 Top1/3/5 |
| **G2** tail（报） | \(f_{90}=P(e_\theta\ge 75^\circ)\)（对齐 O0C ~90° catastrophe） |
| **G3** appearance gain（报） | \(\Delta e=e_{\mathrm{depth}}-e_{\mathrm{RGB}}\)；不要求 B0 完全随机 |

## Patterns（互斥）

| Pattern | 条件 |
|---------|------|
| `appearance_instrument_failure` | ¬G0 |
| `appearance_yaw_generalization_failure` | G0 ∧ ¬G1（B1） |
| `appearance_yaw_supported` | G0 ∧ G1(B1) ∧ ¬G1(B0) |
| `geometry_yaw_supported_unexpectedly` | G0 ∧ G1(B0)（B0 意外过门；优先于 appearance_supported） |

若 G0∧G1(B1)∧G1(B0)：记 `geometry_yaw_supported_unexpectedly`（几何负对照失效，需重审合同）。

## 解锁

| Pattern | 解锁 |
|---------|------|
| `appearance_yaw_supported` | 允许预注册 **O0A1**（composite non-oracle：\(\hat R=R_{\mathrm{axis}}(\hat n)R_y(\hat\theta)\)） |
| `appearance_yaw_generalization_failure` | 允许预注册 **O0T0**（temporal yaw observability；仍是 observability，非完整 O1） |
| 其余 | O0A1 / O0T0 / O1 **LOCKED** |

**O0 target 继续 full \((T,R)\)。** O0Q* 旁支不消耗主线格。O0G6R / O0G5R / O0C2 / SYM-X3 / O1 LOCKED。
