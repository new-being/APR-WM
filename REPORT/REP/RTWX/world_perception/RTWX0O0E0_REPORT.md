# RTWX-O0E0 报告 — Symmetry-Aware Effective-Pose Closure

日期：2026-08-31  
状态：**正式冻结** `observation_support_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0_PREREG.md`  
产物：`runs/rtwx_o0e0/{summary.json,run.json,header.json,cache/{train,val,test}/{obs,gt,seg_gt}.npz}`  
数据：**natural** `place_empty_cup`；seed **36601**；48/24/24 × 120；128²；head+observer  

**硬解释（不得改写）：**

\[
\boxed{
\text{当前 natural-task observation distribution 不足以承载 O0E0 的正式检验}
}
\]

**不是** \(RGBD\not\rightarrow(p,n)\)。  
B0 / B1 / B2 **UNTESTED**（未跑）。不得把本格写成 quotient-CAD axis 失败。

## 一句话

目标 \((p,n)\) 合理，但 **natural 数据分布不够格** 去检验它。coverage 未过门；只读诊断显示 axis 几乎无激励。

\[
\boxed{\texttt{observation\_support\_failure}}
\]

## 因果链（未闭合 science）

\[
\text{O0G6A: axis 几何可辨}
\rightarrow
\text{O0A0: canonical yaw 不可辨}
\rightarrow
\text{O0E0: effective axis 是合理目标}
\]

且

\[
\text{目标定义合理}\;\not\Rightarrow\;\text{当前数据分布足以检验它}.
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 coverage | \(P(V^{\mathrm{any}})\ge0.98\) | **FAIL**（**0.966**；约 98/2880 帧双视角 vis 失败） |
| G0 detection | \(P(\hat M\neq\emptyset\mid V)\ge0.95\) | **未评** |
| G0b excitation | \(r_{\mathrm{excite}}\ge0.30\) | **未作门** |
| G1–G3 B0/B1/B2 | axis / position | **未跑 / UNTESTED** |

不得因 G0 失败改 B2 搜索或重训 O0E0。

## Diagnostic（非门；G0 STOP 后只读 GT）

| 量 | 值 |
|----|-----|
| \(n_{\mathrm{const}}\) | \(\approx(0,0,1)\)（constant-up） |
| median \(\angle(n_t,n_{\mathrm{const}})\) | **2.0°** |
| \(r_{\mathrm{excite}}\) | **0.126** \(<0.30\) |

自然任务 **axis DOF 低熵**。即使把 coverage 提到 0.98，原分布仍无法区分「估了 axis」与「输出 constant-up」。  
这是 **task necessity ≠ representation capability**；不得写成 science claim。

## Pattern

```text
pattern = observation_support_failure
claim = distribution_insufficient_for_O0E0_test
not_claim = RGBD_cannot_recover_p_n
B2_effective_axis = UNTESTED
natural_axis_excitation = diagnostic_insufficient
unlocks_o0e1 = false
unlocks_o1 = false
next = O0E0P0 controlled-pose observation qualification
```
