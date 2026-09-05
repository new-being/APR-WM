# RTWX-X0EH1 报告 — Receding-Horizon Structured-vs-Neural Capacity

日期：2026-08-28  
状态：**正式冻结** `receding_structure_capacity_shift`  
预注册：`REPORT/REG/RTWX/RTWX0EH1_PREREG.md`  
产物：`runs/rtwx_x0eh1/{run.json,cache_splits.npz,metrics.json,W_m2.npy}`  
部署合同：\(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\)；fully fresh seed **12601**（48/24/24×120）。  
不改 X0E/X0E1/X0ES/X0ES1/X0ES2/X0EH ledger。Primary：**B0 vs B1**（无 B2 residual）。

## 一句话

在统一 \(K=4\) 真实观测 reset 的部署合同下，**冻结 X0E-M2（\(P=1752\)）matched 最大 neural reference H=256**，且 test 上 \(E_{\mathrm{exec}}^{K=4}\) 优于 ref。  
正式容量：**\(R_P^{K=4}=0.988\)**，**\(C_P^{K=4}\approx84.0\times\)**（\(P_{\mathrm{NN}}^{\min}=147224\)）。  
与 X0E1 开环合同不同：本格下 **H=128 不 matched**，最小 matched NN 为 **H=256**。

## 部署合同

每 4 个 native action：从真实 \(x_t\) 开环 rollout 4 步，只计执行区间误差；第 4 步末强制 \(\hat x\leftarrow x^{obs}\)。  
catastrophe：segment 内 \(\max E_h>10^6\) 或 nonfinite（与 X0ES 相同）。

Reference competent（H=256，test）：\(r_{\mathrm{cat}}=0\)，\(E_{\mathrm{exec}}=0.737\le0.75\times1.265\)（identity）。**通过。**

Matched vs H=256：\(E_{\mathrm{exec}}\le1.05\times\) ref，\(E_{\mathrm{end}}\le1.10\times\) ref，\(r_{\mathrm{cat}}=0\)。

## Test 主表（\(K=4\)，720 segments / 24 ep）

| 模型 | \(P\) | \(E_{\mathrm{exec}}\) | \(E_{\mathrm{end}}\) | \(E_{\mathrm{id}}\) | \(r_{\mathrm{cat}}\) | matched |
|------|------:|------------------------:|----------------------:|----------------------:|---------------------:|:-------:|
| **B1 M2** | **1752** | **0.615** | **0.568** | 1.265 | **0** | **✓** |
| B0 H=8 | 656 | 1.191 | 1.565 | 1.265 | 0 | ✗ |
| B0 H=16 | 1544 | 1.147 | 1.501 | 1.265 | 0 | ✗ |
| B0 H=32 | 4088 | 1.090 | 1.411 | 1.265 | 0 | ✗ |
| B0 H=64 | 12248 | 0.999 | 1.245 | 1.265 | 0 | ✗ |
| B0 H=128 | 40856 | 0.821 | 0.921 | 1.265 | 0 | ✗ |
| B0 H=256 (ref) | 147224 | 0.737 | 0.774 | 1.265 | 0 | ✓ |

B1 相对 ref：\(E_{\mathrm{exec}}\) 约 **16% 更低**（0.615 vs 0.737），\(E_{\mathrm{end}}\) 约 **27% 更低**（0.568 vs 0.774），均在 matched 裕度内。

## 容量（本格正式定义）

\[
P_{\mathrm{struct}}=1752,\qquad
P_{\mathrm{NN}}^{\min}=147224\ (\text{width }256),
\]

\[
R_P^{K=4}=1-\frac{1752}{147224}=0.988,\qquad
C_P^{K=4}=\frac{147224}{1752}\approx 84.0.
\]

**不得**将 X0E1 开环上的 \(P_{\mathrm{NN}}^{\min}=40856\)（H=128）与本格数字混报：合同不同，matched 宽度不同。

## 部署算力（\(H_{\mathrm{plan}}=16\) 前向 / \(K=4\) 执行）

| 模型 | \(t_{\mathrm{plan}}\) (ms) | \(C_{\mathrm{deploy}}\) (ms/action) |
|------|---------------------------:|------------------------------------:|
| B1 M2 | 0.163 | **0.041** |
| B0 H=256 | 12.4 | 3.10 |

结构 rollout 在本机 proxy 上约 **76×** 更快（每执行 action 摊销）。参数量少且前向更轻，二者一致。

## Pattern 与门

| 检查 | 结果 |
|------|------|
| Reference competent | **通过** |
| B1 matched | **通过** |
| \(P_{\mathrm{NN}}^{\min}>2P_{\mathrm{struct}}\) | **通过**（147224 ≫ 3504） |

Pattern：**`receding_structure_capacity_shift`**  
`capacity_claim=true`（仅指 \(R_P^{K=4}\) 在本格合同下成立）。

## 与机制链的关系

```text
X0ES1  → rare_instability_intrinsic_map     不改
X0ES2  → instability_persists (50-step OL)   不改；contraction STOP
X0EH   → short_horizon_sufficient            不改；定义 K=4 部署合同
X0EH1  → RAN / receding_structure_capacity_shift
         R_P^{K=4}=0.988, C_P^{K=4}≈84× on fresh seed 12601
```

1. **X0EH 把问题从“50 步全局稳定”改写成“4 步可信 + 周期 reset”**；本格在该合同下回答容量。
2. **H=128 在 \(K=4\) 下不够 matched**（\(E_{\mathrm{end}}=0.921>1.10\times0.774\)），说明 receding 合同比开环 one-step/roll50 更严地筛 reference。
3. **M2 不仅更省参，而且在 test 上优于 H=256 ref**——这是 deployment-relevant 意义上的 structure win，不是仅参数量 accounting。
4. 仍 **不是** torque-level physics ID；仍是 native-window macro-transition capacity。

## 账本

见 `REPORT/REP/RTWX/RTWX_LEDGER.md`（正式汇总）。

```text
RTWX-X0EH1 = PASS / receding_structure_capacity_shift
             R_P^(K=4)=0.988  C_P^(K=4)≈84.0×  (put_object_cabinet, seed 12601)
NEXT       = RTWX-X0EH2 cross-task external validity (preregistered)
TASK-XL / R10 = LOCKED
```

**禁止事后**：扩 width、改 M2 basis、改 matched 阈值、追更大 compression ratio。  
下一格价值在 **外部有效性**，不在把 84× 做得更漂亮。
