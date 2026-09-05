# RTWX-X0EH 报告 — Receding-Horizon Sufficiency Audit

日期：2026-08-28  
状态：**正式冻结** `short_horizon_sufficient`  
预注册：`REPORT/REG/RTWX/RTWX0EH_PREREG.md`  
产物：`runs/rtwx_x0eh/{run.json,fresh_splits.npz,metrics.json}`  
冻结 X0E-M2；**纯诊断**。不改 X0E/X0E1/X0ES/X0ES1/X0ES2。无 \(R_P\)。

## 一句话

在 fresh 96×120（seed 11601）上，**每 4 步真实观测 reset 足以消除 deployment-relevant catastrophe**（G1），同时 **4 步执行误差显著优于 identity**（G2），且 **\(E_{\mathrm{end}}(K)\) 随 \(K\) 单调恶化**（G3）。  
结论：**M2 不需要 50-step 全局 contraction**；机器人实际用法应是 \(H_{\mathrm{plan}}=16\)、\(H_{\mathrm{execute}}=4\) 的 receding horizon。

## Part A — open-loop \(T_{\mathrm{div}}\)（6720 窗）

| 量 | 值 |
|----|-----|
| \(N_{\mathrm{cat}}\) | **11**（0.16%） |
| \(K_{99}\) | **50**（99% 窗在 50 步内不爆） |
| \(T_{\mathrm{div}}\) 中位（有限爆窗） | **9** 步 |

生存曲线 \(P(T_{\mathrm{div}}\le h)\)：

| \(h\) | 1 | 2 | 4 | 8 | 16 | 32 | 50 |
|------:|--:|--:|--:|--:|---:|---:|---:|
| \(P\) | 0 | 0 | 0 | 0.06% | **0.16%** | 0.16% | 0.16% |

描述性：\(T_1\) 中位 = 1；\(T_{10}\) 中位 = 32。绝大多数爆炸发生在 **8 步之后**。

## Part B — receding horizon

| \(K\) | \(r_{\mathrm{cat}}\) | \(N_{\mathrm{nf}}\) | \(E_{\mathrm{exec}}\) | \(E_{\mathrm{end}}\) | \(E_{\mathrm{id}}\) | \(R_\rho\) |
|------:|---------------------:|--------------------:|------------------------:|----------------------:|----------------------:|-----------:|
| 1 | 0 | 0 | 0.498 | 0.498 | 0.815 | 3.4% |
| 2 | 0 | 0 | 0.533 | 0.522 | 1.002 | 5.7% |
| **4** | **0** | **0** | **0.574** | **0.522** | 1.218 | 10.5% |
| 8 | 0 | 0 | 0.623 | 0.520 | 1.334 | 18.4% |
| 16 | 0.15% | 0 | — | \(10^{117}\) | 1.357 | 32.4% |
| 50 | 0.52% | 1 | inf | inf | 1.169 | 66.1% |

主候选 **\(K=4\)**：零 catastrophic segment；\(E_{\mathrm{exec}}=0.574 \le 0.75\times1.218\)。

## 门

| 门 | 结果 |
|----|------|
| G0 | **通过**。\(N_{\mathrm{cat}}=11\ge5\) |
| G1 | **通过**。\(r_{\mathrm{cat}}(4)=0 \le 0.1\times0.52\%\)；\(N_{\mathrm{nf}}(4)=0\) |
| G2 | **通过**。\(E_{\mathrm{exec}}(4)=0.574 \le 0.75\times1.218\) |
| G3 | **通过**。Spearman\((K,E_{\mathrm{end}})=0.77\ge0.7\)；\(E_{\mathrm{end}}(16)\gg E_{\mathrm{end}}(4)\) |

Pattern：**`short_horizon_sufficient`**

## Shadow（不进 pattern）

| 策略 | \(r_{\mathrm{cat}}\) |
|------|---------------------:|
| fixed \(K=4\) | 0 |
| fixed \(K=8\) | 0 |
| adaptive \(\rho>1\Rightarrow K{=}1\) else \(K{=}8\) | 0 |

adaptive 在本 fresh 上无额外收益（固定 \(K\le8\) 已零灾）。

## 读数与路线

1. **开环 50 步并非部署契约**：稀有爆炸（0.16%）且多在 \(h\ge8\) 才出现；与 X0ES1 的“稀有局部递归不稳定”一致。
2. **\(K=4\) reset 是有效 mitigation**：不改 M2 即可将 segment catastrophe 压到 0，并保留预测价值。
3. **暂停 contraction 主线**：X0ES2 的 `instability_persists` 针对的是 **50-step open-loop**；本格表明该目标对 deployment 过强。
4. **解锁下一格方向**（非本格声称）：可考虑 **\(R_P^{K=4}\)** — 在 receding-horizon 使用方式下做 structure vs NN capacity；须新开 fully fresh 预注册格。
5. 无 \(R_P\)；不改 X0E1 `structure_not_in_robust_set`。

## 账本

```text
RTWX-X0ES2 = RAN / instability_persists          不改（open-loop contraction STOP）
RTWX-X0EH  = RAN / short_horizon_sufficient
             K=4 reset eliminates segment catastrophes; G0-G3 pass
next allowed: K=4 deployment-relevant capacity cell (new prereg)
contraction: PAUSED unless replanning_insufficient (not this result)
TASK-XL / R10 = LOCKED
```
