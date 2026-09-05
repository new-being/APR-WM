# RTWX-O0MEMB0 报告 — Reference Memory Breadth Probe (P0)

日期：2026-09-01  
状态：**RAN / 架构选择完成**；`scientific_result=false`；**winner=none**  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0MEMB0_PREREG.md`  
产物：`runs/rtwx_o0memb0/`；BEL0 formal cache seed **37611**

## 一句话

\[
\boxed{
\texttt{winner = none}
\quad
H^*_{\mathrm{retr}}=H^*_{B0}=32
\quad
H^*_{\mathrm{surf}}=16
}
\]

**Selection 没有赢过 no-memory chain；integration 更差。**  
不开 fresh confirmation。

## P0 表

| Branch | \(H^*\) | H64 med | H64 P90 | pos H64 | med \(G_n\) | \(P(G_n>0)\) | applied |
|--------|--------:|--------:|--------:|--------:|------------:|-------------:|--------:|
| **B0** none | **32** | 20.8° | 47.3° | 10.9 cm | — | — | 0 |
| B1 retrieval | 32 | 22.5° | **45.0°** | 11.3 cm | 0.0° | 0.13 | 114/200 |
| B2 surface | 16 | 32.8° | 69.3° | 10.8 cm | 0.0° | 0.11 | 111/200 |

Winner 要求 \(H^*>H^*_{B0}\)。B1 并列 32 **不算赢**；B2 在 H=32 打断 incumbent。

## B1 diagnostic：selected age

Applied 的 \(A=t-k^*\)：

| A | 1 | 2 | 4 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|---:|---:|
| P | **0.51** | 0.32 | 0.11 | 0.04 | 0.009 | 0.009 |

几乎总是最近 1–2 步。**没有证据**支持 multi-scale historical reference；\(q\)-argmax 退化成近邻 overlap。

## B2 diagnostic：contamination

\(D_M\) final median **0.57 mm**（很小），问题不是 voxel 发散，而是 **用 drifted \(\hat T_{0,t}\) 对齐后再把 surface 当 correction source**，在 H=32 把仍过门的 chain 打坏（与 RAB0 长基线同一类负结果）。

## Pattern

```text
pattern = memory_breadth_selection_complete
scientific_result = false
winner = none
next_frontier = [stop_world_perception_local_dfs, reopen_wam_multirate]
```

## 架构剪枝

RAB0 已经说明：re-anchor 需要比 chain 更可靠的信息源。  
MEMB0 说明：在这个 regime 里，

- **从历史里选一帧**（且几乎总是最近帧）不够；
- **融合成 persistent surface** 更不够。

简单 memory correction **仍不优于纯 relative backbone**（\(H^*=32\)，H64 仍 FAIL）。  
按冻结规则：**停止 world-perception 局部 DFS**，不围着 retrieval/surface 再修 voxel 或 admission。下一层应回到更高一级（object-state 需求 / 多速率 WAM / task necessity），需新预注册。
