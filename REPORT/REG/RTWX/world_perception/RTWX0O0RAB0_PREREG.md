# RTWX-O0RAB0 预注册 — Re-anchor Breadth Bakeoff (P0)

日期：2026-09-01  
状态：**已冻结 / RAN**（P0；`winner=none`；报告：`REPORT/REP/RTWX/world_perception/RTWX0O0RAB0_REPORT.md`）  
依赖：BEL0 formal cache seed **37611**；SEQ0/REL0 frozen ICP；R6 A1  
**禁止**：adaptive trigger；belief \(\tau\)；history retrieval；persistent surface；pose graph；ICP 调参；stack 机制

## 问题

\[
H_{\rm rab}:\text{ 不用 GT，哪一种最简单的 periodic re-anchor 来源真正有效？}
\]

\(\Delta\mathrm{mechanism}=1\) 广搜。`scientific_result=false`。

## Incumbent

near-upright anchor \(\to\) adjacent relative chain，然后 **先 adjacent update，再 periodic correction**。

## Cells（7）

| Branch | K |
|--------|---|
| B0 no reset | — |
| B1 frozen A1 absolute | 16, 32 |
| B2 initial-anchor ICP | 16, 32 |
| B3 rolling keyframe ICP | 16, 32 |

ICP `valid=false` → no-op（保留 \(s^-\)）；B3 invalid **不更新** keyframe。

## Winner（lexicographic，冻结）

1. 最大 \(H^*\)（checkpoint \(\{1,2,4,8,16,32,64\}\)，BEL0 gate）
2. 最小 P90 \(e_n(H=64)\)
3. 最小 median \(e_n(H=64)\)
4. 更大 \(K\)（32>16）
5. 更简单：B1 > B2 > B3

若所有 \(H^*\le H^*_{B0}\)：`winner=none`，**不跑 RA0**，frontier = history_retrieval / persistent_surface。

## Pattern

`reanchor_breadth_selection_complete`（非科学 PASS）

## 实现

`rtwx_o0rab0.py`；`geometry/periodic_reanchor.py`；`runs/rtwx_o0rab0/`
