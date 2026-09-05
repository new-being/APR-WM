# RTWX-O0RA0 预注册 — Frozen Winner Fresh Confirmation

日期：2026-09-01  
状态：**已冻结 / NOT RUN**（RAB0 `winner=none`，本格不开）  
依赖：RAB0 P0 winner hash；SEQ0 motion；seed **37612**  
**禁止**：改 winner；跑 P0 第二名；adaptive trigger；ICP 调参；GT reset

## 数据

| 项 | 值 |
|----|-----|
| seed | **37612**（\(\neq\) 37611） |
| \(N\) | 100 |
| \(T\) | 64 |

只跑 **B0** + **一个** frozen winner \((B^*,K^*)\)。

## Formal 成功

\(H^*_{winner}=64\) 且 \(H^*_{winner}>H^*_{B0}\)：

- B1 → `periodic_absolute_reanchor_supported`
- B2 → `initial_anchor_relocalization_supported`
- B3 → `rolling_keyframe_reanchor_supported`

若 fresh 上 B0 也 \(H^*=64\)：`reanchor_not_required_on_fresh_split`（不宣称 necessary）。

否则：`non_gt_reanchor_insufficient`。

## 实现

`rtwx_o0ra0.py`；`runs/rtwx_o0ra0/`
