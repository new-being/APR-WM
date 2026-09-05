# RTWX-O0MEMB0 预注册 — Reference Memory Breadth Probe (P0)

日期：2026-09-01  
状态：**已冻结 / RAN**（P0；`winner=none`；报告：`REPORT/REP/RTWX/world_perception/RTWX0O0MEMB0_REPORT.md`）  
依赖：BEL0/RAB0 formal cache **37611**；frozen ICP；no-reset \(H^*=32\) incumbent  
**禁止**：stack retrieval+surface；adaptive trigger；改 K；A1 reset；ICP 调参；fresh 科学 claim

## 问题

\[
H_{\rm mem}:\text{ 历史参考应是 selection 还是 integration？}
\]

\(\Delta\mathrm{mechanism}=1\)。`scientific_result=false`。

## Branches

- **B0** no memory（adjacent chain）
- **B1** exponential keyframe retrieval \(\mathcal K=\{t-1,-2,-4,-8,-16,-32\}\)，\(k^*=\arg\max q\)，tie → newer
- **B2** voxel consensus persistent surface（每 voxel 每帧一次 centroid；用当前 branch \(\hat T_{0,t}\) 映到 memory frame）

Correction **仅** \(t\in\{32,64\}\)。branch history **完全隔离**。ICP invalid → no-op。

## Winner

\(H^*>H^*_{B0}\)；lex：\(H^*\) → H64 P90 → H64 median → 更少 applied → B1>B2。

否则 `winner=none`，停止 world-perception 局部 DFS。

## 实现

`rtwx_o0memb0.py`；`geometry/historical_reference.py`；`geometry/persistent_surface_memory.py`
