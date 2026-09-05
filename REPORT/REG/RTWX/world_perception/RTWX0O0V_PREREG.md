# RTWX-O0V 预注册 — Robust Visual Coverage Contract

日期：2026-08-29  
状态：**已冻结（infrastructure；无 pose claim；无训练）**  
依赖：O0G2 = `coverage_failure`（head-only vis=0.902；G1–G4 条件性通过）  
**禁止**：放宽 O0G2 G0；再采 seed 赌 `head_camera` 回 0.95；改 O0R/O0G/O0G1b；训练分割/pose；解锁 O0G2R/O0C/O1。

## 科学问题

> 哪个**固定** observation contract 能跨 fresh episode / seed 稳定看到目标杯？

不训练；不选“谁在 test 上 pose 更好”的相机。候选来自 O0D2 FOV screening（head / observer / world 曾达 FOV=1.0），**预注册冻结**双视角：

\[
\boxed{\texttt{head\_camera}+\texttt{observer\_camera}}
\]

## 冻结合同

| 项 | 值 |
|----|------|
| 任务 | `place_empty_cup` |
| B0 | `head_camera` only（负对照；预期不稳） |
| B1 | **`head` ∨ `observer`**（主） |
| seeds | **25601, 25602, 25603**（3 个 fully fresh；**禁止**事后换 seed） |
| 每 seed | 24 episodes × 120 steps |
| 可见定义 | 与 O0G2 一致：投影 in_FOV ∧（mask 面积≥50 或 id@uv） |
| \(V_t^{\mathrm{any}}\) | \(V_t^{\mathrm{head}}\lor V_t^{\mathrm{observer}}\) |

## Gates（B1）

| Gate | 条件 |
|------|------|
| Aggregate | \(P(V^{\mathrm{any}})\ge 0.98\)（三 seed 合并） |
| Per-seed | 每个 seed \(P_s(V^{\mathrm{any}})\ge 0.95\) |

B0 仅报告（描述 head-only 跨 seed 不稳）；**不**用 B0 PASS 解锁下游。

## Patterns

| Pattern | 条件 |
|---------|------|
| `dual_view_coverage_supported` | B1 aggregate ∧ 全部 per-seed |
| `dual_view_coverage_insufficient` | ¬(上) |
| （描述）`head_only_unstable` | 若任一 seed 上 head \(P_{\mathrm{vis}}<0.95\)（预期真；不单独作主 pattern） |

## 解锁

- PASS → 允许预注册 **O0G2R**（multi-view geometry-mediated confirmation）  
- **O0G2 不改**；O0C / O1 LOCKED until 后续 confirmation

## Explicit 不做

- 不 architecture / camera 搜索  
- 不把 world_camera 事后塞进主合同（若 B1 FAIL 再新预注册）  
- 不 claim \(RGBD\to s^O\)
