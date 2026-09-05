# RTWX-O0SEQ0 预注册 — Relative Propagation Horizon Audit

日期：2026-08-31  
状态：**已冻结 / RAN**  
依赖：HYB0/REL0 frozen ICP；R6 A1 descriptive only  
**禁止**：router；absolute fallback；persistent surface；pose graph；ICP 调参

## 主假设

\[
H_{\rm seq}:\text{ near-upright anchor + adjacent relative chain 在非平凡 horizon 内维持 effective state。}
\]

## Branches

- **A0** adjacent chain（formal）
- **A1** direct \(\mathcal P_0\leftrightarrow\mathcal P_H\)（comparator）
- **A2** R6 A1 per \(H\)（descriptive）
- Oracle periodic-reset \(K\in\{2,4,8\}\)（diagnostic only）
- GT-mask chain（20 seq diagnostic only）

## 数据

| 项 | 值 |
|----|-----|
| seed | **37609** |
| \(N_{\rm seq}\) | **100** |
| \(T\) | **16** transitions（17 frames） |

\(\beta_0\in\{0°,2°,5°\}\)；每步 \(\delta\beta\in\{0,5,10\}°\)（0.3/0.4/0.3）；\(\|\delta p\|\in\{0,1,2\}\) cm；camera 小步连续。

## Gates

- L1 observation → `sequence_observation_failure`
- L2 anchor（HYB0 尺度）→ `sequence_anchor_failure`
- Primary outcome **\(H^*\)**：\([1,2,4,8,16]\) 从 1 起连续 PASS 的最大 \(H\)

## Patterns

- `relative_sequence_insufficient`：\(H^*<4\)
- `relative_sequence_supported`：\(H^*\ge4\) + tags `stable_to_16` / `drift_after_h4` / `drift_after_h8`

Unlock：`unlocks_relative_sequence_branch`（若 supported）

## 实现

`rtwx_o0seq0.py`；`geometry/relative_state_chain.py`；`runs/rtwx_o0seq0/`
