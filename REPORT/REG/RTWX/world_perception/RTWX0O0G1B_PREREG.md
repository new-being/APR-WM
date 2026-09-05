# RTWX-O0G1b 预注册 — Object Reference-Frame Alignment

日期：2026-08-29  
状态：**已冻结（诊断格；不改 O0G 门限）**  
依赖：O0G = `geometry_insufficient`（G0 PASS；G1 FAIL 主因 surface≠pose origin）  
**禁止**：放宽 O0G 5 cm 门；拟合 O0G test bias \(+5.11\,\mathrm{cm}\)；改 O0R；解锁 O1；训练新 backbone；G2 直至本格 PASS。

## 科学问题

> 若显式处理 object pose origin 与 visible-surface centroid 的**固定几何偏移**（已知资产先验，非事后 bias），RGB-D 几何位置能否恢复到可用精度？

\[
\hat p_B^{\mathrm{pose}}
=
\hat p_B^{\mathrm{surf}}
-
R_{BO}\,\delta_O
\]

\(\delta_O\) 定义在 **object frame**，经 \(R_{BO}\) 转到 world。禁止直接减 world-\([0,0,5.1\,\mathrm{cm}]\)（除非已证始终竖直且与 object-frame 修正数值等价——仅作 secondary audit）。

## 冻结合同

| 项 | 值 |
|----|------|
| 任务 / 相机 | `place_empty_cup` / **`head_camera`** |
| cup | `021_cup` **model_id=0**（与任务一致） |
| seed | **23601** |
| collect | 8×32；oracle mask + Position→world centroid（同 O0G G1 pipeline） |
| \(\delta_O\) | `model_data0.json` 的 `center * scale`（mesh AABB 中心，object/mesh frame）= **known geometry prior** |
| 学习 | **无**；不训练分割/回归 |

## Baselines

| ID | 定义 |
|----|------|
| **B0** | \(\hat p^{\mathrm{surf}}\) vs \(p^{\mathrm{pose}}\)（原 G1；负对照） |
| **B1** | \(\hat p^{\mathrm{surf}}-R_{BO}\delta_O\) vs \(p^{\mathrm{pose}}\)（主） |

## Gates（不放宽）

| Gate | 条件 |
|------|------|
| Primary | \(\mathrm{median}\|\hat p^{\mathrm{pose}}-p^{\mathrm{pose}}\|\le 5\,\mathrm{cm}\) 且 \(E_p\le 0.20\) |
| Strong | \(\mathrm{median}\le 2\,\mathrm{cm}\)（机制成功） |

## Patterns

| Pattern | 条件 |
|---------|------|
| `geometry_reference_aligned` | B1 primary ∧ strong |
| `geometry_usable_but_coarse` | B1 primary ∧ ¬strong |
| `geometry_alignment_insufficient` | ¬B1 primary |
| `g2_unlocked` | 仅标注：primary PASS 时允许预注册 G2；**本格不跑 G2** |

## Explicit 不做

- 不用 O0G test 的 \(+5.11\,\mathrm{cm}\) 拟合 \(\delta\)
- 不改 O0G / O0R pattern
- 不 claim \(RGB\to s^O\) 最终成功
- O1 LOCKED

## 状态语义备忘（非本格门）

以后 \(s^O\) 应标明 reference frame（pose origin vs surface），避免把 surface centroid 与 actor origin 都叫 “position”。
