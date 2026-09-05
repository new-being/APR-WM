# RTWX-O0E0P0 报告 — Controlled-Pose Perception Qualification

日期：2026-08-31  
状态：**正式冻结** `controlled_pose_observation_qualified`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0P0_PREREG.md`  
产物：`runs/rtwx_o0e0p0/{summary.json,run.json,header.json,cache/{train,val,test}/{obs,gt}.npz}`  
seed **37601**；250/100/200 static snapshots；tilt \(\{0,10,20,35,50\}^\circ\) 等权；head+observer；**无 physics**；**B2 未跑**

## 一句话

受控静态多姿态数据同时闭合 support 与 excitation；现在才有合法检验 O0E0 的 observation distribution。这不是 natural-task claim，也不是 axis estimator 结果。

\[
\boxed{\texttt{controlled\_pose\_observation\_qualified}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0a support | \(P(V^{\mathrm{any}})\ge0.98\) | **PASS**（**1.000**；\(P_h=P_o=1\)；n_fail=0） |
| G0b excitation | \(r_{\mathrm{excite}}\ge0.30\) | **PASS**（**0.600**；median \(\angle\approx20^\circ\)） |
| G0c yaw nuisance | 每 \(\beta\) ≥6/8 octants | **PASS**（每 bin **8/8**） |

B0/B1/B2 **仍 UNTESTED**。

## 生成器（预注册兑现）

| \(\beta\) | test 份额 |
|----------|----------:|
| 0° / 10° / 20° / 35° / 50° | 各 **0.20** |

\(n_{\mathrm{const}}\approx e_z\)。\(P(\beta>15^\circ)=0.60\) 由等权采样保证，非事后挑帧。

## 措辞

- **能说**：受控、多姿态视觉条件下，数据合同足以检验 effective axis。
- **不能说**：自然 `place_empty_cup` 需要或已经支持 axis estimation。
- **不能说**：B2 已从 RGB-D 恢复 axis。

## Pattern

```text
pattern = controlled_pose_observation_qualified
G0a_P_any = 1.000
G0b_r_excite = 0.600
G0c_octants = 8/8 per tilt
B2_not_run = true
unlocks_o0e0_science_on_p0_cache = true
unlocks_o0e1 = false
unlocks_o1 = false
```

解锁：在 **本格 cache** 上重开 O0E0 science（冻结 B0/B1/B2，不改搜索）。O0E1 / O1 LOCKED。
