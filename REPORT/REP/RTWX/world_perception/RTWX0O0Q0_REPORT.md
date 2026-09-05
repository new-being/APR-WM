# RTWX-O0Q0 报告 — Task-Object Causal Quotient Audit

日期：2026-08-30  
状态：**正式冻结** `counterfactual_instrument_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0Q0_PREREG.md`  
产物：`runs/rtwx_o0q0/{summary.json,run.json}`  
Host：`place_empty_cup` / `021_cup`；seeds 45101/02/03；**无 RGB**；\(\tau=0.05\)  
**O0 target 未改。** O0Q1 / O1 / O0G6R / O0C2 / SYM-X3 LOCKED。

## 一句话

Identity 重放没有闭合（\(D_H(I)_{\max}=237\)），本格按预注册 **STOP**。  
不能据此把 O0 改成 \((p,n)\)，也不能宣称 yaw 已被证明是或不是因果 gauge。

\[
\boxed{\texttt{counterfactual\_instrument\_failure}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-integrity | \(D_H(I)<0.02\) | **FAIL**（max **236.6**，n=24） |
| G1-excite | 每 regime ≥3 snap；B2 \(D_H(R_y90)>\tau\) 且拒绝率 ≥0.80 | **FAIL**（覆盖有；B2 Ry90 min=**0.004**；拒绝率 0.75） |
| G2-nominal | B1 \(P(D_H\le\tau)\ge0.90\) | **FAIL**（0.25）— **不作科学结论**（G0 已 STOP） |
| G3-regime | 每 regime B1 \(P\ge0.80\) | **FAIL**（仅 P1 过） |

`play_once` 因 CuRobo 缺失失败；P2/P3 用 EE 邻域放置 / 闭爪 hold。

## 分 regime（诊断，非门）

| Regime | B1 median \(D_H\) | B1 \(P_{\mathrm{acc}}\) | B2 \(D_H(R_y90)\) |
|--------|------------------:|-------------------------:|------------------:|
| **P0** 空中 | **1.13** | **0.00** | 1.55 |
| **P1** 桌面 | **0.005** | **1.00** | **0.004** |
| P2 EE 邻域 | 78.5 | 0.00 | 98.9 |
| P3 闭爪 | 83.8 | 0.00 | 128.9 |

## 读数

1. **G0 失败的位置是 P2/P3。** 把杯子瞬移到 EE 附近，restore+replay 不是确定性反事实；\(D_H(I)\) 可以到几百。这两档不能用来判对称。
2. **桌上（P1）仪器是安静的**：23 个 yaw 全接受，B1 与 B2 都 \(\sim 0.005\)。静止接触 **激励不到** 惯量破缺——B2 在 P1 上不合格。
3. **空中（P0）所有非平凡 yaw 的 \(D_H\sim 1\gg\tau\)**，包括 nominal B1。这与「真实 cup 自由运动并不把 yaw 当 gauge」**相容**，但 G0 已 STOP，**不得**写成 G2 结论。
4. **O0G6A 仍然只回答了形状**。本格没有资格把视觉退化升级成 \(s^{O,Q}=(p,n)\)。

## Pattern

```text
pattern = counterfactual_instrument_failure
G0_integrity = FAIL (D_H(I)_max=236.6)
G1_excite = FAIL (B2 Ry90 min=0.004; P1 not excited)
G2_nominal = FAIL (not interpretable)
G3_regime = FAIL (only P1)
unlocks_o0q1_prereg = false
o0_target_remains_full_TR = true
```

## 解锁

无。下一格是仪器修复 **O0Q0R0**（native snapshot，禁止瞬移），不是重跑本格、也不是 O0Q1。
