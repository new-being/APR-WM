# RTWX-O0Q0R0 报告 — Native Counterfactual Instrument Closure

日期：2026-08-30  
状态：**正式冻结** `counterfactual_restore_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0_PREREG.md`  
产物：`runs/rtwx_o0q0r0/{summary.json,run.json}`  
Host：`place_empty_cup` / `021_cup`；官方 demo seeds **0/1/2**；**无 RGB**；无瞬移进接触  
**不对 yaw 下科学结论。** O0 target 未改。O0Q0R1 / O0Q1 / O1 LOCKED。

## 一句话

入池 snapshot 的 identity 未闭合（\(D_H(I)_{\max}=10.27\)）。  
P1 仪器是合格的；P2 根本没走到；P3 虽来自对齐 seed 的 hdf5，restore 仍不闭合。R1 不开。

\[
\boxed{\texttt{counterfactual\_restore\_failure}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-restore | 已入池 \(D_H(I)<0.02\) | **FAIL**（max **10.27**，n=8） |
| G-native | P0–P3 各 ≥3；P2/P3 前向 | **FAIL**（P2=0，P3=2） |
| G1-excite | 每档 probe \(P(D_H(R_y90)>\tau)\ge0.80\) | **FAIL**（P2 无样本；其余档的 probe 本身能亮） |

未把 P3 删掉后再谈合格。

## 分 regime（诊断，非 yaw 结论）

| Regime | n | \(D_H(I)\) | probe \(D_H(R_y90)\) | 来源 |
|--------|--:|-----------:|---------------------:|------|
| P0 | 3 | 0.12 / 0.25 / 0.24 | 1.81 / 1.61 / 1.54 | free-flight IC |
| **P1** | 3 | **0.000** | **0.15 / 0.16 / 0.16** | table reset |
| P2 | 0 | — | — | 选择规则从未触发 |
| P3 | 2 | 7.58 / 10.27 | 0.55 / 9.37 | seed-matched hdf5 |

`play_once` / `grasp_actor` 仍因 `target_pose is None` 失败。官方 hdf5 已与 `seed.txt` 对齐重放。

## 刚体审计（非门）

\[
m=0.010,\quad
I_B\approx(1.30,1.14,1.29)\times10^{-4},\quad
\frac{|I_x-I_z|}{I_x}=0.78\%,
\quad g I_B g^\top = I_B\ (R_y90).
\]

COM 几乎在杯轴上（\(c_x,c_z\sim0\)，\(c_y=0.041\)）。  
**可见外形与 simulator 惯量都近似轴对称。** 只解释；不作 yaw claim。

## 读数

1. **P1 闭合。** 桌面 hold 的 identity 是 0；各向异性摩擦 probe 把 \(D_H(R_y90)\) 拉到 \(\sim0.15>\tau\)。上一格「桌上 B2 惯量激不起来」是机制错配，不是桌面仪器不可用。
2. **P0 的 H=8 `take_action` 太长。** 抬高 12 cm 后杯子会在 horizon 内落地，identity 0.12–0.25 不是「空中刚体不可复现」的干净证据。
3. **P2 不可用。** 对齐 hdf5 也从未出现「开爪且 EE–cup∈[0.02,0.12]」且剩余 H=8。官方文件只有机器人 qpos，没有物体 pose；重放仍不是可靠的接近段。
4. **P3 入池但 G0 失败。** 爪闭 + 接触被规则选中；\(D_H(I)\sim8\)–\(10\)。接触 restore 仍不闭合。按预注册记 restore_failure，不删档。
5. **O0Q0R1 仍 LOCKED。** 本格没有资格跑 23 格 nominal yaw。

## Pattern

```text
pattern = counterfactual_restore_failure
G0_restore = FAIL (D_H(I)_max=10.27)
G_native = FAIL (P2=0, P3=2)
G1_excite = FAIL (P2 missing; P0/P1/P3 probes can fire)
I_x ≈ I_z (rel 0.78%); commutes_Ry90 = true
unlocks_o0q0r1_prereg = false
o0_target_remains_full_TR = true
```
