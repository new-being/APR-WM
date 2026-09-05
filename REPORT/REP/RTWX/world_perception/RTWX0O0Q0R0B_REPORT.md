# RTWX-O0Q0R0B 报告 — Pre-contact Branch Counterfactual Closure

日期：2026-08-30  
状态：**正式冻结** `precontact_branch_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0B_PREREG.md`  
产物：`runs/rtwx_o0q0r0b/{summary.json,run.json}`  
Seeds 0/1/2；**无接触 restore**；无 23 格 yaw  
**O0 target 未改。** O0Q0R1 LOCKED。

## 一句话

预接触根上的 open-loop 重放已经闭合；P0 的 ballistic horizon 也闭合。  
失败在 **P2/P3 从未被 \(A^*\) 走进去**（mplib IK 失败，\(A^*\) 只有闭爪/hold，\(\min d\sim0.27\)）。

\[
\boxed{\texttt{precontact\_branch\_failure}}
\]

## Gates

| Gate | 结果 |
|------|------|
| G0-root | **PASS**（P0 / P1 根 / \(A^*\) 全部 \(D_H(I)\sim 2\times10^{-4}\)） |
| G1-event | **FAIL**（P2=0，P3=0） |
| G2-repro | **FAIL**（无 event） |
| G3-excite | **FAIL**（P2/P3 无样本；P0 probe \(0.036<\tau\)；P1 **0.16** 仍过） |

## 主表

| 项 | 值 |
|----|-----|
| \(H_{P0}\) | **6**（\(t_{\mathrm{hit}}=\sqrt{2\cdot0.12/g}\)，\(\Delta t_a=0.02\)，margin 0.03） |
| P0 \(D_H(I)\) | **0.00016**；窗口内接触 **false** |
| P1 \(D_H(I)\) | **0.00018** |
| \(A^*\) \(D_H(I)\) | **0.00014** |
| \(\|A^*\|\) | 14（8 close + 6 hold；**0** 次成功 IK） |
| \(\min d(\mathrm{EE},\mathrm{cup})\) | 0.26 / 0.33 / 0.27 |
| P0 probe \(D_H(R_y90)\) | 0.036 \(<\tau\) |
| P1 probe \(D_H(R_y90)\) | **0.16** \(>\tau\) |

## 读数

1. **R0 的 P0 失败确是 horizon。** 按弹道合同改成 \(H=6\) 个 scene-step 组后，identity 归零且全程无桌面接触。旧格 \(D_H\sim0.12\)–\(0.25\) 是落地污染。
2. **不要 restore 接触态这条路是对的。** 同一预接触根上重放冻结 \(A^*\)，\(D_H(I)\sim10^{-4}\)。
3. **P2/P3 仍 unavailable。** 冻结 scripted 的 IK 全部失败，手臂没走近杯子。不是删档，是 G1 失败。
4. **P0 短窗 + \(I_x\approx I_z\)：** 即使 \(I_x\leftarrow1.5 I_z\)，\(D_H=0.036\) 仍低于 \(\tau\)。仪器闭合前，旧 P0 nominal \(D_H\sim1\) 更不能读成惯量不对称。
5. **O0Q0R1 仍 LOCKED。**

## Pattern

```text
pattern = precontact_branch_failure
G0_root = PASS
G1_event = FAIL (P2=0, P3=0, min_d~0.27)
G2_repro = FAIL
G3_excite = FAIL (P1 ok; P0 0.036<τ)
unlocks_o0q0r1_prereg = false
o0_target_remains_full_TR = true
```
