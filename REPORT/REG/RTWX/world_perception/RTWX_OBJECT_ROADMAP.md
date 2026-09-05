# RTWX Object/World 路线（冻结顺序）

日期：2026-08-29  
状态：**设计冻结**  
依赖：X0EH1/X0EH2 robot-state capacity PASS；X0RGB/X0RGB1 = RGB→\(s^R\) **CLOSED（负结果）**。

## 关闭的子线

\[
\cancel{RGB\rightarrow(q,\dot q)}
\]

第三人称 RGB 在已测仪器下不足以恢复 12-DoF 本体状态。部署上 \(s^R\) 来自 proprioception。不继续 ViT/keypoint 救这条线。

## 已验证

\[
s^R=(q,\dot q),\quad a=q^{\mathrm{tar}},\quad K=4,\quad \text{M2 }R_P^{K=4}=0.988
\]

## 目标架构（不在本文件实现）

持久粗世界 + 任务/因果/风险驱动的局部高精度 + RGB 校正/按需细化 + 按对象分配算力。  
\(z_g,z^p\) 不进入物理状态；\(z^{\mathrm{res}}\) 很晚才考虑。

## 实验顺序（一格一问）

| 阶段 | 唯一问题 | 解锁 |
|------|----------|------|
| **O0** | RGB 能否恢复显式 \(s^O\)？ | object pose instrument |
| **O1** | persistent filter 是否优于逐帧重建？ | RGB as correction |
| **D0** | \((s^R,s^O,a)\to s^{O'}\) 短时预测？ | object WM |
| **I0** | 显式 interaction 是否压缩容量？ | structured world dynamics |
| **A0/A1** | task vs causal attention | selective state |
| **C0** | 选择是否变成算力？ | adaptive compute |
| **C1/S0** | 观测/风险触发粗→细？ | adaptive fidelity |
| **U0** | 不确定性是否值得主动观测？ | active evidence |

禁止把 RGB + slots + attention + compute 塞进同一格。

**O0G6A = RAN / `global_geometry_ambiguous`**（倾倒可分、杯轴 yaw 退化）。  
**O0A0 = RAN / `appearance_yaw_generalization_failure`**（单帧 RGB/depth/RGB-D 均为 chance；train 亦为 chance）。  
**O0T0 = PREREGISTERED / NOT RUN / DEPRIORITIZED**（仅当任务明确需要 yaw 时再考虑）。  
**O0E0 = RAN / `observation_support_failure`**（natural 分布不够格检验；**不是** \(RGBD\not\to(p,n)\)；**B2 UNTESTED**）。  
**O0E0P0 = RAN / `controlled_pose_observation_qualified`**（G0a=1.0；G0b=0.60；G0c=8/8；**B2 仍 UNTESTED**）。  
**O0E0R3 = RAN / `reference_coupled_failure`**（instrument 0.15°；formal 22.4°）。  
**O0E0R4 = RAN / `multiview_reference_insufficient`**（\(K\le8\) naive union 仍 ~20°；center–\(\lambda\) 诊断：3.5 cm → 28°，\(\lambda=0.5\) → 14.6°）。  
**O0E0R5 = RAN / `visibility_reference_joint_failure`**（G\_I1 **14.6°** PASS；formal joint **16.6°** FAIL）。  
**O0E0R6 = RAN / `joint_inference_insufficient`**（A1 \(n^{(1)}\) P90 **129°→33°**）。  
**O0REL0A = RAN / `overlap_support_hypothesis_supported`**（C0 overlap-collapse 机制确认；REL0 formal 不变）。  
**O0SEQ0**：**RAN / `relative_sequence_supported`（\(H^*=16\)）** — [`../../../REP/RTWX/world_perception/RTWX0O0SEQ0_REPORT.md`](../../../REP/RTWX/world_perception/RTWX0O0SEQ0_REPORT.md)  
**O0HYB0 = RAN / `hybrid_state_update_insufficient`**（relative-only 四 regime 全 PASS；router 不再主路径）。  
**O0REL0 = RAN / `ego_motion_compensation_failure`**（C1/C2 强 PASS；C0 L2 FAIL）。  
**O0E0R7 = PAUSED**（persistent surface；frontier 候选）。  
**O0E1 / O1 LOCKED**。

## Beam search（2026-08-31）

单帧 absolute reference 族达 \(D_{\max}=2\)，**暂停 DFS**。优先级 \(\propto I(H)\times U(H)/C(H)\)：

| Beam | Probe | 状态 |
|------|-------|------|
## WAM / 多速率（2026-09-01）

World-perception 局部 DFS **STOP**。相对主干冻结。  
假说拆开：先 \(H_A\)（action reuse），后 \(H_S\)（state bandwidth）；**禁止**用当前 strided ICP 比较 \(K_s^*\) vs \(K_a^*\)。

```text
WP local DFS STOP
→ relative backbone frozen
→ AC0 CLOSED (covariate_shift_suspected; winner B0)
→ AC1-D0 state_support_metric_invalid (do not skip G1)
→ AC1-D1 support_geometry_unqualified (winner=none)
→ repair support instrument before any DAgger/R0
→ re-run MR0-P0 G3 only after competent policy
→ MR0-A only if action_chunk_qualified
→ if Ka*>=4, then consider a legal state-rate probe
```

| 格 | 状态 |
|----|------|
| AC0 | RAN / `covariate_shift_suspected`；winner B0；bakeoff 96/96 success=0 |
| AC1-D0 | RAN / `state_support_metric_invalid` |
| AC1-D1 | RAN / `support_geometry_unqualified`；winner=none |
| MR0 | RAN / infrastructure `action_chunk_contract_failure` |
| MR0 | RAN / infrastructure `action_chunk_contract_failure` |
| MR0-P0 | RAN / 同 contract；权重已拷、**G3 未重跑**；`unlocks_mr0_a=false` |
| MR0-A | BLOCKED |
| 十字 \(K_s\times K_a\) | **关闭**（G0-S 不合格） |
| 2 | relative-first backbone（no-reset \(H^*=32\)） | **locked** |
| 3 | periodic re-anchor / retrieval / persistent surface | **CLOSED**（RAB0+MEMB0 winner=none） |  
**O0Q\* = SIDE / FROZEN**（无 causal quotient 结论）。O0Q0R1 / O0Q1 LOCKED。O0G6R / O0G5R / O0C2 LOCKED。  
SYM-X0–X2 RAN；表示合同 [`../../../SYMX/SYMX_MECHANISM.md`](../../../SYMX/SYMX_MECHANISM.md)；**X3 LOCKED**。


