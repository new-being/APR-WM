# RTWX-AC2 预注册 — Policy-State Sufficiency Probe

日期：2026-09-01  
状态：**已冻结**  
依赖：AC0 `split_manifest.json` 与 train normalizer；**禁止** window 级再划分  
**禁止**：RNN/GRU/LSTM/Transformer/diffusion/process/latent memory；改 Ha；进 MR0；用 offline L1 选 winner；用 expert/qpos 冒充 B2 的 past action

## 问题

当前 \(s_t\) 是否足以决定下一动作 \(a_t\)？先前 \(H_a=8\) chunk 辅助目标是否不必要？  
测 **policy-state sufficiency**，**不**宣称世界 Markov。

## P0（训练前）

held-out（val+test 前 8）原 demo seed + hdf5 `action` 逐 tick `qpos` replay。  
每任务 \(N=8\)。门：\(S_{\rm replay}^{\rm each}\ge 0.875\)（7/8）。  
FAIL → `expert_action_replay_failure`，**停训**。

## 三支（\(H_a=1\)，同一 256×3 GELU MLP，L1）

- B0：\((s_t,z_g)\to a_t\)（\(z_g\) 已在 pack_state）
- B1：\(H_s=4\)，\((s_{t-3:t},z_g)\to a_t\)
- B2：再加 \(a_{t-3:t-1}\)；闭环 past action = **policy 发出的 command**；左填充 \(a_{\rm reset}=q_0\)，不用 \(a_0^E\)

训练 100 epoch，ckpt=val L1。复用 AC0 \(\mu,\sigma\)。

## Stage A

G0 shape 14；G1 finite；G2 bounds；G3 deterministic；G4 历史对齐单测（无 future leakage）。

## Stage B

seeds **53601–53616**，\(N=16\)/task，CRN，\(K_a=1\)。  
资格：pooled \(\ge0.20\)，至少两任务 success>0，safety \(\le0.20\)。  
\(S_{B1}-S_{B0}\ge0.10\) 才 `state_history_utility_supported`；\(S_{B2}-S_{B1}\ge0.10\) 才 `action_history_utility_supported`。  
<5 pp → complexity tie 更简单支；5–10 pp → `inconclusive_small_gain`。  
三支都不够格 → `short_memory_policy_insufficient`。  
B0 明显恢复且历史无增益 → `single_step_output_supported`。

Winner **不**进 MR0。先确认看什么，再谈 chunk 多远。
