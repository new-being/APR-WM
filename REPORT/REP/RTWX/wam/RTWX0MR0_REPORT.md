# RTWX-MR0 报告 — Multi-Rate State–Action Resampling Probe

日期：2026-09-01  
状态：**RAN / STOP**；pattern = `action_chunk_contract_failure`  
预注册：`REPORT/REG/RTWX/RTWX0MR0_PREREG.md`  
产物：`runs/rtwx_mr0/`；G0-S 用 BEL0 formal cache seed **37611**；ICP hash `0bcce33f…da00d82`

## 一句话

\[
\boxed{
H_a=0
\quad\Rightarrow\quad
\texttt{action\_chunk\_contract\_failure}
}
\]

**没有**冻结的 action-chunk policy（\(H_a\ge 8\)）。按合同 **禁止** hold-last-qpos 假装 \(K_a\in\{2,4,8\}\)。十字 7 cell 的 task-success 实验 **未跑**（不是负的 asymmetry 结论）。

## G0-A

| 检查 | 结果 |
|------|------|
| 本机 frozen chunk checkpoint | **无**（`policy_path` 空；`/root/autodl-tmp` 无 `.pt/.ckpt` WAM 权重） |
| \(H_a\) | **0** |
| 单步 qpos hold 作为 \(K_a=8\) | **禁止**（那是 target hold，不是 plan resampling） |

因此 \(K_s^*,K_a^*\) **未定义**。下列 pattern **均未检验**：

```text
multirate_asymmetry_supported
multirate_asymmetry_not_supported
joint_low_rate_supported
```

## G0-S（compute-skipping stride ICP）

20 条 BEL0 序列；frozen ICP；\(P_t\leftrightarrow P_{t+K_s}\)（不是 adjacent 连跑）。阈值 valid \(\ge 0.80\)。

| \(K_s\) | valid rate | instrument_ok |
|--------:|-----------:|:-------------:|
| 2 | 0.772 | **false** |
| 4 | 0.753 | **false** |
| 8 | 0.725 | **false** |

\(K_s=2\) 已低于 0.80。即使将来有 chunk policy，**当前 stride registration 也不能用来解释“state 必须高频”**——格会是 instrument unsupported，不是 science gate。

## 已实现但未部署到任务闭环的合同

- `WorldStateClock`：proprio 在 loop 外每 tick 新鲜；\(s^O\) 只在 \(0,K_s,\ldots\) 更新；无 dynamics extrapolate。
- `ActionChunkBuffer`：chunk 短于 stride → hard fail，无 hidden replan。
- 十字 cell、三任务名、\(N=50\) CRN、non-inferiority 阈值均冻结在 `MR0_CONFIG`。
- CLI：`rtwx-mr0`；不开放 strides/tasks/阈值。

## Pattern

```text
pattern = action_chunk_contract_failure
scientific_result = true
Ks_star = null
Ka_star = null
episodes_run = 0
```

## 下一步（不改 perception memory）

1. 接入 **本来就输出 \(H_a\ge 8\) chunk** 的冻结 WAM/policy（deterministic / DDIM）。  
2. 再开 MR0：**先**把 G0-S 做成 instrument-ok，或把 S-sweep 明确标为 unsupported 后只解释 A-sweep。  
3. **不要**用单步策略 + 保持上一个 qpos target 去填 1050 episodes。

MEMB0 的停止 WP DFS 仍然有效。MR0 目前卡在 **action 表示合同**，不是卡在再加一个 perception 模块。
