# RTWX-MR0-P0 报告 — Action-Chunk Capability Qualification

日期：2026-09-01  
状态：**RAN / STOP**；`scientific_result=false`（instrument）  
pattern = `action_chunk_contract_failure`  
预注册：`REPORT/REG/RTWX/wam/RTWX0MR0P0_PREREG.md`  
产物：`runs/rtwx_mr0_p0/`

## 一句话

\[
\boxed{
H_a=0
\quad
\texttt{action\_chunk\_contract\_failure}
\quad
\neg\texttt{unlocks\_mr0\_a}
}
\]

这是 **infrastructure block**，不是对 \(H_A\)（action temporal reuse）的科学否定。G1/G2/G3 未进入；G3 阈值已预先冻结，没有看结果再改。

## 层级（与感知线的关系）

| 层级 | 状态 |
|------|------|
| near-upright + adjacent relative backbone | **已支持**（SEQ0/BEL0） |
| absolute / re-anchor / retrieval / surface | **已削弱**（HYB0/RAB0/MEMB0） |
| \(H_A\): action plan 可跨 tick 复用 | **未测**（缺 chunk policy） |
| \(H_S\): state 需要更高 refresh | **未测**（G0-S 仪器不合格；本格不修） |

## G0

候选路径（均不存在）：

- `runs/rtwx_mr0_p0/frozen_chunk_policy.pt`
- `runs/frozen_chunk_policy.pt`

`reason=no_frozen_chunk_policy`。  
冻结 spec：\(H_a=8\)，\(d_a=14\)，`joint_position`，`policy_dt_env_ticks=1`，deterministic。  
config hash `bce27086…caa228ff`。

**未**把单步策略 repeat/rollout 成 chunk。TASK-X1 diffusion **仍 LOCKED**。本格 **不训练**。

## G1 / G2 / G3

跳过（无 loadable policy）。G3 合同已冻：\(N=32\)；pooled success \(\ge0.40\)；单任务 \(\ge0.20\)；safety \(\le0.20\)；三任务与 MR0 相同。

## Pattern

```text
pattern = action_chunk_contract_failure
scientific_result = false
unlocks_mr0_a = false
```

## 下一步

单独的 **instrument provisioning**（拿到真正一次 forward 出 \(H_a\ge8\) 的冻结策略）→ 重跑 P0 → 仅当 `action_chunk_qualified` 才开 **MR0-A**。  
不要修 strided ICP 来“凑齐十字实验”。

**2026-09-01 更新：** AC0 已将 interim winner B0 拷到 `runs/rtwx_mr0_p0/frozen_chunk_policy.pt`（见 [`RTWX0AC0_REPORT.md`](RTWX0AC0_REPORT.md)）。**本格 G3 仍未重跑**；上表仍是无 checkpoint 时的合同失败，不是新的 G3 结果。
