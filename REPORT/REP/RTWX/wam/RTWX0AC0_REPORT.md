# RTWX-AC0 阶段性报告 — Explicit-State Action-Chunk Provisioning

日期：2026-09-01  
状态：**RAN / CLOSED**；Stage A 完整；Stage B **\(N=16\)/task 已满**  
pattern = `covariate_shift_suspected`  
winner = **B0**（`complexity_tie_margin`）  
预注册：`REPORT/REG/RTWX/wam/RTWX0AC0_PREREG.md`  
产物：`runs/rtwx_ac0/`  
CLI：`rtwx-ac0`

## 一句话

\[
\boxed{
\text{offline }L_1\approx 0.015,\ e_a(0)\approx 0.010,\ e_a(7)\approx 0.022
\quad\text{但已完成闭环 success}=0
}
\]

\[
\boxed{
\texttt{covariate\_shift\_suspected}
\quad
\neg\texttt{diffusion}
\quad
H_A\text{ 未测}
}
\]

这是 **动作块表示 / 监督映射** 的仪器准备格，不是 perception 格，也不是 MR0-A。  
`state_source = oracle_structured`；`scientific_scope = action_representation_only`。

## 合同（未改）

| 项 | 冻结值 |
|----|--------|
| 任务 | `place_empty_cup`, `put_object_cabinet`, `stamp_seal` |
| 状态 | \(s=[s^R,s^O]\)，\(d_s=57\)（非整份 simulator dump） |
| 动作 | \(H_a=8\)，\(d_a=14\)，`joint_position` |
| 模型 | 同一 256×3 GELU MLP + task embedding \(e_g\in\mathbb R^{16}\) |
| loss | chunk L1；ckpt **只**按 val L1 |
| split | episode 40/5/5 再切窗；尾窗丢弃；seed **50600** |
| process | 因果量 \([d_{EE,O},c_{\rm grasp},d_{O,G},c_{\rm contact},c_{\rm goal}]\)；禁止 \(t/T\) |
| bakeoff | \(K_a=1\) 只执行 \(A_t[0]\)；seeds 从 **50601**（避开 P0 的 38601） |

**未**比较 Transformer / CVAE / diffusion。B1−B0 只多一组固定解析 process 特征。

## Stage A：offline

训练集统计归一化；二值 predicate 保持 0/1。

| Branch | 输入 | \(n_{\theta}\) | val \(L_1\) | \(e_a(0)\) | \(e_a(7)\) | \(D_{\rm pred}\) | \(D_{\rm demo}\) |
|--------|------|---------------:|------------:|-----------:|-----------:|-----------------:|-----------------:|
| B0 | \(s,z_g\) | 179360 | 0.01467 | 0.01024 | 0.02192 | 0.0526 | 0.0508 |
| B1 | \(s,z_g,z^p\) | 180640 | 0.01435 | 0.01005 | 0.02141 | 0.0514 | 0.0508 |

- 远期 \(h=7\) 变差约 2×，**没有崩溃**。
- \(D_{\rm pred}\approx D_{\rm demo}\)：不是 \([a_t,\ldots,a_t]\) repeat-fill。
- B1 的 val L1 略低，幅度远小于闭环差异；**不能**据此宣布 process 有效。

因此 **不是** Pattern A（确定性 chunk 映射 offline 不足）。

## Stage B：闭环 bakeoff（不完整）

预注册 \(3\times 16\times 2=96\) 幕。实际落盘：

| 任务 | B0 | B1 | 成功 |
|------|----:|----:|------|
| place_empty_cup | 16/16 | 16/16 | 0 |
| stamp_seal | 16/16 | 16/16 | 0 |
| put_object_cabinet | 16/16 | 16/16 | 0 |
| **合计** | **48/48** | **48/48** | **0** |

96 幕全部失败；safety = 0。  
部署：每 tick 重新预测 chunk，只执行第一步。Oracle 结构化状态，无 RGB。

**Stage B 协议已闭合。** 字典序 winner 不再是 interim。

## Winner

字典序：pooled → min-task → 低 safety → 低 val L1。  
complexity tie：B1 pooled 优势 \(<5\) pp 且 min-task 不高过 B0 → **B0**。

```text
winner = B0
reason = complexity_tie_margin
checkpoint = runs/rtwx_ac0/B0/best.pt
sha256 = 3ac69210…9794ffc2
config_hash = 90c4fc1a…287c0eb5
copied_to_p0 = true
stage_b_protocol_complete = true
```

选择 **没有 process 的更简单模型**。这不是 “B0 闭环更好”，而是 **process 没有带来可见成功率**。

## Pattern 与解释树

```text
pattern = covariate_shift_suspected
scientific_result = true
stage_b_protocol_complete = true
unlocks_diffusion = false
unlocks_mr0_a = false             # 须重跑 P0-G3；本格未跑
H_A = untested
```

| Pattern | 是否成立 |
|---------|----------|
| A：B0/B1 offline 都差 | **否** |
| B：offline 好、rollout 差 | **是（已完成子集）** |
| C：B1 ≫ B0 | **否** |
| D：B0 已够闭环 | **否**（success=0） |

按预注册：下一步优先考虑 **state perturbation / DAgger 类纠正数据**，**不要**因为 Diffusion Policy 强就跳到 TASK-X1。  
CVAE/diffusion 只在机制更像 **多模态** 而不是 covariate shift 时才解锁。

**不能**写：显式状态已把动作压成可部署监督映射；**不能**写 \(f_s>f_a\)；**不能**写 action chunk 已 qualified。

## MR0-P0

winner 权重已复制到 P0 规定路径。  
**尚未**重跑 P0。G3 门仍冻结：\(N=32\)，pooled \(\ge 0.40\)，单任务 \(\ge 0.20\)，safety \(\le 0.20\)。  
以当前 bakeoff 的 0 成功，**预期** G3 FAIL → `action_chunk_policy_incompetent`，仍不得进 MR0-A。这是预测，不是 G3 结果。

## 产物

```text
runs/rtwx_ac0/
  split_manifest.json
  B0/{best.pt, metrics.json, rollout.json}
  B1/{best.pt, metrics.json, rollout.json}
  selection.json
  summary.json
```

## 下一步（协议顺序）

1. （可选）补跑柜子缺测 8×2 幕，把 Stage B 标 complete。  
2. **原样重跑 MR0-P0**（不放宽 G3）。  
3. G3 FAIL → 按 Pattern B 做纠正数据 / 扰动，仍锁 diffusion。  
4. 仅 `action_chunk_qualified` 才开 MR0-A。
