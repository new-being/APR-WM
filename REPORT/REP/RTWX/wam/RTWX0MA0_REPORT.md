# RTWX-MA0 报告 — Mature-Stack Alignment Probe

日期：2026-09-05  
状态：**RAN / P1 cup G0 FAIL**  
pattern = `official_act_stack_unqualified`  
预注册：`REPORT/REG/RTWX/wam/RTWX0MA0_PREREG.md`  
产物：`runs/rtwx_ma0/`  
官方 `process_data.sh`：50/50。官方 `train.sh`：6000/6000。官方 `eval.sh`：**3/16 = 18.8%**。**未**开 MA0-C0 / MA1。

## 一句话

\[
\boxed{S_{\rm cup}=3/16=0.188 < G0=0.20}
\]

官方 ACT 整栈**可以非零闭环**（3 次成功），但未过冻结的 cup 门 \(4/16\)。按预注册 **不扩 cabinet/stamp**。这已经把「本机 RoboTwin 完全做不了」排除掉；剩下的是门槛差一点，以及环境/数据是否仍与公开 leaderboard 参考不一致。

## 冻结合同

- 任务：仅 `place_empty_cup`
- 数据：50 条 `demo_clean` converted HDF5；未自建标签
- ACT：`num_epochs=6000`，`chunk_size=50`，`joint`，`aloha_agilex`，\(d_a=14\)
- 评估：官方 `get_action()` 后 chunk 内逐个 `take_action`
- cup 门：\(N=16\)，\(S\ge0.20\)（4/16）

## MA0-P0

见 `runs/rtwx_ma0/P0/MA0_conversion_audit.json`。  
`action_alignment` 数值为 **`next_state_as_action`**（\(e_{\rm next}=0\)）；`frequency=15`；\(d_a=14\)。无 raw，converter 未重跑。

## MA0-P1

| 步 | 结果 |
|----|------|
| XPolicyLab | `XPolicyLab-main.zip`（`a9ccf8d`，新于 pin `c37109c`） |
| cuRobo | v0.7.8，`sm_120`；`warp-lang==1.12.0` |
| embodiment yml | 官方 `update_embodiment_config_path.py` |
| train | `policy_last.ckpt`，6000 epoch |
| eval | `3/16`，exit 0 |

成功累计出现在 seed **100007 / 100010 / 100015**。若干 seed 被 expert_check 跳过（如 100003、100006）。

产物：`runs/rtwx_ma0/P1/eval_cup.json`

## 资格

```text
unlocks_ma0_c0 = false
unlocks_ma1 = false
nonzero_closed_loop = true
```

下一步按预注册：先查数据版本 / RoboTwin·XPolicyLab commit / config / embodiment / camera / `action_type` / adapter，**不要**立刻开 C0 或 MA1 拆变量。若要把 \(G0\) 从 4/16 改成「非零即过」，需要单独改预注册，不能事后改门。
