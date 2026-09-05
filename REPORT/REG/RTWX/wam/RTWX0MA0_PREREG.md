# RTWX-MA0 预注册 — Mature-Stack Alignment Probe

日期：2026-09-01  
状态：**已冻结**  
MA = Mature Alignment（成熟方案对齐）  
依赖：现有 `demo_clean` / `place_empty_cup` / `aloha_agilex` 的 50 条 XPolicyLab 轨迹；**禁止**自建 dataset、自写 ACT rollout、把 oracle structured state 塞进阳性对照。

这首先是 **infrastructure / scientific instrument qualification**，不是 APR-WM 架构实验。MA1 在 MA0 cup PASS 之前 **不跑**。

## 科学问题

官方成熟链

\[
\text{raw/converted RoboTwin demo}
\rightarrow
\text{官方 converter / ACT process\_data}
\rightarrow
\text{官方 ACT}
\rightarrow
\text{官方 XPolicyLab deploy}
\rightarrow
\text{RoboTwin simulator}
\]

在当前环境和 `place_empty_cup` 上是否具有非零闭环能力？

## 禁止

- 我们的 alignment + 我们的 \(K_a=1\) + 只借 ACT 网络
- 把官方 `num_epochs=6000` 改成 3000 / 100 epoch
- 预测 chunk 后只执行第一步再重规划
- 一开始三任务一起训/评
- cup FAIL 后立刻开 cabinet / stamp / MA1

## 冻结栈（阳性对照）

| 项 | 值 |
|----|----|
| 任务 | `place_empty_cup` only（P0/P1） |
| 数据 | 已有 50 条 `demo_clean` converted HDF5；**不**自己重建标签 |
| embodiment 目录 | `aloha_agilex`（与现有数据路径一致；\(d_a=14\)） |
| ACT `action_type` | `joint` |
| ACT 训练 | 官方 `policy/ACT/train.sh`：`num_epochs=6000`，`chunk_size=50` |
| 评估 | 官方 `eval.sh` → XPolicyLab `get_action()` → 官方 `eval_policy_xpolicylab.py` 逐个 `take_action` |
| cup 门 | \(N=16\) fresh seeds；\(G0: S_{\rm cup}\ge 0.20\)（至少 4/16） |

Converter 默认（无 `--same-index-action`）：

\[
action[t]=\texttt{joint\_action}[t+1]
\quad\text{（} \texttt{action\_alignment}=\texttt{next\_state\_as\_action} \text{）}
\]

## 阶段

### MA0-P0 — 数据合同审计，不训练

写 `MA0_conversion_audit.json`：source、`action_alignment`（metadata 或数值推断）、frequency、\(d_a\)、\(T_{\rm raw}\) vs \(T_{\rm converted}\)。

若无 legacy raw：`process_data_xpolicylab.py` **不可**原样重跑；审计现有 converted 文件，并记录 `converter_rerun=false`。

### MA0-P1 — 官方 ACT 原样训练 + 官方 deploy

RGB + proprioception → ACT → action chunk。  
cup \(N=16\)。FAIL → `official_act_stack_unqualified`，STOP。  
若 XPolicyLab 未检出 → `xpolicylab_checkout_missing`（仪器未就绪，不是 ACT 失败）。

### MA0-C0 — 仅 cup PASS 后

三任务各 \(N=16\)，只跑官方 ACT。  
门：\(S_{\rm pooled}\ge0.40\)，\(S_{\rm each}\ge0.20\)，\(Safety\le0.20\)。  
PASS → `mature_stack_qualified`。

## MA1（本格不执行）

每次只换 alignment / chunk execution / history / 输入模态之一。
