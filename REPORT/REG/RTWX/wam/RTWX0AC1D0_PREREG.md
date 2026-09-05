# RTWX-AC1-D0 预注册 — Closed-Loop State-Coverage Diagnosis

日期：2026-09-01  
状态：**已冻结**  
依赖：AC0 正式 winner（补齐柜子 bakeoff 后）  
**禁止**：训练；改 policy；调 \(k\)；学 OOD detector；用 38601 / 50601 seeds；解锁 diffusion / AC1-R0（除非本格 G1–G3 全过）

## 问题

\[
\text{闭环 policy 是否很快进入 demonstration 训练状态分布之外，且该偏移明显早于任务失败？}
\]

不训练。support metric **shadow-only**，不改变 \(A_t[0]\)。

## 距离

标准化用 **AC0 train** \(\mu_s,\sigma_s\)。  
每任务单独 kNN，\(k=5\) 冻结。

\[
d_{\rm NN}(s)=\frac1k\sum_{i\in N_k(s)}\|\tilde s-\tilde s_i\|_2
\]

同时报告 \(d_{\rm all},d_R,d_O\)（\(s^R=q,\dot q\) 28 维；\(s^O\) 其余）。不是三个科学 branch。

## 阈值

\(\tau_{95}=Q_{0.95}(d^{\rm val})\)，val = AC0 held-out **expert validation**（相对该任务 train）。  
G1 在 AC0 **expert test** 上评估 \(r_{\rm OOD}\)（不用 val 自评，避免恒为 ~5%）。

持续离域：连续 3 tick \(d>\tau_{95}\)。

## Seeds / N

诊断 seeds 从 **51101** 起；\(N=16\)/task；\(K_a=1\)。  
不用 P0 38601、不用 AC0 bakeoff 50601。

## Gates（冻结，不看结果改）

- **G1** test expert \(r_{\rm OOD}\le 0.10\)（每任务且平均）。失败：`state_support_metric_invalid`
- **G2** AUROC(\(d_{\rm NN}\)) \(\ge 0.80\) 区分 test expert vs policy rollout。分数 = 距离，policy 为正类。
- **G3** 失败幕中 \(P(T_{\rm exit}<0.5 T_{\max})\ge 0.70\)

## Pattern

- G1–G3 PASS → `state_coverage_failure_supported` → **允许** AC1-R0
- G1 FAIL → `state_support_metric_invalid`
- G1 PASS 且 G2 或 G3 FAIL → `covariate_shift_not_localized` → **禁止** DAgger；回 history / state sufficiency / execution / multimodality；仍锁 diffusion
