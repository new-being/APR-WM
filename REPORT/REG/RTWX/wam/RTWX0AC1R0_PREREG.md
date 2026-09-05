# RTWX-AC1-R0 预注册 — Corrective State-Coverage Repair

日期：2026-09-01  
状态：**已冻结 / LOCKED until AC1-D0 = `state_coverage_failure_supported`**  
依赖：AC1-D0 PASS；且 `CorrectiveExpert.can_query_arbitrary_snapshot()==True`  
**禁止**：在 D0 未过门时训练；finetune 不同初始化；B1/B2 样本量不等；用 38601 或 50601 或 51101 做正式闭环；改 MLP / process / diffusion / CVAE / history；用 nearest-demo 当 expert label

## 假说

若闭环失败由训练状态覆盖不足导致，则在 policy 实际访问的偏离状态上增加 **expert corrective chunks**，应显著改善闭环，且优于**同等数量**的新干净示范。

## Expert gate

必须能 \(\pi_E(s_t^{\pi})\to A_{t:t+7}^{E}\)（从 snapshot 执行 8 步专家控制）。  
若只有预录 hdf5、不能任意偏离查询：`corrective_expert_unavailable` **STOP**。禁止最近邻 demo 冒充。

## Branches（同架构 AC0-B0，同 init seed，同 \(N_{\rm update}\)）

| Branch | 数据 |
|--------|------|
| B0 | \(\mathcal D_0\) AC0 expert |
| B1 | \(\mathcal D_0+\mathcal D_{\rm clean}\)，\(\lvert\mathcal D_{\rm clean}\rvert=\lvert\mathcal D_{\rm corr}\rvert\) **每任务匹配**，fresh normal demos |
| B2 | \(\mathcal D_0+\mathcal D_{\rm corr}\)，仅 \(d_t>\tau_{95}\) 后：\(t_{\rm exit},+8,+16,\ldots\) 每幕最多 8 点 |

Winner：pooled success → min-task → safety → \(L_{\rm corr}\)（offline 不选 winner）。

支持 `corrective_coverage_supported` 当且仅当：

\[
S_{B2}-S_{B0}\ge 0.10,\quad S_{B2}-S_{B1}\ge 0.05,
\]

且 min-task 不低于 B1。

否则：B1≈B2 提升 → `data_volume_effect`；三者都差 → `coverage_repair_insufficient`（仍不自动 diffusion）。

## 闭环评测

seeds 从 **51601**；\(N=32\)/task；3 branch → 288 幕。  
R0 后再拷 winner → 原样 MR0-P0（G3 不放宽）。
