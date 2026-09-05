# RTWX-X0RGB 预注册 — Sim RGB → State → Frozen M2

日期：2026-08-28  
状态：**已冻结（采集/训练/评分前）**  
依赖：X0EH1 = `receding_structure_capacity_shift`；X0EH2 = `cross_task_structure_supported`。  
**禁止**：改 M2 basis / \(W\) 拟合协议；改 \(H_{\mathrm{plan}}/K_{\mathrm{execute}}\)；改 X0E–X0EH2 ledger；TASK-XL；R10；`RGB→RGB` 端到端作为主 claim；真实相机声称。

## 科学问题

\[
\boxed{
\text{当 state 不再 oracle 给出，而必须从仿真 RGB 恢复时，
冻结 M2 的 }K=4\text{ 部署效用保留多少？}
}
\]

RoboTwin 渲染 RGB **仍是仿真视觉**，不是真实相机。本格只验证 **oracle dynamics → sim RGB observation** 这一跳。

## 冻结合同（继承 X0EH1）

| 项 | 冻结值 |
|----|--------|
| 任务 | `put_object_cabinet`（与 X0EH1 source 同任务，便于 \(\Delta_{\mathrm{vision}}\) 对照） |
| seed | **14601**（fresh；48/24/24 × 120） |
| 相机 | `front_camera`（固定第三人称）；resize **64×64** RGB uint8 |
| 历史 | \(L=2\) 帧（\(RGB_{t-1}, RGB_t\)）用于 \(\hat{\dot q}\) |
| 动力学 | 冻结 X0E-M2 basis \(\phi\)，\(P_{\mathrm{struct}}=1752\)；train LS \(W\)（oracle train） |
| 部署 | \(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\) |
| \(q^{\mathrm{tar}}\) | **oracle**（与 X0E 采集一致；本格不引入视觉目标估计） |

## 模型（primary B0 vs B1）

### B0 — Oracle upper bound

\[
(q_t,\dot q_t,q_t^{\mathrm{tar}})\rightarrow \Delta x_t \rightarrow \hat s_{t+1}
\]

同 X0EH1 B1 M2；segment 起止均用 **oracle** \(s_t\)。

### B1 — RGB → state → frozen M2（主模型）

\[
RGB_{t-L:t}\xrightarrow{E_{\mathrm{vis}}}(\hat q_t,\hat{\dot q}_t),\quad
(\hat q_t,\hat{\dot q}_t,q_t^{\mathrm{tar}})\xrightarrow{M2}\hat s_{t+1}
\]

- \(E_{\mathrm{vis}}\)：小型 CNN/MLP（架构在代码中冻结，正式前不得改）
- M2 的 \(W\)：**仅用 oracle train 拟合**，不 joint fine-tune
- **Segment reset**：每 \(K\) 步边界用 \(E_{\mathrm{vis}}(RGB_{t-L:t})\) 估计 \((\hat q,\hat{\dot q})\) 作 reset（非 oracle）
- Segment 内：open-loop rollout 从估计 state 出发

### B2 — RGB visual-latent baseline（exploratory，不进 primary pattern）

\[
RGB_{t-L:t},q_t^{\mathrm{tar}}\rightarrow \Delta x_t
\]

同参数量级小 MLP；用于 **显式 state bottleneck vs visual-latent** 探索性对照。

## 指标

### Perception（test）

\[
E_q=\mathrm{NRMSE}(\hat q,q),\qquad
E_{\dot q}=\mathrm{NRMSE}(\hat{\dot q},\dot q)
\]

NRMSE 定义同 `aprwm_v0/rtwx_x0._nrmse`（RMSE(pred,ref) / RMS(ref)）。

### Deployment（test，\(K=4\)，主指标）

沿用 X0EH1：\(E_{\mathrm{exec}}^{K=4}\)，\(E_{\mathrm{end}}^{K=4}\)，\(r_{\mathrm{cat}}^{K=4}\)，\(E_{\mathrm{identity}}^{K=4}\)。

### Interface loss（exploratory）

\[
\Delta_{\mathrm{vision}}=E_{\mathrm{exec}}^{K=4}(\mathrm{B1})-E_{\mathrm{exec}}^{K=4}(\mathrm{B0})
\]

\[
\eta_{\mathrm{retain}}=
\frac{E_{\mathrm{identity}}-E_{\mathrm{rgb}}}
{E_{\mathrm{identity}}-E_{\mathrm{oracle}}}
\]

（仅当分母 \(>0\) 时定义。）

## Primary gates

- **G_oracle**：B0 competent（\(r_{\mathrm{cat}}=0\)，\(E_{\mathrm{exec}}\le0.75\,E_{\mathrm{identity}}\)）
- **G_vis**：\(E_q\le0.35\) 且 \(E_{\dot q}\le0.50\)（test）
- **G_deploy**：B1 \(r_{\mathrm{cat}}=0\)
- **G_retain**：B1 \(E_{\mathrm{exec}}\le1.30\,E_{\mathrm{exec}}^{\mathrm{B0}}\)

## Patterns

| Pattern | 条件 |
|---------|------|
| `structure_survives_sim_rgb` | G_oracle ∧ G_vis ∧ G_deploy ∧ G_retain |
| `vision_limited_structure` | G_oracle ∧ G_vis ∧ G_deploy，但 ¬G_retain |
| `perception_failure` | ¬G_vis |
| `oracle_regression` | ¬G_oracle |
| `catastrophe_returns` | G_oracle ∧ G_vis，但 B1 \(r_{\mathrm{cat}}>0\) |

B2 结果仅 exploratory 披露，不参与 primary pattern。

## 明确不做

- 不预测 RGB\(_{t+1}\)
- 不把 object pose / 任务进度 / diffusion 纳入 \(s\)
- 不声称真实世界视觉验证
- 不 joint fine-tune M2 \(W\) with RGB
- 不开 TASK-XL / R10

## 成功读数

`structure_survives_sim_rgb` ⇒ 支持「显式 state bottleneck + 冻结结构动力学」在仿真视觉接口下仍 deployment-relevant；可规划 **real RGB shadow** 格，但仍非 R10。
