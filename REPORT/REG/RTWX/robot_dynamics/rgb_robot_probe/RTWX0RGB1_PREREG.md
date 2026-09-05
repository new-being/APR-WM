# RTWX-X0RGB1 预注册 — Spatial-Temporal State Recovery

日期：2026-08-29  
状态：**已冻结（采集/训练/评分前）**  
依赖：X0RGB = `perception_failure`（否掉的是 64×64 flatten MLP，不是视觉路线）。  
**禁止**：改 M2 basis / \(W\)；改 \(H_{\mathrm{plan}}/K_{\mathrm{execute}}\)；改 \(E_q\le0.35\)、\(E_{\dot q}\le0.50\)；改 X0E–X0RGB ledger；TASK-XL；R10；visual latent world model 作为本格主模型；真实相机声称。

## 科学问题

\[
\boxed{
\text{加入空间与时间归纳偏置后，第三人称 RGB 能否可靠恢复 }(q,\dot q)\text{？}
}
\]

本格 **不做容量结论**。唯一允许变化的是 \(E_{\mathrm{vis}}\)。

## 冻结合同

| 项 | 冻结值 |
|----|--------|
| 任务 | `put_object_cabinet` |
| seed | **15601**（fully fresh；48/24/24 × 120） |
| 相机 | `front_camera`；采集 **128×128** |
| 下游 | 冻结 X0E-M2，\(P=1752\)；oracle train LS \(W\) |
| 部署 | \(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\) |
| \(q^{\mathrm{tar}}\) | oracle |
| G_vis 阈值 | \(E_q\le0.35\)，\(E_{\dot q}\le0.50\)（**不放宽**） |
| 评价 NRMSE | 同 `rtwx_x0._nrmse`（不换 metric） |

## Perception instruments

### P0 — frozen negative control（X0RGB 仪器）

\[
64\times64,\ L=2,\ \mathrm{flatten}\to\mathrm{MLP}\to(\hat q,\hat{\dot q})
\]

仅对照，不决定 primary pattern。

### P1 — Spatial only

\[
RGB_t^{128}\xrightarrow{\mathrm{CNN}} f_t \rightarrow (\hat q_t,\hat{\dot q}_t)
\]

小型 4-stage stride-2 ConvNet + global pool。检验空间偏置本身。

### P2 — Spatial + temporal（主 candidate）

\[
\hat q_t = E_q(RGB_t),\qquad
\hat{\dot q}_t = E_v(f_{t-3:t}),\quad L=4.
\]

\(q\) 单帧；\(\dot q\) 用 \(L=4\) feature 的小 temporal MLP。\(\lambda_v=1\)。

训练在逐维标准化坐标上：

\[
\mathcal L=\mathcal L_q+\lambda_v\mathcal L_{\dot q},\quad\lambda_v=1.
\]

## Per-joint audit（必须报告）

\(E_{q,j}\)，\(E_{\dot q,j}\)，以及 \(\#\{j:E_{q,j}<0.35\}\)。

## 部署（仅 G_vis 通过后作科学解释）

B0：oracle reset + 冻结 M2。  
B1：视觉 reset + 冻结 M2；**identity 用同一视觉 reset**。

\[
\eta_{\mathrm{retain}}=
\frac{E_{\mathrm{id}}^{\mathrm{vis}}-E_{\mathrm{exec}}^{\mathrm{vis}}}
{E_{\mathrm{id}}^{\mathrm{oracle}}-E_{\mathrm{exec}}^{\mathrm{oracle}}}
\]

仅 G_vis PASS 后解释。

合理部署：\(r_{\mathrm{cat}}=0\) 且 \(E_{\mathrm{exec}}^{\mathrm{vis}}\le0.75\,E_{\mathrm{id}}^{\mathrm{vis}}\)。

## Patterns（primary = P2）

| Pattern | 条件 |
|---------|------|
| `sim_rgb_state_interface_supported` | P2：\(E_q\le0.35\)，\(E_{\dot q}\le0.50\)，\(r_{\mathrm{cat}}=0\)，且 vis-contract competent |
| `velocity_perception_failure` | \(E_q\le0.35\) 且 \(E_{\dot q}>0.50\) |
| `spatial_perception_failure` | \(E_q>0.35\) |
| `visual_dynamics_interface_failure` | G_vis PASS 但 \(r_{\mathrm{cat}}>0\) 或 vis-contract 不 competent |

若 P2 spatial 失败，P1 的 \(E_q\) 仅诊断披露（不改 pattern 规则）。

## 明确不做

- 不改 M2；不上 ViT / 大 visual latent WM  
- 不预测 RGB\(_{t+1}\)；不开多视角 / keypoint / R10  
- 不因失败放宽 G_vis
