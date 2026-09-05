# RTWX-TASK-X1-I1 预注册 — Diffusion Parameterization Breadth Probe

日期：2026-09-01  
状态：**已冻结**  
依赖：TASK-X1 `B1/best.pt` 作 B0 control（**不重训**）；B1/B2 **各训满 100 epoch**  
**禁止**：碰 sampler 合同以外的 DDIM 超参；truncated schedule；Min-SNR；clip；闭环；改 \(K\)/网络/归一化；按 \(L_\epsilon,L_v,L_{x_0}\) 选 winner；覆盖 MR0-P0；进 52601 Stage B

## 假说（可证伪）

当前 Stage A 爆炸是否主要来自 \(\epsilon\)-prediction 在低 SNR 端的 \(1/\alpha_t\) 数值放大？

**不**声称 v-prediction 一定更好。

## 三支

| 支 | `prediction_type` | 训练 |
|----|-------------------|------|
| B0 | `epsilon` | 否（引用 X1） |
| B1 | `v_prediction` | 是；\(\hat x_0=\alpha_t x_t-\sigma_t\hat v\) |
| B2 | `sample`（\(x_0\)） | 是；直接 \(\|\hat x_0-x_0\|_2^2\) |

其余合同继承 X1（AC0 split、\(s_t,z_g\)、\(H_a=8\)、\(d_a=14\)、z-score、256×3 MLP、cosine \(K=100\)、10-step DDIM、\(\eta=0\)、AdamW、100 epoch、无 clip/process/history）。

## 选择指标

同一批 val chunks，\(t\in\{10,50,99\}\)，按各自公式回到 \(x_0\) 空间比 \(L1_{\rm norm}(t)\) 与 \(D_{\rm pred}(t)\)。

- G0：\(L1(10)\le 0.15\)，\(L1(50)\le 0.30\)
- G1：\(L1(99)\le 2.0\)
- G2：\(D_{\rm pred}(99)\le 10\,D_{\rm demo}\)

reconstruction PASS 后才跑完整 10-step DDIM 的 Stage A G0–G4。winner 要求 Stage A 全 PASS。v 与 \(x_0\) 都 PASS 时先比 \(L1(99)\)，再比 \(D_{\rm pred}/D_{\rm demo}\)；差异 \(<10\%\) 则 complexity tie → **v**。

## Pattern

- 两支高噪声仍爆 → `diffusion_high_noise_learning_failure`
- 仅 v PASS rec → `v_parameterization_supported`
- v 与 \(x_0\) 都 PASS rec → `epsilon_parameterization_pathology_supported`
- 仅 \(x_0\) PASS rec → `direct_sample_prediction_supported`

本格 **不** 跑 Stage B。合格 winner 只 **资格** 打开原 X1 Stage B（门不变）。
