# RTWX-TASK-X1-I0 报告 — Diffusion Sampler Closure

日期：2026-09-01  
状态：**RAN**  
pattern = `diffusion_denoiser_insufficient`  
预注册：`REPORT/REG/RTWX/wam/RTWX0TASKX1I0_PREREG.md`  
产物：`runs/rtwx_task_x1_i0/`  
checkpoint = TASK-X1 `B1/best.pt`（**未重训、未 clip、未进 Stage B**）

## 一句话

\[
\boxed{
\text{C0/C1 PASS：sampler + }\epsilon\text{-公式 + z-score 合同成立。}
}
\]

\[
\boxed{
\text{C2 FAIL 仅在 }t=99\text{：模型噪声在终端噪声步被 }1/\sqrt{\bar\alpha_{99}}\text{ 放大。}
}
\]

不要宣布 diffusion 失败，也不要修采样器去过 G2。

## 合同核查

| 项 | 结果 |
|----|------|
| clean \(A_0\) 空间 | AC0 train **z-score**（`n_act`） |
| 与 AC0-B0 normalizer | **一致** |
| 采样后 denorm | **一次** |
| 参数化 | **epsilon**（与训练一致） |
| \(\bar\alpha_0\) | 0.99937 |
| \(\bar\alpha_{99}\) | \(2.43\times 10^{-7}\) |
| clip | **无** |

## Closure

| 测 | 结果 | 细节 |
|----|------|------|
| C0 真 \(\epsilon\) 10-step DDIM | **PASS** | max-abs \(4.0\times 10^{-4}\) |
| C1 \(t=10\) | **PASS** | \(1.2\times 10^{-7}\) |
| C1 \(t=50\) | **PASS** | \(2.4\times 10^{-7}\) |
| C1 \(t=99\) | **PASS** | \(4.0\times 10^{-4}\)（tiny \(\bar\alpha\) 浮点） |
| C2 \(t=10\) model-\(\epsilon\) | PASS | L1_norm 0.091 |
| C2 \(t=50\) | PASS | L1_norm 0.193 |
| C2 \(t=99\) | **FAIL** | L1_norm **558**；raw L1 352；\(D=1966\) vs demo 0.047；\(\epsilon\)-MSE 仅 0.12 |

C0/C1 说明：把真实 \(\epsilon\) 塞进同一套 reverse，能回到真实 \(A_0\)。G2 爆炸 **不是** 重复 denorm，也 **不是** 把 \(\epsilon\) 当成 \(x_0\) 预测。

C2 在 \(t=99\)：\(\epsilon\) 误差看起来不大（MSE 0.12），但

\[
\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t},\quad \sqrt{\bar\alpha_{99}}\approx 4.9\times 10^{-4}
\]

把 \(O(10^{-1})\) 的 \(\epsilon\) 误差放大成 \(O(10^{2})\) 的动作。这与 Stage A 的 \(L_1\approx 160\)、\(D_{\rm pred}\approx 787\) **同量级**。

```text
pattern = diffusion_denoiser_insufficient
fix_sampler_then_restage_a = false
unlocks_retrain = true
unlocks_stage_b = false
```

## 下一步

有资格讨论：重训 / 改噪声末端 / 改参数化——那是**新格**，本格不扫 \(K\)、不加 clip、不进 Stage B。
