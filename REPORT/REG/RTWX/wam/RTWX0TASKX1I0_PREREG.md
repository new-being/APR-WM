# RTWX-TASK-X1-I0 预注册 — Diffusion Sampler Closure

日期：2026-09-01  
状态：**已冻结**  
依赖：TASK-X1 `B1/best.pt`（**不重训**）  
**禁止**：clip 动作过 G2；改 \(K\)/DDIM steps；进 Stage B；覆盖 MR0-P0；用 rollout 选 ckpt

## 问题

训练好的 \(\epsilon_\theta\) 与 sampler 是否满足最基本的 forward/reverse consistency？

## 合同

- clean \(A_0\) = AC0 z-score 动作（`n_act`）；采样 \(x_0\) 同空间，**只在最后 denorm 一次**。
- 参数化 = **epsilon**；oracle 反推必须用同一公式。
- 无 clip。

## Tests

- **C0** 真实 \(\epsilon\) 走完整 10-step DDIM：\(\hat A_0\approx A_0\)（norm max-abs \(<10^{-3}\)）。
- **C1** \(t\in\{10,50,99\}\) 单步 oracle \(x_0\)（\(t\neq99\): \(<10^{-5}\)；\(t=99\): \(<10^{-3}\)）。
- **C2** 改用 \(\hat\epsilon_\theta\) 再重建。门槛：norm L1 \(<0.50\)。

## Pattern

- C0/C1 FAIL → `diffusion_sampler_contract_failure` → 修 sampler，**同 ckpt** 重跑 Stage A。
- C0/C1 PASS、C2 FAIL → `diffusion_denoiser_insufficient` → 才允许讨论重训。
- 全 PASS → `diffusion_sampler_closed` → 再进原 Stage A（仍不跳 B）。
