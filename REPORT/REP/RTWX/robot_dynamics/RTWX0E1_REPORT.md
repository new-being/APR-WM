# RTWX-X0E1 报告 — Structured vs latent capacity

日期：2026-08-28  
状态：**completed / `structure_not_in_robust_set`**  
**不是** capacity PASS。`run.json` 里的 `capacity_claim=true` 是实现过宽字段，**科学记录以本报告为准：无 \(R_P\) 声称。**  
预注册：`REPORT/REG/RTWX/RTWX0E1_PREREG.md`  
产物：`runs/rtwx_x0e1/`  
**不改** X0E PASS / M2 basis / X0E1 门。无 R10。

## 冻结结论

\[
\boxed{\texttt{RTWX-X0E1 = completed / structure\_not\_in\_robust\_set}}
\]

允许的最强表述：

\[
\boxed{
\text{冻结非线性 increment 结构在 held-out prediction 上极省参数，}
\text{但其跨 split rollout 稳健性不足，不能计入稳定容量替代。}
}
\]

失败的是 **recursive rollout robustness**，不是 one-step representation。这 **不等于** “结构没有价值”。

## 三层读数

**1. Held-out 上结构极有竞争力。** 冻结 M2：\(P=1752\)，test \(E_1=0.740\)，\(E_{\mathrm{roll}10}=0.647\)，\(E_{\mathrm{roll}50}=0.723\)，优于 \(H=256\) reference 的 \(0.763/0.694/0.759\)。只看 test 会得到过强的“结构替代容量”印象。

**2. \(G_{\mathrm{robust}}\) 挡住了该结论。** train \(E_{\mathrm{roll}10}\sim 10^{37}\)。B1/B2 均不在稳健容量集。\(R_P^{\mathrm{struct}},R_P^{\mathrm{res}}\) **必须保持 undefined**。

**3. 纯网络有干净下限。** \(P_{\mathrm{NN}}^{\min}=H=128\approx 4.09\times 10^4\) 参数。\(H=64\) 一步接近，rollout 未过 frozen match。

## 账本

```text
RTWX-X0C  = FAIL / servo_signal_insufficient
RTWX-X0D  = FAIL / low_order_structure_insufficient
RTWX-X0E  = PASS / nonlinear_increment_required   不改
RTWX-X0E1 = completed / structure_not_in_robust_set
            NOT capacity PASS; R_P undefined
            P_NN^min = 128 (neural width floor only)
X0E-S     = 稳定性诊断（新格，不改 X0E/X0E1）
TASK-XL / R10 = LOCKED
```
