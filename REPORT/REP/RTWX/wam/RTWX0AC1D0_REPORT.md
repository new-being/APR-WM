# RTWX-AC1-D0 报告 — Closed-Loop State-Coverage Diagnosis

日期：2026-09-01  
状态：**RAN**  
pattern = `state_support_metric_invalid`  
预注册：`REPORT/REG/RTWX/wam/RTWX0AC1D0_PREREG.md`  
产物：`runs/rtwx_ac1_d0/`  
policy = AC0-B0；seeds **51101**；\(N=16\)/task；\(K_a=1\)；\(k=5\)

## 一句话

\[
\boxed{
\text{协变量偏移看起来很像，但现有 }d_{\rm NN}\text{ 支持域度量不够可靠，机制尚未闭合。}
}
\]

G1 **FAIL**。不得因 G2/G3 漂亮而绕过 G1。  
**不**写 `state_coverage_failure_supported`。**不**开 AC1-R0。  
**不是**对 covariate shift 的否定，也 **不是** coverage failure 已被证明。

## Gates

| 门 | 合同 | 结果 |
|----|------|------|
| G1 | test expert \(r_{\rm OOD}\le 0.10\)（每任务） | **FAIL** |
| G2 | AUROC(\(d_{\rm NN}\))\(\ge 0.80\) policy vs test expert | PASS 0.994 |
| G3 | 失败幕 \(P(T_{\rm exit}<0.5T_{\max})\ge 0.70\) | PASS 1.00（48/48 fail） |

G1 分任务（\(\tau_{95}\) 由 **val** 标定，\(r_{\rm OOD}\) 在 **test**）：

| 任务 | \(r_{\rm OOD}^{expert,test}\) |
|------|------------------------------:|
| place_empty_cup | 0.005 |
| stamp_seal | 0.000 |
| put_object_cabinet | **0.206** |
| 三任务平均 | 0.070 |

柜子专家 test 有 20.6% 被判 OOD：仪器对**合法专家状态**假阳性过高。  
因此 AUROC 只说明 rollout 与 expert 易分，**不能**证明那是“坏的覆盖偏移”。G3 只说明按这个有缺陷的 \(\tau\)，policy 很早被判离域。

闭环 pooled success 仍为 0（诊断幕，非 P0）。

```text
pattern = state_support_metric_invalid
unlocks_ac1_r0 = false
covariate_shift = plausible_unlocalized
```

## 下一步

修 **诊断仪器**（AC1-D1），不修 policy，不开 DAgger。
