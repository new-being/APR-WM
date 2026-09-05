# RTWX-AC1-D1 报告 — State-Support Metric Breadth Audit

日期：2026-09-01  
状态：**RAN**  
pattern = `support_geometry_unqualified`  
winner = **none**  
预注册：`REPORT/REG/RTWX/wam/RTWX0AC1D1_PREREG.md`  
产物：`runs/rtwx_ac1_d1/`  
标准化：AC0-B0 train \(\mu_s,\sigma_s\)（与 D0 相同）  
Winner **未**使用 policy AUROC。未跑 policy rollout。未开 R0。

## 一句话

\[
\boxed{
\text{四种简单支持域几何都无法把三任务 expert false-OOD 压到 }0.10\text{ 以下。}
}
\]

柜子仍是瓶颈。covariate shift **仍 plausible、仍未定位**。缺的是合格仪器，不是 DAgger。

## Selection（AC0 split 50600；\(\tau_{95}\)←val；\(r_{\rm OOD}\)←test）

| 支 | cup | cabinet | stamp | 三任务 \(\le 0.10\) |
|----|----:|--------:|------:|:------------------:|
| B0 Euclidean kNN \(k=5\) | 0.0047 | **0.2063** | 0.0000 | no |
| B1 全局 Mahalanobis | 0.0806 | **0.2070** | 0.0306 | no |
| B2 阶段条件 kNN | 0.0047 | **0.2063** | 0.0000 | no |
| B3 局部对角协方差 \(k_{\rm loc}=30\) | 0.0403 | **0.1910** | 0.0778 | no |

B0 柜子 0.2063 **复现 D0 G1**（同一 whitening / 同一 split）。  
B2 与 B0 在 cup/cabinet 上数值相同：当前因果四相位 **没有**改写柜子假阳性。  
B3 柜子略降到 0.191，仍高于 0.10。  
B1 未改善柜子，还抬高了杯子假阳性。

无一合格 → 不做 seed 52100 confirmation。

```text
pattern = support_geometry_unqualified
winner = none
unlocks_d0_rescore = false
unlocks_ac1_r0 = false
```

## 解释边界

- **不是** “柜子 expert test 坏了所以放弃 covariate shift”。
- **不是** 阶段假说已被否定到可以忽略——只是 **这一版** \(\phi\)（approach/engage/transport/terminate，无未来信息）不足以当 support 条件。
- 全局/局部协方差也没把合法专家流形校准到合同门槛。

## 下一步（仍修仪器，不修 policy）

若继续：需要 **新的、预注册的** 支持域表示（例如更细的交互相、逐对象切片、或非欧结构），而不是放宽 0.10，也不是用 rollout AUROC 回选度量。  
AC1-R0 / diffusion / MR0-A 仍 LOCKED。
