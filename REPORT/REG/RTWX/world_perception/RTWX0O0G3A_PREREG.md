# RTWX-O0G3A 预注册 — Correspondence Availability Audit

日期：2026-08-30  
状态：**已冻结**  
依赖：O0G3R=`coverage_failure`  
**禁止**：改 \(N_{\min}=50\) 救原格；训练 correspondence；科学结论 claim。

## 问题（描述性）

> 为何 \(P(V^{any})=1.0\) 但 \(P_{\mathrm{enough}}=0.875\)？看得见 ≠ 有足够几何点做 orientation。

## 数据

复用 `runs/rtwx_o0g3r/cache_o0g3r_s*_test.npz`（seeds 29601/02/03）。

## 报告量

每 visible 帧：\(N_{\mathrm{corr}}^{head}\)、\(N_{\mathrm{corr}}^{obs}\)、\(N_{\mathrm{corr}}^{union}\)；mask area；depth-valid fraction；dual-view valid。

曲线（分箱）：

- \(P(N_{\mathrm{union}}\ge N_{\min}\mid \text{mask area})\)
- \(P(N_{\mathrm{union}}\ge N_{\min}\mid \text{depth-valid fraction})\)

## Pattern

固定：`correspondence_availability_characterized`（无 PASS/FAIL 科学门；不事后改阈值）。
