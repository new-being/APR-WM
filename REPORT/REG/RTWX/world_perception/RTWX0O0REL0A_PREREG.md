# RTWX-O0REL0A 预注册 — Ego Transform vs Visibility-Overlap Audit

日期：2026-08-31  
状态：**已冻结 / RAN**  
依赖：O0REL0=`ego_motion_compensation_failure`（**formal pattern 不变**）  
**零训练**；复用 `runs/rtwx_o0rel0/cache/{c0,c1,c2}`；**不改** ICP / \(S_0\) / B2

## 唯一问题

\[
\boxed{
H_{\mathrm{ov}}:
\text{C0 FAIL 来自错误 }T_{BC}\text{ 补偿，
还是正确坐标系下 partial-surface overlap collapse？}
}
\]

本格 **不 rescue** REL0 formal 结果；只做 evaluator-only oracle diagnostics。

## 冻结输入

- REL0 cache seed **37606**；600 pairs
- Formal S0-path 指标：读 `runs/rtwx_o0rel0/per_pair.json`（不重训）
- GT mask / GT pose：仅 evaluator
- ICP：`RelativeICPConfig` 与 REL0 相同（hash 冻结）

## Diagnostic A — GT-mask ICP（C0 primary）

C0 上：

\[
M^{GT}_{0,1}\rightarrow\mathcal P^{GT}_{0,1}\rightarrow\text{same frozen ICP}
\]

报告 vs S0-path 的 \(e_{\Delta n}\)（median / P90）。

**解读（冻结）**：
- GT-mask 仍 \(\sim20°\) → segmentation **排除**为主因
- GT-mask \(<5°\) → view-specific / seg-coupled cloud

## Diagnostic B — Per-frame CAD consistency（C0 primary）

静止物体，evaluator-only：

\[
d_t = D(\mathcal P_t^B,\, T_{BO,t}^{GT}\mathcal G_{CAD})
\]

\(D\) = cloud→CAD median NN distance（object frame；CAD HR profile）。

报告 \(d_0, d_1\)（median / P90，normalize by \(D_O\)）。

**解读（冻结）**：
- \(d_0,d_1 \ll D_O\) 且 \(\rho_{\mathrm{overlap}}\approx0\) → 两帧各自几何正确，但看到不同 surface

## Diagnostic C — \(e_{\Delta n}\) vs \(\rho_{\mathrm{overlap}}\)（pooled）

S0-path；C0+C1+C2 pooled；固定 bins：

\[
[0,0.1),\ [0.1,0.25),\ [0.25,0.5),\ [0.5,1]
\]

每 bin：median / P90 \(e_{\Delta n}\)；附 regime 分层（descriptive）。

**解读（冻结）**：
- \(\rho<0.1 \Rightarrow e_{\Delta n}\sim20°+\)；\(\rho>0.25\Rightarrow e_{\Delta n}\sim5°\) 且跨 regime → **shared-surface observability** 控制 validity

## Patterns（descriptive only）

```text
overlap_support_hypothesis_supported
overlap_support_hypothesis_weak
segmentation_coupled_c0_failure
```

**不**改 `ego_motion_compensation_failure`；**不**解锁 O0E1/O1。

Unlock（descriptive）：`supports_overlap_gated_relative_tracking`（若 C 成立）

## 实现

```text
aprwm_v0/rtwx_o0rel0a.py
runs/rtwx_o0rel0a/{summary,header}.json
```

CLI：`rtwx-o0rel0a --rel0-run runs/rtwx_o0rel0`

## 一句话

\[
\boxed{
\text{用 oracle 把 C0 的 20° 拆开：
是坐标补偿错，还是 overlap 不够？}
}
\]
