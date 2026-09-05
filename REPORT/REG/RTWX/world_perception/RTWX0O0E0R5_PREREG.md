# RTWX-O0E0R5 预注册 — Visibility-Aware Object Reference

日期：2026-08-31  
状态：**已冻结 / RAN**  
依赖：O0E0R4=`multiview_reference_insufficient`；B2 / \(S_0\) / CAD / camera contract **冻结**  
**唯一机制替换**：\(\hat p^{\mathrm{surf}}-\delta_y\hat n \rightarrow c_{\mathrm{obs}}-\delta_{\mathrm{vis}}(n,T_{BC})\)

**禁止**：训练 memory / learned center；改 B2 \((h,r)\) objective；cloud union multiview；用 seed 37604 作 formal test。

## 科学假设

\[
\boxed{
H_{\mathrm{vis}}:
\text{center bias 主要来自 partial visibility；
CAD+viewpoint 可预测 }\delta_{\mathrm{vis}}\text{ 并得到足够准确的 object center。}
}
\]

## 冻结不变

\(S_0\) natural UNet；021\_cup CAD；B2 Fibonacci 162 → Top-8 → \(5°/3°/1°\)；128²；P0 controlled generator。

## Visibility model

\[
\delta_v(n)=a(\alpha_v,d_v)\,n+b(\alpha_v,d_v)\,r_v(n),\quad
\hat p_h(n)=c_h-\delta_h(n),\quad
\hat p_o(n)=c_o-\delta_o(n),\quad
\hat p_{\mathrm{vis}}(n)=\tfrac12(\hat p_h+\hat p_o).
\]

LUT：CAD-only；\(\alpha=0°\ldots180°\) step 2°；distance grid from workspace；8-fold yaw marginalization。

## 数据

| 项 | 值 |
|----|-----|
| Formal test | **新** seed **37605**；n=200 |
| Segmentation | \(S_0\) natural |

## 三层门

**L1 Data**：\(P_{\mathrm{any}}\ge0.98\)；\(r_{\mathrm{excite}}\ge0.30\)；8/8 yaw octants。

**L2 G\_I0**：\(S_0\) cloud + GT \(p\) axis \(\le15°\)/P90 \(\le30°\)；GT \(n\) position \(\le5\) cm。

**L2 G\_I1**：oracle \(n^{GT}\) + \(\hat p_{\mathrm{vis}}\)；position gate 不变；axis at \(\hat p_{\mathrm{vis}}(n^{GT})\) \(\le15°\)/P90 \(\le30°\)。  
Diagnostic：\(P(e_p\le1.75\text{ cm})\)（R4 机制尺度，**非 hard gate**）。

**L3 Formal**：\(\hat n=\arg\min_n S(n;\hat p_{\mathrm{vis}}(n))\)；paired baseline \(B_{\mathrm{old}}\) 同 seed。

## Patterns

| Pattern | 条件 |
|--------|------|
| `effective_pose_supported` | L3 vis formal PASS |
| `visibility_reference_instrument_failure` | G\_I1 FAIL |
| `visibility_reference_joint_failure` | G\_I1 PASS，L3 FAIL |
| `segmentation_cloud_failure` | G\_I0 FAIL |
| `observation_support_failure` | L1 FAIL |

O0E1 **LOCKED** 除非 `effective_pose_supported`。

## Diagnostics（D1–D5）

center error；by tilt；by camera；\(D_{ho}=\|\hat p_h-\hat p_o\|\)；legacy vs visibility。
