# RTWX-O0E0R4 预注册 — Multi-Observation Reference Test

日期：2026-08-31  
状态：**已冻结 / RAN**  
依赖：O0E0R3=`reference_coupled_failure`；B2 / \(S_0\) **冻结**  
**禁止**：训练 memory model；改 B2 search；开 O0E1 / O1；用 R3 seed 37603 作 multiview formal test。

## 科学问题

\[
\boxed{
\text{单帧 partial surface reference 有偏时，
}K\text{ 个已知 }T_{BC}\text{ 观测融合能否稳定 object reference？}
}
\]

对象静止；无 dynamics；无新网络；**仅增加证据**。

## 数据

| 项 | 值 |
|----|-----|
| Segmentation | \(S_0\) natural（O0E0 cache） |
| Multiview test | **新** fresh controlled；seed **37604**；n=**200**；同 P0 generator |
| 每 instance | 杯 pose 冻结；**8** 组 frozen robot configs → 8 dual-view captures |
| R3 cache | seed **37603** — **仅** center-\(\lambda\) diagnostic |

## Center–axis diagnostic（R3 cache，非门）

\[
p_\lambda=(1-\lambda)\hat p_{\mathrm{ref}}+\lambda p^{GT},\quad
\lambda\in\{0,.25,.5,.75,1\}
\]

\(\hat p_{\mathrm{ref}}=\hat p^{\mathrm{surf}}-\delta_y\hat n^{B2}_{\mathrm{formal}}\)。冻结 B2 在 \(p_\lambda\) 上估 axis。报告 \(e_{\mathrm{axis}}=f(e_{\mathrm{center}})\)。

## Multiview fusion（冻结）

对 \(K\in\{1,2,4,8\}\)：取前 \(K\) 个 view（deterministic order）；每 view dual-view fuse 后 **等权** subsample ≤ `N_OBS//K`；拼接后再 cap `N_OBS=256`。然后 **同一冻结 formal B2**（\(\hat p^{\mathrm{surf}}-\delta_y\hat n\)）。

## Gates（formal，per \(K\)）

Axis：median \(\le15^\circ\)，P90 \(\le30^\circ\)。  
Position：\(E_p\le0.20\)，median \(\le5\) cm。

\(K=1\) sanity：应 \(\sim20\text{–}22^\circ\)（R3 级）。

## Patterns

| Pattern | 条件 |
|--------|------|
| `multiview_reference_supported` | 某 \(K\) formal axis+position 均 PASS |
| `multiview_reference_insufficient` | \(K=8\) 仍 ¬G1 |
| `single_observation_reference_coupled` | \(K=1\) FAIL 且 multiview 未 PASS（sanity tag） |

O0E1 / O1 **LOCKED** 除非 `multiview_reference_supported`。
