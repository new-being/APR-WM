# RTWX-O0E0R5 报告 — Visibility-Aware Object Reference

日期：2026-08-31  
状态：**正式冻结** `visibility_reference_joint_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0R5_PREREG.md`  
产物：`runs/rtwx_o0e0r5/`；fresh test seed **37605**；唯一机制替换：fixed \(\delta_y\) → \(\delta_{\mathrm{vis}}(n,T_{BC})\)；B2 **冻结**

## 一句话

\[
\boxed{\texttt{visibility\_reference\_joint\_failure}}
\]

**Oracle-axis visibility center 已闭合**（G\_I1 axis **14.6°** PASS；\(e_p\) **1.83 cm**），证明 partial-view bias 可被 CAD visibility 显式修正。但 **formal joint** \( \hat n=\arg\min S(n;\hat p_{\mathrm{vis}}(n)) \) 仍 FAIL（median **16.6°**，P90 **121.6°**）。相对同 seed baseline（**21.1°**）改善 **4.5°**，未过门。

## 三层门

| 层 | 结果 |
|----|------|
| **L1 Data** | PASS — \(P_{\mathrm{any}}=1.0\)；\(r_{\mathrm{excite}}=0.6\)；8/8 octants |
| **L2 G\_I0** | PASS — S₀+GT \(p\) axis **0.16°**；GT \(n\) position **1.78 cm** |
| **L2 G\_I1** | **PASS** — \(\hat p_{\mathrm{vis}}(n^{GT})\) ep **1.83 cm**；B2@vis center axis **14.62°** / P90 **19.8°** |
| **L3 Formal vis** | **FAIL axis** — median **16.64°** P90 **121.6°**；position **PASS** **2.12 cm** |
| **L3 Formal old** (paired) | median **21.13°** P90 **85.5°**；position **3.36 cm** |

## 机制解读（情景 B）

\[
\boxed{
\hat p_{\mathrm{vis}}(n^{GT})\text{ good}
\;\Rightarrow\;
\text{visibility model 可行}
}
\]

\[
\boxed{
p_{\mathrm{vis}}(n)\leftrightarrow n \text{ joint search FAIL}
\;\Rightarrow\;
\text{center–axis coupling / local ambiguity}
}
\]

R4 预言的 **~1.75 cm** 机制尺度在 oracle instrument 上成立：median ep **1.83 cm**，axis 跨 **15°** 门。Formal 仅差 **1.6°** median，但 P90 尾部仍灾难性——与 R3/R4 reference coupling 同型，现为 **联合** 而非 **单帧 fixed offset** 问题。

## Diagnostics

| 项 | 值 |
|----|-----|
| D1 median \(e_p^{\mathrm{vis}}\) | **1.83 cm**；\(P(e_p\le1.75\text{ cm})=43.5\%\) |
| D4 \(D_{ho}=\|\hat p_h-\hat p_o\|\) median | **4.6 cm** |
| D5 \(\Delta e_{\mathrm{axis}}^{\mathrm{med}}\) (old→vis) | **+4.49°** 改善 |
| D3 per-camera ep (oracle \(n\)) | head **3.52 cm**；obs **3.93 cm**（融合后 **1.83 cm**） |

## Pattern

```text
pattern = visibility_reference_joint_failure
unlocks_o0e1_prereg = false
unlocks_o1 = false
```

## 措辞（硬）

| 能说 | 不能说 |
|------|--------|
| visibility-aware center 在 oracle axis 下闭合 axis 门 | effective_pose_supported |
| joint center–axis 仍是 formal blocker | visibility model 不足（G\_I1 已否定） |
| 下一步：alternating/joint center–axis inference | O0E1 已解锁 |
| formal vis 优于同 seed old baseline | visibility 已完全救 O0E0 |

## 下一步（机制导向，未 prereg）

\[
\boxed{
\text{joint / alternating center–axis registration}
}
\]

若仍 FAIL → R6 persistent surface。B2 / \(S_0\) 冻结。
