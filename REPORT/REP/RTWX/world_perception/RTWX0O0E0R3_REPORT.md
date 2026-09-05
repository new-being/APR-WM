# RTWX-O0E0R3 报告 — S0 Formal Effective-Pose Confirmation

日期：2026-08-31  
状态：**正式冻结** `reference_coupled_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0R3_PREREG.md`  
产物：`runs/rtwx_o0e0r3/`；fresh test seed **37603**（≠ R2 的 37602）；\(S_0\) natural UNet；B2 **冻结**

## 一句话

\[
\boxed{\texttt{reference\_coupled\_failure}}
\]

在全新 fresh controlled test 上，**\(S_0\) cloud instrument 闭合**（G_I1 **0.15°**，G_I2 **1.85 cm**），但 **formal non-oracle B2 未过 axis 门**（median **22.4°**，P90 **30.9°**）。position 已过（median **3.55 cm**）。机制链被钉死：

\[
\text{segmentation}(S_0)\Rightarrow\text{cloud OK}
\Rightarrow
\boxed{\text{reference/centering blocks formal axis}}
\Rightarrow
\text{quotient B2}
\]

与 R1 的 GT-mask+reference **20.3°** 一致量级。

## 三层门

| 层 | 结果 |
|----|------|
| **L1 Data** | PASS — \(P_{\mathrm{any}}=1.0\)；\(r_{\mathrm{excite}}=0.6\)；G0c 8/8 per \(\beta\) |
| **L2 Instrument** | **PASS** — G_I1 median **0.15°** P90 **0.25°**；G_I2 median **1.85 cm** |
| **L3 Formal** | **FAIL axis** — median **22.39°** P90 **30.89°**；position **PASS** median **3.55 cm** |

## 与 R1/R2 对照

| Branch | median \(e_{\mathrm{axis}}\) | 本格语境 |
|--------|------------------------------|----------|
| R1 GT mask + reference | 20.3° | 同机制 |
| R1 GT mask + GT \(p\) | 0.17° | oracle center |
| R2 \(S_0\) instrument (seed 37602) | 0.17° | 复现 |
| **R3 \(S_0\) instrument (seed 37603)** | **0.15°** | **稳定** |
| **R3 formal \(S_0\)+reference** | **22.4°** | **本格 claim** |

\[
\boxed{\text{S0 cloud 闭合} \neq \text{formal effective pose 闭合}}
\]

正式验证。

## Pattern

```text
pattern = reference_coupled_failure
science_ran = true
unlocks_o0e1 = false
claim_scope = controlled_021_cup_rgbd_S0_only
```

## 措辞（硬）

| 能说 | 不能说 |
|------|--------|
| reference/centering 是 formal axis 主 blocker | B2 objective 应 retune |
| \(S_0\) segmentation 在 fresh controlled 上稳定 instrument PASS | effective_pose_supported |
| 下一步：pose-equivariant object center / multi-frame reference | O0E1 已解锁 |
| position formal 已过（3.55 cm） | segmentation 仍需修 \(S_1\) |

## 下一步（机制导向，未 prereg）

\[
\boxed{
\textbf{如何从 partial RGB-D cloud 建立 view-robust object center/reference？}
}
\]

单帧 \(\hat p^{\mathrm{surf}}-\delta_y\hat n\) 在 cloud 已好时仍留下 ~22° axis gap；与 persistent object memory / multi-frame registration 方向一致。**不在本格内改 B2 或重开 \(S_1\)。**
