# RTWX-O0REL0 报告 — Relative Effective-Pose Tracking Probe

日期：2026-08-31  
状态：**正式冻结** `ego_motion_compensation_failure`（C1/C2 science **PASS**，C0 instrument **FAIL**）  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0REL0_PREREG.md`  
产物：`runs/rtwx_o0rel0/`；pair seed **37606**；600 pairs（C0/C1/C2 × 200）

## 一句话

\[
\boxed{\texttt{ego\_motion\_compensation\_failure}}
\]

相对 ICP 在 **纯物体运动**（C1）与 **混合运动**（C2）下显著闭合 effective state；**相机 ego-motion 抵消**（C0）未过 instrument 门。Relative branch 有强机制证据，但需先解决 C0 或接受 camera-static 子 regime。

## Regime 表（formal）

| Regime | 问题 | axis median | axis P90 | pos median | pos \(E_p\) | L gate |
|--------|------|------------:|---------:|-----------:|------------:|--------|
| **C0** | ego-motion cancel | **19.9°** | **35.1°** | 3.0 cm | 0.031 | **L2 FAIL** |
| **C1** | pure relative | **4.5°** | **14.7°** | **0.66 cm** | **0.010** | **L3 PASS** |
| **C2** | mixed | **4.7°** | **14.6°** | **0.74 cm** | **0.010** | **L3 PASS** |

## 门控

| Gate | 结果 |
|------|------|
| **L1a** observation | PASS（\(P(V_0\land V_1)=1.0\)；mask nonempty=1.0） |
| **L1b** excitation | PASS（identity median **20.0°**，P90 **35.0°** > 正式门） |
| **L2** C0 instrument | **FAIL**（axis 19.9°/35.1°；pos 3.0 cm/6.0 cm） |
| **L3** C1 / C2 | **均 PASS** |

## vs absolute baseline（descriptive）

Frame-1 **R6 A1** \(n^{(1)}\)（不参与 rescue）：

| Regime | A1 \(n^{(1)}\) median | A1 P90 |
|--------|----------------------:|-------:|
| C0 | 25.3° | 45.2° |
| C1 | 20.3° | 36.1° |
| C2 | 19.9° | 35.4° |

相对 ICP（C1/C2）将 axis median 从 ~20° 压至 **~4.5°**，position 从 ~2–3 cm 级压至 **<1 cm**。

## Diagnostics

- \(\rho_{\mathrm{overlap}}\) median：C0 **0.04**；C1/C2 **~0.48**（GT warp 与 \(\mathcal P_1\) 重叠充足）
- C0 低 overlap 与 ego-motion 失败一致

## Pattern

```text
pattern = ego_motion_compensation_failure
unlocks_relative_tracking_branch = false
unlocks_o0e1 = false
unlocks_o1 = false
```

**解读（按预注册）**：C1+C2 PASS 表明 partial-cloud rigid registration **在已知 anchor 条件下**可闭合 \(\Delta T\)；C0 FAIL 更可能来自 **viewpoint-induced correspondence collapse**（\(\rho\approx0.04\)），而非 \(T_{BC}\) 补偿本身错误。见 **REL0A** 审计。

## 机制读法（REL0A 后修正，不改 formal pattern）

正式 pattern 仍为 `ego_motion_compensation_failure`，但 diagnostics 支持区分：

\[
\underbrace{\text{ego-coordinate compensation}}_{\text{未必是主因}}
\quad\text{vs}\quad
\underbrace{\text{viewpoint-induced correspondence support}}_{\text{C0 主信号}}
\]

| 证据 | 数值 |
|------|------|
| C0 \(\rho_{\mathrm{overlap}}\) | **0.04** |
| C1/C2 \(\rho_{\mathrm{overlap}}\) | **~0.48** |
| C2 同样 \(T_{BC,0}\neq T_{BC,1}\) 但 ICP | **4.7°/14.6°** PASS |

**REL0A**（`runs/rtwx_o0rel0a/`）：
- **A** GT-mask ICP C0：仍 **20.0°/35.1°** → segmentation **排除**为主因
- **B** CAD 一致性：\(d_0,d_1\approx0.18\) cm \(\ll D_O\)，但 \(\rho\approx0.04\) → 两帧各自几何正确、表面不重叠
- **C** pooled：\(\rho<0.1\Rightarrow e_{\Delta n}\sim20°\)；\(\rho>0.25\Rightarrow e_{\Delta n}\sim4.8°\)

**架构含义**：relative tracking validity \(\approx\) **shared-surface observability**；低 overlap 时应 abstain / re-anchor，而非继续调 ICP。

## 正证据（不因 C0 FAIL 忽略）

\[
\boxed{C1:\ 4.5^\circ,\ 0.66\text{ cm}\qquad C2:\ 4.7^\circ,\ 0.74\text{ cm}}
\]

vs absolute A1 \(\sim20^\circ\)：在有足够跨帧几何对应时，relative state recovery **显著易于** single-frame absolute recovery。

## 实现注记

- 采集：RoboTwin 600 pairs 完成；`chdir` 后相对路径 bug 已修（`_resolve_apr`）
- L1b excitation 门逻辑已与预注册对齐（identity 须 **不过** 15°/30° 正式门）
