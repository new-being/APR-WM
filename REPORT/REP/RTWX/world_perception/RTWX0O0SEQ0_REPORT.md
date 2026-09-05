# RTWX-O0SEQ0 报告 — Relative Propagation Horizon Audit

日期：2026-08-31  
状态：**正式冻结** `relative_sequence_supported`（tag：`stable_to_16`）  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0SEQ0_PREREG.md`  
产物：`runs/rtwx_o0seq0/`；seed **37609**；100 seq × \(T=16\)

## 一句话

\[
\boxed{
H^*=16,\quad
\texttt{relative\_sequence\_supported}
}
\]

Near-upright anchor + frozen adjacent relative chain **在全 checkpoint 维持 formal gate**；absolute 全线 ~23°–31° FAIL。

## Primary outcome

\[
\boxed{H^*=16}
\]

GT-mask diagnostic（20 seq）：\(H^*_{\mathrm{GTmask}}=16\)（segmentation 非瓶颈）。

## Horizon 表（A0 chain / A1 direct / A2 absolute）

| H | Chain med/P90 | Direct med/P90 | Absolute med/P90 | Chain pos med |
|---|--------------|----------------|------------------|---------------|
| 1 | **2.9°/6.7°** | 2.9°/6.7° | 23.4°/36.2° | 2.8 cm |
| 2 | **3.3°/6.9°** | 3.2°/6.5° | 25.1°/36.2° | 2.9 cm |
| 4 | **4.2°/8.0°** | 3.7°/7.3° | 26.0°/39.9° | 2.9 cm |
| 8 | **6.0°/10.9°** | 4.9°/8.5° | 29.7°/62.0° | 3.1 cm |
| 16 | **8.0°/16.1°** | 6.9°/24.5° | 31.0°/71.4° | 3.3 cm |

**Chain 与 direct 在 H≤8 几乎重合**；H=16 时 chain P90 **优于** direct（16.1° vs 24.5°）——长期 direct matching 尾部更差，**积分 drift 温和、非主失效模式**。

## 门控

| Gate | 结果 |
|------|------|
| L1 observation | PASS（vis=1.0，adj=1.0） |
| L2 anchor | PASS（\(e_{n_0}\) med **2.4°**；\(e_{p_0}\) med **2.7 cm**） |
| Chain \(H^*\) | **16**（全 horizon PASS） |

## 机制解读

1. **Relative-first 可作为 persistent backbone**：无需每帧 absolute reconstruction。
2. **HYB0 router 路线可放弃**：absolute A1 ~23° 全线劣于 chain；hybrid 失败因切回 absolute。
3. **Drift 存在但可控**：chain median 2.9°→8.0° over 16 steps，仍过 15°/30° 门。
4. **Direct anchor 在 H=16 尾部劣于 chain**：下一格 sparse re-anchor 仍值得做，但非紧急。

## Pattern

```text
pattern = relative_sequence_supported
tags = [stable_to_16]
unlocks_relative_sequence_branch = true
unlocks_o0e1 = false
```

## Beam 含义

主线确立：

\[
\boxed{
\text{cheap anchor} \rightarrow \text{relative propagation} \rightarrow \text{uncertainty-gated sparse re-anchor}
}
\]

下一格候选：**sparse/log-history re-anchor** 或 **belief uncertainty accumulation**（非 router、非 persistent surface DFS）。
