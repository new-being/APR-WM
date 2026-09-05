# RTWX-O0HYB0 报告 — Prior-Anchored Hybrid State Update

日期：2026-08-31  
状态：**正式冻结** `hybrid_state_update_insufficient`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0HYB0_PREREG.md`  
产物：`runs/rtwx_o0hyb0/`；cal **37607** / formal **37608**；800 pairs

## 一句话

\[
\boxed{\texttt{hybrid\_state\_update\_insufficient}}
\]

**架构互补性成立**（oracle 四 regime 全 PASS），**non-oracle router 判别力强**（AUROC **0.95**），但 formal hybrid 在 **H01/H11**（object move）P90 尾部未过门。

## 四支方法表（formal，axis median / P90 / pos median cm）

| Method | H00 | H01 | H10 | H11 |
|--------|-----|-----|-----|-----|
| **Absolute** | 22.1°/26.3°/3.9 | 23.2°/39.9°/3.7 | 22.0°/26.2°/3.8 | 25.4°/37.6°/4.0 |
| **Relative** | **2.7°/6.1°/2.5** | **5.9°/12.3°/2.6** | **2.3°/6.4°/2.5** | **5.8°/12.6°/2.6** |
| **Oracle** | **2.7°/6.1°/2.5** | **5.7°/11.9°/2.6** | **2.3°/6.4°/2.5** | **5.8°/12.6°/2.6** |
| **Hybrid** | **2.7°/6.1°/2.5** | 7.9°/**37.0°**/2.7 | **2.3°/6.4°/2.5** | 6.5°/**31.9°**/2.9 |

## 门控

| Gate | 结果 |
|------|------|
| **L1** prior anchor | **PASS**（\(e_{n_0}\) med **2.6°** P90 **6.0°**；\(e_{p_0}\) med **2.5 cm**） |
| **L2** oracle hybrid | **PASS**（四 regime 全过） |
| **L3** formal hybrid | **FAIL**（H01/H11 P90 **37°/32°** > 30°） |

## Router

- \(\tau^*=\) **0.449**（cal Youden，冻结）
- AUROC(\(q_{\mathrm{rel}}, y_{\mathrm{safe}}\)) = **0.954**
- Formal relative selection rate：**82%**
  - H00 **100%** rel；H10 **100%** rel（large cam + static → 正确全选 relative）
  - H01 **61%** rel；H11 **67%** rel（object move 时仍有 ~1/3 误选 absolute → 尾部恶化）

## 架构结论（修正措辞）

**更强的事实**：relative-only 在 H00–H11 **四格全 PASS**；absolute ~22° 全线 FAIL。Formal hybrid 失败来自把部分 relative 输出替换成 absolute；oracle 几乎与 relative-only 重合。

因此主假设转向：

\[
\boxed{
\text{cheap anchor}\rightarrow\text{relative propagation}\rightarrow\text{sparse re-anchor}
}
\]

**不再**以 absolute A1 为默认 fallback。Router 实验（HYB0）已闭合；下一格 **SEQ0** horizon audit。

## Pattern

```text
pattern = hybrid_state_update_insufficient
unlocks_hybrid_world_state_branch = false
unlocks_o0e1 = false
```

## 下一步（beam，非本格 scope）

- 不改 ICP / A1；router 特征或 conservative policy（低 \(q_{\mathrm{rel}}\) 强制 abstain 而非 abs fallback）
- sequence / drift 格仅在 hybrid one-step 闭合后再开
