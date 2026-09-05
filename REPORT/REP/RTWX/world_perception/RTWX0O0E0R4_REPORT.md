# RTWX-O0E0R4 报告 — Multi-Observation Reference Test

日期：2026-08-31  
状态：**正式冻结** `multiview_reference_insufficient`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0R4_PREREG.md`  
产物：`runs/rtwx_o0e0r4/`；multiview test seed **37604**；R3 diagnostic seed **37603**；\(S_0\) natural UNet；B2 **冻结**

## 一句话

\[
\boxed{\texttt{multiview\_reference\_insufficient}}
\]

**朴素多视角 cloud union 不能把 formal axis 压过 15°/30° 门**（\(K=8\) median **19.7°**，仍 FAIL）。但 center–\(\lambda\) diagnostic 给出强机制曲线：当前 reference center 误差 **3.50 cm** 时 axis median **28.1°**；center 插值到 GT 一半（\(\lambda=0.5\)）即 **14.6° PASS**。position 全程 PASS（~3.2–3.5 cm）。

## Center–axis diagnostic（R3 cache，非 formal claim）

| \(\lambda\) | median \(e_{\mathrm{axis}}\) | P90 |
|-------------|-------------------------------|-----|
| 0.0（\(\hat p_{\mathrm{ref}}\)） | **28.1°** | 39.0° |
| 0.25 | 21.6° | 30.5° |
| **0.5** | **14.6°** | 20.1° |
| 0.75 | 7.3° | 10.3° |
| 1.0（GT \(p\)） | 0.17° | 0.26° |

\[
e_{\mathrm{center}}^{\mathrm{ref}}\text{ median}=3.50\text{ cm},\quad
P90=4.88\text{ cm}
\]

**解读**：axis 对 center 高度敏感；task-sufficient position（~3.5 cm）远低于 representation-internal reference 需求（\(\lambda=0.5\) 对应约 **1.75 cm** 有效 center 才跨 15° 门）。这与 R3「position PASS / axis FAIL」完全一致。

## Multiview formal（fresh seed 37604）

| \(K\) | median \(e_{\mathrm{axis}}\) | P90 | median \(e_p\) | G1 | G2 |
|-------|------------------------------|-----|----------------|----|----|
| 1 | **20.5°** | 33.9° | 3.28 cm | FAIL | PASS |
| 2 | 21.7° | **91.9°** | 3.51 cm | FAIL | PASS |
| 4 | 21.3° | 36.6° | 3.43 cm | FAIL | PASS |
| 8 | 19.7° | **84.9°** | 3.20 cm | FAIL | PASS |

- \(K=1\) sanity **~20°**（R3 级 **22°**）✓  
- \(K\) 增大 **无单调改善**；\(K=2,8\) P90 灾难性膨胀  
- 全部 position PASS；**无一 \(K\) formal PASS**

## Pattern

```text
pattern = multiview_reference_insufficient
patterns = [multiview_reference_insufficient, single_observation_reference_coupled]
unlocks_o0e1_prereg = false
unlocks_o1 = false
```

对应预注册 **情景 C**：简单 \(\mathcal P_1\cup\cdots\cup\mathcal P_K\) 不足。

## 机制结论

```text
qualified data → S₀ seg → good cloud → reference/centering (~22°)
  → naive multiview union (K≤8) → still ~20°  [本格否定]
  → center must ≲1–2 cm (λ≈0.5) for axis gate  [diagnostic]
  → frozen quotient B2 (oracle-viable)
```

1. **不是**「再多几个等权视角就能闭合 reference」——需 visibility-aware weighting / persistent surface / learned reference。  
2. **是** center–quotient **强耦合**——center 改善到 GT 一半即可过门，说明 B2 本身仍 viable。  
3. **task-sufficient ≠ representation-internal reference accuracy** 再次实证。

## 措辞（硬）

| 能说 | 不能说 |
|------|--------|
| naive multiview reference 不足 | multiview_reference_supported |
| center–axis 灵敏度曲线 \(e_{\mathrm{axis}}=f(e_{\mathrm{center}})\) | O0E1 已解锁 |
| \(K=1\) sanity ~20° 复现 R3 | 简单 union 已救 O0E0 |
| 下一步需 visibility / persistent model / learned center | B2 objective 应 retune |

## 下一步（机制导向，未 prereg）

\[
\boxed{
\text{visibility-aware / persistent surface reference，或显式 center estimator}
}
\]

**O0E1 / O1 仍 LOCKED**。B2 / \(S_0\) 冻结。
