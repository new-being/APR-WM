# RTWX-O0G5C 报告 — Fresh Canonical Identity Generalization

日期：2026-08-30  
状态：**正式冻结** `canonical_identity_generalization_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G5C_PREREG.md`  
产物：`runs/rtwx_o0g5c/{summary.json,run.json,cache_o0g5c_s*}`  
依赖：O0G5B instrument PASS；train cache 31601/02/03；fresh **33601/02/03**；128²；K=512；同一 B1/B2 / InfoNCE  
**无** RANSAC；**无** Kabsch primary；**无** pose rescue。O0G5R / O0C2 / O1 LOCKED。

## 一句话

离散 CAD-anchor 在 **fresh episodes 上不过 matching 门**。B1≈B2，都 FAIL：不是 “缺 global context”，而是 **train 上可记的 identity 没有变成跨 episode 的视觉/几何泛化**。

\[
\boxed{\texttt{canonical\_identity\_generalization\_failure}}
\]

对应预注册情况 **C**。

## Gates

| 支路 | 条件 | 结果 |
|------|------|------|
| G0-support | \(P(N_{\mathrm{usable}}\ge64)\ge0.90\)；每 seed ≥0.85 | **PASS**（0.915；seeds 0.915 / 0.979 / **0.852**） |
| B1 fresh | med \(\le0.10\) ∧ Top-5 \(\ge0.75\) | **FAIL**（med **0.119**；Top-5 **0.609**） |
| B2 fresh | 同上 | **FAIL**（med **0.119**；Top-5 **0.596**） |

Quantization 不是借口：fresh 最近锚点 med **0.033** \(D_O\)（与 train 同量级）。

## 主表（fresh，\(n=16384\)）

| 支路 | med | P90 | Top-1 acc | Recall@0.05 | Top-5 | Recall@0.10 |
|------|----:|----:|----------:|------------:|------:|------------:|
| **B1** \((u,v,d,rgb)\) | **0.119** | 0.799 | 0.238 | 0.269 | **0.609** | 0.467 |
| **B2** + \(g_{\mathrm{object}}\) | **0.119** | 0.797 | 0.237 | 0.265 | **0.596** | 0.463 |
| O0G5B train B1（对照） | 0.045 | 0.715 | — | — | 0.830 | — |
| O0G5B train B2（对照） | 0.049 | 0.729 | — | — | 0.802 | — |

## 机制 audit

**B1-uv-shuffle**（打乱 test \((u,v)\)，保持 \(d,rgb\)）：med **0.331**，Top-5 **0.446**（\(\Delta\)Top-5 **−0.164**）。  
\((u,v)\) 在 fresh 上仍有贡献，但 **带着图像位置也不过门**。

\(H(Y\mid U,V)\)：train **3.92**（44 cells）→ fresh **4.39**（38 cells）。条件分布更混，但 train 上本来就不是低熵 lookup。

Mask area：无 \(A>1024\)（large \(n=0\)）。small / medium 都 FAIL，且几乎同量级——不是 “aggregate 过、小目标崩”。

## 读数

1. **情况 C，不是 A**：要声称 global context 关键，需要 B1 崩、B2 过。这里 B2 没有优于 B1。
2. **也不是 “完全随机”**：Top-5 \(\approx0.61\) ≫ \(5/512\approx0.01\)。有可迁移信号，但不够过预注册门。
3. **O0G5B 解决的是 optimization，不是 observability**：离散 identity 能记 train；fresh 后中位从 0.045 升到 0.119，Top-5 从 0.83 降到 0.61。
4. **P90 \(\approx0.80\,D_O\)**：中位刚过 0.10 的边、尾部仍是对侧/mode error。这与 O0G5B 的重尾、以及更早的 orientation P90 同属结构问题，不是本格新开的门。
5. **不开 O0G5R**。RANSAC 不能从不够用的 candidate matches 里创造 canonical identity。

## Pattern

```text
pattern = canonical_identity_generalization_failure
G0_support = PASS
B1_local_fresh = FAIL
B2_global_fresh = FAIL
B1 ≈ B2  (no global_context_supported)
unlocks_o0g5r_prereg = false
unlocks_o1 = false
```
