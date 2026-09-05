# RTWX-O0G5B 报告 — Global-Context CAD Descriptor Instrument

日期：2026-08-30  
状态：**正式冻结** `global_canonical_identity_supported`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G5B_PREREG.md`  
产物：`runs/rtwx_o0g5b/{summary.json,run.json}`  
依赖：O0G5A=`local_descriptor_ambiguous`；复用 128² cache（seeds 31601/02/03）  
**无** fresh claim；**无** RANSAC；允许预注册 **O0G5C**；O0G5R / O0C2 / O1 LOCKED。

## 一句话

离散 CAD-anchor + InfoNCE **能记住训练集**：B0 闭合；B2 过 instrument 门。  
B1（local \(u,v,d,rgb\)）同样过门且与 B2 同量级——**不能**据此声称“只有 global context 才赋予身份”。

\[
\boxed{\texttt{global\_canonical\_identity\_supported}}
\]

## 仪器支路

Quantization ceiling（GT nearest of 512 FPS）：med **0.0335**。

| 支路 | med \(e^{norm}\) | P90 | Top-5 | 门 | 结果 |
|------|-----------------:|----:|------:|----|------|
| **B0** lookup | 0.0335 | 0.047 | 0.948 | acc≥0.99 | **PASS**（acc=**1.000**） |
| **B1** local \((u,v,d,rgb)\) | 0.045 | 0.715 | **0.830** | 描述 | PASS（消融） |
| **B2** local + PointNet global | **0.049** | 0.729 | **0.802** | med≤0.10 ∧ Top-5≥0.75 | **PASS** |

\(n=8192\) 点；K=512；CAD = mesh × scale（object-frame）。

## 读数

1. **Target 路径闭合**：B0 完美复现 label；e_match 贴 quantization → 不是 pipeline bug。
2. **相对 O0G3B**：连续 \(x^O\) 记不住；**离散 region identity** 在 train 上可记。这支持“降低 target 难度”而不是加大回归网。
3. **B1 ≈ B2，且 B1 略优**：train 上 **没有** 出现 “local≈随机、global≫local”。不能把 O0G5A 的 FPFH 失败直接写成“必须 object-level context”。B1 输入含 **图像坐标 \((u,v)\)**，与 FPFH 曲率不是同一局部证据；in-sample 下 \((u,v)\) 可能在记像素位置。
4. **尾部仍在**：B1/B2 P90 \(\approx0.72\,D_O\)，中位过门、高分位远差于 quantization（0.047）。instrument PASS 不等于均匀可辨。
5. **解锁 O0G5C**：fresh matching（仍无 RANSAC）。O0G5C 必须分清 B1 是泛化还是位置记忆。**O0G5R LOCKED**。

## Pattern

```text
pattern = global_canonical_identity_supported
B0_lookup = PASS
B1_local = PASS (ablation; not primary)
B2_global = PASS
unlocks_o0g5c_prereg = true
unlocks_o0g5r = false
unlocks_o1 = false
```
