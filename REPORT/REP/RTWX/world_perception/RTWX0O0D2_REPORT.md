# RTWX-O0D2 报告 — Perception Instrument Closure

日期：2026-08-29  
状态：**正式冻结** `image_memorization_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0D2_PREREG.md`  
产物：`runs/rtwx_o0d2/{run.json,metrics.json}`  
**无科学 claim**；**不跑 O0R**；**不解锁 O1**。O0 不改。

## 一句话

训练链 **不是**死的：index lookup \(NRMSE\sim10^{-7}\)。  
64×64 flatten 线性读出 **未**同时通过真/随机门（真 \(0.110\)，随机 \(0.395\)）。  
**CoordConv 无 GAP** 记住 \(p_x\)（\(0.011\)）；**GAP CNN 失败**（\(0.631\)）。  
**`head_camera` / observer / world 的 \(P_{\mathrm{FOV}}=1.0\)**；`front_camera` 仅 \(0.17\)。  
O0R 仍锁（B 未过）。

## 表

| 门 | 结果 |
|----|------|
| D2-A lookup 真/随机 | **过** \(9.8\times10^{-8}\) / \(6.0\times10^{-8}\) |
| D2-B flatten MLP 真/随机 | **不过** \(0.110\) / \(0.395\) |
| D2-C GAP | \(0.631\) 不过 |
| D2-C CoordConv+flatten map | **\(0.011\) 过** |
| FOV \(\max_c P_c\ge0.9\) | **过**（\(c^\star=\) `head_camera`，\(P=1\)） |

Pattern（互斥）：**`image_memorization_failure`**（A∧¬B）。  
`spatial_bottleneck_on_GAP=true`。`unlocks_o0r=false`。

## FOV（新 observation 候选，不改 O0）

| camera | \(P(\mathrm{FOV})\) |
|--------|---------------------|
| front_camera | 0.167 |
| **head_camera** | **1.000** |
| observer_camera | 1.000 |
| world_camera1/2 | 1.000 |
| any-camera | 1.000 |

腕部相机本合同未采集（`collect_wrist_camera=false`）。  
O0R 若开，须在预注册中冻结 **非 front** 的相机合同（例如 `head_camera`），不得回写 O0。

## 读数

1. **不是 training_pipeline_failure。** Lookup 已闭合 optimizer/evaluator。  
2. Flatten 在 O0 的 32 张 64² 上未把随机标签记死；**不能**用 flatten 失败否定「图像路径」。CoordConv 已把真 \(p_x\) 记到 \(0.011\)。  
3. GAP 与位置回归冲突，与 O0D1 tiny CNN 失败机制一致。  
4. **front_camera 无资格作 O0 主相机**；覆盖已有单相机达到 0.9（head）。  
5. 下一步若要 `instrument_closed`：让 D2-B 过（更多 unique 帧/epoch）**或**把 memorization 合同改到 spatial-aware（须新预注册），再开 O0R。本格不改门、不开 O0R。
