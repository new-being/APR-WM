# RTWX-O0G4 报告 — Sparse Geometric Orientation（G0 Oracle Sparse Keypoints）

日期：2026-08-30  
状态：**正式冻结** `sparse_keypoints_unobservable`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G4_PREREG.md`  
产物：`runs/rtwx_o0g4/{summary.json,run.json,metrics.json}`  
合同：4 个冻结 CAD keypoints；\(\tau_{\mathrm{vis}}=1.5\,\mathrm{cm}\)；\(N_{\mathrm{geom}}=3\)；复用 O0G3R test cache（29601/02/03）；**无训练**  
**不**解锁 O0G4R / O0C2 / O1。

## 一句话

可见时 **3 个 sparse CAD 点 + Kabsch 以机器精度恢复 \(R\)**（med/P90 \(e_R=0\)，\(f_{90}=0\)）。  
失败在 **availability**：\(P(N_{\mathrm{vis}}\ge3)=0.694<0.90\)。看得见杯子 ≠ 同时看见 3 个预注册 landmark。

\[
\boxed{\texttt{sparse\_keypoints\_unobservable}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-cov | any≥0.98；enough≥0.90；每 seed enough≥0.85 | **FAIL**（any=**1.000** ✓；enough=**0.694** ✗） |
| G0-ori | enough 帧 med≤15°，P90≤30° | **PASS**（med **0**；P90 **0**） |

按预注册互斥链：\(\neg\)G0-cov → `sparse_keypoints_unobservable`（STOP；不训 heatmap）。

## 主表

| | med \(e_R\) | P90 | \(f_{90}\) | n (enough) |
|--|------------:|----:|-----------:|-----------:|
| 聚合（enough 帧） | **0** | **0** | **0** | 2996 / 4320 |

| seed | \(P_{\mathrm{enough}}\) | median \(N_{\mathrm{vis}}\) | bottom | rim+X | rim+Z | rim−X |
|------|------------------------:|----------------------------:|-------:|------:|------:|------:|
| 29601 | 0.774 | 3 | 0.333 | 0.862 | 0.842 | 0.795 |
| 29602 | 0.568 | 3 | 0.118 | 0.847 | 0.701 | 0.753 |
| 29603 | 0.738 | 3 | 0.256 | 0.907 | 0.574 | 0.906 |

## 读数

1. **几何天花板对 sparse 成立**：与 O0G3 dense oracle 同结论——**不是 Kabsch 病态**。需要的视觉信息可以少到 3 个非共线点。
2. **阻塞是观测合同**：bottom 可见率仅 0.12–0.33；四 rim/bottom 难以同时 ≥3。这与 O0G3A 的小 mask regime 一致，但是 **新合同**（\(N_{\mathrm{geom}}=3\)），不是把 50 改成 3。
3. **不解锁 O0G4R**：G0 未过 coverage；禁止直接训 2D landmarks。
4. **下一层备选**：按预注册 fork，进入 **CAD descriptor matching**（新预注册），而不是降 \(N_{\mathrm{geom}}\) 或放宽 \(\tau_{\mathrm{vis}}\)。
5. **O0C2 / O1 LOCKED**。

## Pattern

```text
pattern = sparse_keypoints_unobservable
G0_ori = PASS (med=P90=0 on enough frames)
G0_cov = FAIL (P_enough=0.694)
unlocks_o0g4r_prereg = false
unlocks_o1 = false
```
