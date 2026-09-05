# RTWX-O0 报告 — Explicit Object-State Observability

日期：2026-08-29  
状态：**正式冻结** `object_pose_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0_PREREG.md`  
产物：`runs/rtwx_o0/{run.json,metrics.json,cache_o0_{train,val,test}.npz}`  
任务：`place_empty_cup`；对象 `env.cup`；seed **16601**；48/24/24 × 120；`front_camera` **224×224**；\(L=4\)。  
仪器：ResNet18 + spatial + temporal head；**无 M2**。  
**不**估计 \(s^R=(q,\dot q)\)；不声称真实相机。

## 一句话

第三人称 RGB **没有**恢复 manipulation 所需的显式 \(s^O=(p,R,v,\omega)\)。  
B2 与 train-mean 对照 B0 **数值上重合**（\(E_p=0.536\)，median \(\|e_p\|=25.6\) cm），相对 B0 无信息增益。  
Pattern：**`object_pose_failure`**。不解锁 O1。

## Gates（阈值未放宽）

| Gate | 条件 | 结果 |
|------|------|------|
| G0 coverage | test 满额；位姿有限；train \(\mathrm{std}(p_x)+\mathrm{std}(p_y)>0.02\) m | **通过**（24 ep；\(\mathrm{std}(p_x)+\mathrm{std}(p_y)=0.747\) m） |
| G1 position | \(E_p\le 0.30\) | **失败**（\(E_p=0.536\)） |
| G2 orientation | median \(e_R\le 15^\circ\) 且 \(P_{90}\le 30^\circ\) | **失败**（median \(7.94^\circ\)，**\(P_{90}=118.7^\circ\)**） |
| G3 velocity | \(E_v\le 0.50\) 且 \(E_\omega\le 0.60\) | **失败**（\(E_v=1.000\)，\(E_\omega=1.000\)） |
| G_info | B2 \(E_p\) 与 median \(e_R\) 均 \(\le 0.85\times\) B0 | **失败**（比约 \(1.00\times\)） |

G2 的 median 单独看会过 \(15^\circ\)，**\(P_{90}\) 未过**；按预注册合取，G2 失败。  
因 ¬G1 ∧ ¬G2 ∧ ¬G_info，G3 失败不单独定义新 pattern。

## 主表（test，2880 帧）

| 仪器 | \(E_p\) | median \(\|e_p\|\) (cm) | median \(e_R\) | \(P_{90}(e_R)\) | \(E_v\) | \(E_\omega\) |
|------|--------:|------------------------:|---------------:|----------------:|--------:|-------------:|
| B0 train-mean pose | 0.536 | 25.64 | \(7.93^\circ\) | \(118.7^\circ\) | 1.000 | 1.000 |
| B1 上一帧 GT（静态诊断） | 0.036 | 0.00 | \(0.00^\circ\) | \(0.00^\circ\) | 1.353 | 1.404 |
| **B2 RGB（主）** | **0.536** | **25.63** | **\(7.94^\circ\)** | **\(118.7^\circ\)** | **1.000** | **1.000** |

B2 / B0：\(E_p\) 比 \(0.9997\)；median \(e_R\) 比 \(1.001\)。  
编码器在本格设定下落到 **常值均值预测**，像素未提供可测 pose 信息。

## 读数

1. **G0 通过：问题不是「杯子没动、没跨 episode 变化」。**  
   train 首帧 \(p_{xy}\) 有跨 episode 散布（\(p_x\) 全距 \(0.59\) m）。多数 episode 内杯子几乎静止（within-ep \(\|Δp\|\) 中位 \(\approx 0\)），少数有大幅位移（train p90 \(0.62\) m）。B1 的 \(E_p=0.036\) 说明 **逐帧 GT 几乎可抄上一帧**；要赢的是 **跨 episode 的绝对位姿**，不是帧间增量。
2. **本格否掉的是「224 ResNet18 + \(L=4\)」对 \(s^O\) 无信息增益**，不是否掉「世界必须有物体状态」。B0 的 median \(e_R\approx 8^\circ\) 只说明杯子多数朝向接近均值；\(P_{90}\approx 119^\circ\) 说明尾部朝向完全没被均值（也没被 RGB）抓住。
3. **不解锁 O1。** 预注册：G1/G2 通过才进入 persistent filtering。本格未过。  
   **不**回到 \(RGB\to(q,\dot q)\)；**不**改 M2 / \(K=4\)；**不**把本格写成 X0RGB2。  
   下一格若继续 WORLD 线，需要 **新预注册**（仍问 \(RGB\to s^O\) 的仪器，而不是滤波或 object dynamics）。

## 账本

```text
RTWX-O0 = RAN / object_pose_failure
          place_empty_cup; seed 16601; 224×224 ResNet18; L=4
          G0=true; G1=false (E_p=0.536); G2=false (P90 e_R=118.7°)
          G3=false; G_info=false
          B2 ≈ B0 (no RGB information gain)
```
