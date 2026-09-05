# RTWX-O0G4 预注册 — Sparse Geometric Orientation（G0：Oracle Sparse Keypoints）

日期：2026-08-30  
状态：**已冻结（本文件只开 G0）**  
依赖：O0G3=`oracle_orientation_geometry_supported`；O0G3R=`coverage_failure`；O0G3B=`pixel_coord_not_representable`；O0G3A=`correspondence_availability_characterized`  
**禁止**：dense \(x^O\)；direct `RGB→R` 扩网；改 \(N_{\min}=50\) 救 O0G3R；ICP/RANSAC 主路径；CAD descriptor 本格；arch sweep；解锁 O1/O0C2。

## 科学问题（G0 only）

> 在当前 `head+observer`、64² RGB-D 合同下，**预注册的 4 个 CAD keypoints** 经 GT pose 变换后，仅用 visible sparse correspondences + Kabsch，是否足以稳定恢复 \(R_{BO}\)？

\[
k_i^B = R_{BO}^{\mathrm{GT}} k_i^O + p_{BO},\quad
\{k_i^O,k_i^B\}_{i\in V}
\xrightarrow{\mathrm{Kabsch}}
\hat R
\]

**不**训练网络。Learned 2D landmarks + depth back-projection = **O0G4R**（仅 G0 PASS 后新预注册）。

CAD descriptor matching **不在本格**；仅当 G0 显示 sparse features **不可观测** 时作为下一层备选。

## 为何是 sparse，不是 dense / 不是 CAD descriptor

O0G3B：lookup→\(x^O\) 能记；\((u,v,d)\) 与 crop-RGBD **不能** memorization → dense canonical 过强。  
O0G3：正确 geometric evidence → \(R\) 无问题。  
O0G3A：\(N_{\min}=50\) 是 dense 合同；sparse **不沿用**。

## 冻结合同（G0）

| 项 | 值 |
|----|------|
| \(\mathcal O_t\) | **`head` + `observer`** |
| CAD keypoints | **4 点**，由 `021_cup/model_data0.json` AABB **运行前冻结**（bottom；rim \(+X\)；rim \(+Z\)；rim \(-X\)） |
| \(k_i^B\) | oracle：\(R^{\mathrm{GT}}k_i^O+p\) |
| 可见性 | mask 内 depth 点到 \(k_i^B\) 的最小距离 \(<\tau_{\mathrm{vis}}=1.5\,\mathrm{cm}\)；双视角 **OR** |
| 几何合同 | \(N_{\mathrm{vis}}\ge N_{\mathrm{geom}}=3\) 且非共线（**不是** \(N_{\min}=50\)） |
| 融合 | visible sparse 点 **一次 Kabsch** |
| 禁止 | 训练；从 test 选点；改 \(\tau_{\mathrm{vis}}\) 扫参 |
| 数据 | 复用 O0G3R **test** cache（seeds **29601/02/03**；12×120）；无新采集 |
| 规模 | 3 seeds × 1440 frames；**无训练** |

Learned O0G4R 将使用 **fresh seeds 30601/02/03**（本格不跑）。

## Gates

| Gate | 条件 |
|------|------|
| G0-cov | agg \(P(V^{any})\ge0.98\)；每 seed \(\ge0.95\)；\(P(N_{\mathrm{vis}}\ge3\land\neg\mathrm{degen})\ge0.90\)；每 seed \(\ge0.85\) |
| G0-ori | 在 enough 帧上：\(\mathrm{median}\,e_R\le15^\circ\)，\(P_{90}\le30^\circ\) |

必报：\(P(e_R\in[75^\circ,105^\circ])\)；每点可见率；deficit 帧 \(N_{\mathrm{vis}}\)。

## Patterns（G0，互斥）

| Pattern | 条件 | 下一格 |
|---------|------|--------|
| `sparse_keypoints_unobservable` | ¬G0-cov | CAD descriptor matching（新预注册） |
| `sparse_geometry_insufficient` | G0-cov ∧ ¬G0-ori | 审查 state/observability |
| `sparse_oracle_orientation_supported` | G0-cov ∧ G0-ori | 允许预注册 **O0G4R** |

## 解锁

G0 PASS → 允许预注册 **O0G4R**：`RGB → 2D sparse landmarks` + `depth+K,T → 3D` + Kabsch。  
G0 FAIL（unobservable）→ **STOP**；禁止直接训 heatmap。  
**O0C2 / O1 LOCKED**。
