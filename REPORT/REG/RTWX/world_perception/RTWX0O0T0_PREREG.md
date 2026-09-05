# RTWX-O0T0 预注册 — Temporal Yaw Observability

日期：2026-08-30（更新 2026-08-31）  
状态：**PREREGISTERED / NOT RUN / DEPRIORITIZED**  
原解锁：O0A0=`appearance_yaw_generalization_failure`  
**战略决定（2026-08-31）**：主线不再为 canonical yaw 消耗实验预算；本格**不判 FAIL**，保持未运行。

## 何时才值得跑

仅在任务**明确需要 yaw** 时（例如：杯柄、logo 朝向、方向性摩擦、多视角已知 \(T_{BC,t}=FK(q_t)\) 的非对称特征恢复）。  
对当前无柄近轴对称 `021_cup`，单帧 geometry 与 appearance 均已 chance → **低优先级**。

## 科学问题（冻结，不改写）

\[
RGBD_{t-L:t}+a_{t-L:t}\;\rightarrow\;\theta_{\mathrm{yaw},t}
\]

仍是 observability audit，不是 O1。

## 冻结合同（若未来解锁运行）

| 项 | 值 |
|----|------|
| 输入 | 短窗 RGB-D（head+observer）+ 本体 \(a\) 或 \(q\) |
| 相机运动 | 若用多视角，须显式 \(T_{BC,t}\)（FK）；固定外参-only 机械臂运动**不**构成 viewpoint baseline |
| 目标 | \(\theta_{\mathrm{yaw}}\)（与 O0A0 同 24×15° 或 circular error） |
| Split | 按 episode 分组 |
| Primary gate | median \(\le15^\circ\)，P90 \(\le30^\circ\)；报 \(f_{90}\) |

## 与主线关系

- **NEXT** 已改为 **O0E0**（symmetry-aware effective pose \((p,n)\)）。
- O0A1 保持 LOCKED。
- 运行 O0T0 **不**解锁 O0E0 / O1；须单独科学决策。
