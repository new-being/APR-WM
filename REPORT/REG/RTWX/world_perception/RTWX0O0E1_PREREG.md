# RTWX-O0E1 预注册 — Multi-Symmetry Object Pose

日期：2026-08-31  
状态：**LOCKED** until O0E0=`effective_pose_supported`  
依赖：O0E0 在 `021_cup` 上闭合 \((p,[R]_{SO(2)_y})\)  
**禁止**：改 O0E0 门；统一强行 full \(SE(3)\)；开 O1 除非本格 PASS。

## 科学问题

> 不同对象能否共用**同一 object-state interface**，仅通过 symmetry metadata \(G_i\) 自动调整有效姿态自由度？

\[
\boxed{
s_i^{\mathrm{pose}}=(p_i,[R_i]_{G_i})
}
\]

## 冻结对象表（实现前不得改）

| 对象类型 | 示例 | \(G_i\) | 有效 orientation |
|----------|------|---------|------------------|
| sphere | — | 全旋转 | 不表示 orientation |
| 无柄杯 / 圆柱 | `021_cup` | \(SO(2)_y\) | 轴 \(n\in S^2\) |
| cuboid | — | 离散 \(G_{\mathrm{discrete}}\) | \(SO(3)/G_{\mathrm{discrete}}\) |
| 有柄 mug | — | \(\{I\}\) | full \(SO(3)\) |

## Gates（继承 O0E0 尺度）

每对象 fresh：axis/full-\(R\) error 与 position error 按该对象 \(G_i\) 定义；**不**用统一 full-\(R\) gate 判 FAIL。

## 解锁

PASS → 允许预注册 **O1**（persistent object memory on \(m_{i,t}=(p,[R]_{G_i},v,\ldots)\)）。  
FAIL → O1 LOCKED。
