# RTWX-AC1-D1 预注册 — State-Support Metric Breadth Audit

日期：2026-09-01  
状态：**已冻结**  
依赖：AC1-D0 = `state_support_metric_invalid`  
**禁止**：训练 policy；用 policy/expert AUROC 选 metric；调 \(k\)、\(\tau\) 分位数、phase 阈值；开 AC1-R0；把 AC0-B1 process 当 policy 输入来“救 D0”

## 问题

\[
\text{为什么 }d_{\rm NN}\text{ 在 cup/stamp 上正常，却在 cabinet 把 }20.6\%\text{ expert test 判成 OOD？}
\]

只做 instrument selection。Winner **不看** policy rollout。

## 数据

同一套 AC0 oracle expert 缓存。标准化 \(\mu_s,\sigma_s\) **冻结为 AC0-B0 train 统计**（与 D0 相同），四支共用。  
\(\tau_{95}=Q_{0.95}(d^{val})\) 每任务单独。

- **Selection**：AC0 split seed 50600 的 train/val/test。报告 test 上 expert false-OOD \(r_{\rm OOD}\)。
- **Confirmation**：新 episode split seed **52100**，仍 40/5/5；**不**重选 winner。在新 test 上再报 \(r_{\rm OOD}\)。

## Branches（\(\Delta=\) 距离几何）

| 支 | 度量 |
|----|------|
| B0 | 标准化 Euclidean kNN，\(k=5\)（D0 原仪器） |
| B1 | 标准化空间全局 Mahalanobis（ridge \(10^{-3}I\)） |
| B2 | 阶段条件 kNN：\(\phi\in\{approach,engage,transport,terminate\}\) 仅由**当前** process 谓词得到，无 \(t/T\)。\(c_{\rm goal}{\Rightarrow}term\)；grasp/contact\({\Rightarrow}\)engage；\(d_{EE,O}\ge 0.20{\Rightarrow}\)approach；否则 transport。相位内 \(k=5\)；该相训练点 \(<k\) 则回退全局 B0。 |
| B3 | 局部对角协方差 kNN：\(k_{\rm loc}=30\) 邻域 \(\mu,\sigma\)，距离 \(\|(x-\mu)/(\sigma+\varepsilon)\|_2\) |

B2 检验的是 \(p(s\mid task,\phi)\) 相对 \(p(s\mid task)\)，**不是** AC0 process→action 假说。

## Winner（仅 expert false-OOD）

合格：selection test **三任务** \(r_{\rm OOD}\le 0.10\)。  
多合格：更低的 \(\max_t r_{\rm OOD}\) → 更低均值 → **B0>B1>B2>B3**。  
无一合格：`winner=none`。

**禁止**用 AUROC(policy, expert) 选支。

## Confirmation

若 winner 非 none：新 split 上三任务仍 \(\le 0.10\) 才 `support_metric_qualified`。  
否则 `support_metric_unstable`。

无一合格：`support_geometry_unqualified`。

合格后才允许 **另格** 用该仪器重打 D0 的 G2/G3（本格不跑 policy）。R0 仍 LOCKED。
