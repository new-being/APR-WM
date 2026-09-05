# RTWX 是什么

**RTWX 不是领域标准缩写。** 本仓库里它表示：

\[
\text{RTWX} \approx \text{RoboTwin Experiments / RoboTwin X-series}
\]

即 **在 RoboTwin 上跑的验证实验线**，不是论文术语。

科学问题已经分成两条，不要再往 `X0RGB2/3/...` 堆：

| 支线 | 问题 | 目录 |
|------|------|------|
| **robot_dynamics** | \(s^R=(q,\dot q)+a\to s^{R'}\) | `robot_dynamics/` |
| **rgb_robot_probe**（关闭） | \(RGB\to(q,\dot q)\) | `robot_dynamics/rgb_robot_probe/` |
| **world_perception** | \(RGB\to s^O\) | `world_perception/` |
| **object_dynamics** | \(s^O\) 预测 / interaction | `object_dynamics/` |
| **attention_compute** | task/causal/risk → 算力 | `attention_compute/` |
| **wam** | 多速率 state/action 调度 | `wam/` |

论文结构对应：Robot dynamics → World perception → Object dynamics → Adaptive attention/compute。

文件名（`RTWX0EH1_REPORT.md` 等）**不改**。旧扁平路径见各 `MOVED.md` 指针。
