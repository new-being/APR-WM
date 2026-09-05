# RTWX-X0C 预注册 — 原生 qpos 闭环伺服输入诊断

日期：2026-08-27  
状态：**已冻结**；**正式 RAN 2026-08-27**；**`servo_signal_insufficient`**。报告：`REPORT/REP/RTWX/RTWX0C_REPORT.md`。  
依赖：X0/X0R/X0S/X0F 均为 **SAPIEN-in-RoboTwin** 诊断；本单元是首个 **native `take_action(..., action_type="qpos")`** 单元格。  
不做：潜变量/残差网络；TASK-XL；RGB；场景 \(\phi\)-ID；柜体接触动力学；R10；容量 \(R_P\) 声称。  
不做：`set_qf` / `get_qf` / `qf_post` / 指令力矩通道 / `qacc` 缓冲 / 滞后扫描。

## 科学问题

\[
\boxed{\textbf{RTWX-X0C — native qpos \(\to\) 有效控制器输入}}
\]

\[
\boxed{
\text{RoboTwin 原生 qpos 目标能否从位置/速度误差转换成稳定的有效控制器输入，}
\text{并足以支撑结构化自由空间机械臂动力学预测？}
}
\]

本单元是 **oracle 仿真器状态 / native-qpos 闭环诊断**，**不是**官方 RoboTwin 观测基准，**不是**力传感器级辨识。

## 坐标与执行器

- 后端 `robotwin`：**仅**调用 `take_action(action, action_type="qpos")`。
- 动作布局与 `take_action` 一致：左臂 qpos + 左夹爪 + 右臂 qpos + 右夹爪。夹爪保持当前值。
- 真实关节：`get_left_arm_real_jointState` / `get_right_arm_real_jointState`（**不是** drive target 的 `get_*_arm_jointState`）。
- \(\dot q\)：对应 articulation `get_qvel`，索引与真实 qpos 相同。
- 驱动：`get_drive_target`、`get_drive_velocity_target`（**禁止**未检查就默认 \(\dot q^{\mathrm{tar}}=0\)）、`get_stiffness`、`get_damping`、`get_drive_mode`、`get_force_limit`。
- TOPP 失败时 `topp_*_flag=False` 且 **跳过** `set_arm_joints`：整段 episode 判无效并按协议重采样。
- 驱动模式无法判定或混合/未知：pattern = **`controller_mode_unresolved`**，停止拟合假力矩。
- `force`：\(\hat\tau = K_p e_q + K_d e_v\)。`acceleration`：\(a_{\mathrm{drive}}=K_p e_q+K_d e_v\)，**不得**称为力矩；M3 按加速度驱动改写并记录。
- 主加速度：**仅** \( \ddot q_{\mathrm{fd}} = (\dot q_{\mathrm{post}}-\dot q_{\mathrm{pre}})/\Delta t_{\mathrm{control}} \)。\(\Delta t_{\mathrm{control}}\) 为一次 native `take_action` 内 `scene.step` 次数 \(\times\) 仿真步长（对 `scene.step` 计数，不改物理）。
- 自由空间：同 `put_object_cabinet` 具身，但 **禁止** 机器人–物体接触与自碰撞。接触 → 整段无效并重采样。禁止为凑门限而挑 episode。
- 主指标：**仅活动臂关节**；夹爪排除。
- \(\tau_{\mathrm{ID}}\)：Pinocchio `compute_inverse_dynamics`（及质量阵）给出的 **由观测运动 + 刚体模型所需的广义力**，**不是**测力传感器。

## 激励（冻结）

\[
q_j^{\mathrm{tar}}(t)=q_{j0}+\sum_{k=1}^{3} A_{jk}\sin(2\pi f_k t+\varphi_{jk}),\quad
f\in\{0.15,0.31,0.57\}\,\mathrm{Hz}
\]

\(\sum_k |A_{jk}|\le 10\%\) 关节行程；裁剪到限位内侧 10%。从无碰撞合法 qpos 起步。

## 规模（robotwin 正式 **冻结**，不得因曲线改）

| 分割 | episode | 每段 native qpos 动作 |
|------|---------|------------------------|
| train | **48** | **120** |
| val | **24** | **120** |
| test | **24** | **120** |

numpy 后端 **仅** 单测可减小 \(n\)。`step_lim` 必须 \(>120\)。

## 模型（无网络）

| id | 定义 |
|----|------|
| M0 | \(\hat{\ddot q}=0\)，故 \(E_{\ddot q}^{M0}=1\)（零加速度/恒等基线，NRMSE 相对 \(\mathrm{rms}(\ddot q)\)） |
| M1 | \(\ddot q_j=a_j e_{q,j}\)，train 上逐关节 LS |
| M2 | \(\ddot q_j=a_j e_{q,j}+b_j e_{v,j}\)，train 上逐关节 LS |
| M3 | 结构化 PD + 刚体；**记录的** \(K_p,K_d\)（不训练）。力模式：\(\hat\tau=K_p e_q+K_d e_v\)，\(\hat{\ddot q}=M^{-1}(\hat\tau-h)\)。加速度模式：\(\hat{\ddot q}=a_{\mathrm{drive}}\) |
| M4 | **仅辨识检验**：train 上对 \(\tau_{\mathrm{ID}}\) 拟合对角非负 \(K_p,K_d\)，冻结后用于 val/test。无满阵、无 NN |

\(e_q=q^{\mathrm{tar}}-q\)，\(e_v=\dot q^{\mathrm{tar}}-\dot q\)，目标来自 native 指令 / 驱动接口。

NRMSE：与既有 RTWX `_nrmse` 一致（RMSE / \((\mathrm{rms}(\mathrm{target})+10^{-8})\)）。

## 指标

- \(E_\tau=\mathrm{NRMSE}(\hat\tau,\tau_{\mathrm{ID}})\)（力模式）；\(\rho_\tau=\mathrm{corr}\)
- \(E_{\ddot q}\)；逐关节 \(E_j\)
- 滚动：从真 \((q,\dot q)\) 出发，**未来目标序列开环、状态闭环**，\(H\in\{10,50\}\)。\(E_{\mathrm{roll}H}\) 为预测 \((q,\dot q)\) 对记录状态的 NRMSE。

## 门限（收集前冻结）

| 门 | 规则 | 失败 pattern |
|----|------|----------------|
| **G0** | native qpos 确认；目标可读；刚度/阻尼已记录；驱动模式已知；有效数据接触率=0；激励非退化 | `instrument_failure` |
| **G1** | M2 留出 \(E_{\ddot q}\le 0.75\) 且 \(\ge 80\%\) 臂自由度 \(E_j<0.9\) | `servo_signal_insufficient` |
| **G2** | M3 **或** 预注册 M4：\(E_{\ddot q}<0.75\) 且 \(<\) 恒等，且 \(E_{\mathrm{roll}10}<\) 恒等。G1 过 G2 不过 → `servo_signal_only` / `control_signal_present_structure_open` | 结构未闭合 |
| **G3** | 若 G2 主张用 M4：train 增益冻结，test \(E_{\ddot q}\le 1.10\times\) val | 过拟合/不可辨识，不宣称物理参数恢复 |

驱动模式未解析：`controller_mode_unresolved`。

最终 pattern 之一：

`closed_loop_structure_closed` | `servo_signal_only` | `servo_signal_insufficient` | `controller_mode_unresolved` | `instrument_failure`

**仅 `closed_loop_structure_closed` 解锁 X0C1。** 无容量声称。不得在看到曲线后改冻结门限或正式 \(n\)。
