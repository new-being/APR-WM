# RTWX-X0D 报告 — Native Closed-Loop Structure Test

日期：2026-08-27  
状态：**正式冻结** \(\texttt{low\_order\_structure\_insufficient}\)；**禁止回头修改 M1–M3**  
预注册：`REPORT/REG/RTWX/RTWX0D_PREREG.md`  
产物：`runs/rtwx_x0d/{run.json,header.json,run.header.txt,metrics.json,summary.json,run.log}`  
不做：网络 / residual；\(\tau\) 重建；\(\ddot q\) 主指标；容量；TASK-XL；RGB；R10

## 一句话

在冻结规模（48/24/24 × 120 native `qpos`）上，**G0 通过**（恒等 \(E_1\approx 0.95\)，状态转移非平凡）。  
预注册的低阶闭环族 **M1/M2/M3 全部劣于恒等**：test \(E_1\approx 1.52 > 0.75\times 0.947\)。  
因此 **不能** 声称 native \((q,\dot q,q^{tar})\to(q',\dot q')\) 已被这组有效闭环结构坐标压缩。X0D1 **未解锁**。按预注册：**STOP 本低阶闭环结构族**。

本单元是 **oracle 仿真器状态 / native-qpos 闭环结构诊断**，**不是**官方观测基准，**不是**质量/惯量/力矩增益辨识。

## 科学合同

\[
x_t=(q_t,\dot q_t),\quad u_t=q_t^{tar},\quad
\hat x_{t+1}=F_{\mathrm{ctrl-struct}}(x_t,u_t;\vartheta)
\]

\(\vartheta\) = **effective closed-loop system coordinates**。禁止把 \(\vartheta\) 写成 mass / inertia / torque gain。主指标在 \((q,\dot q)\) 上，不以 \(\ddot q\) 或 \(\tau\) 为主。

## 门限（评分前冻结）

| 门 | 结果 | 细节 |
|----|------|------|
| G0 | **通过** | train/val/test \(E_1^{M0}\in\{0.965,0.983,0.947\}\ge 0.02\)；\(\Delta q\) 非退化 |
| G1 | **未过** | test 上 M1/M2/M3 均 \(E_1> E_1^{M0}\)；无模型达到 \(\le 0.75\,E_1^{M0}\) |
| G2 | **未评** | val 上无模型过 G1，\(k^\star=\emptyset\) |
| G3 | **未评** | 无冻结结构参数可迁移 |

Pattern：`low_order_structure_insufficient`  
`unlocks_x0d1=false`；`capacity_claim=false`

## 测试集主指标（臂 12 DoF）

| 模型 | \(E_1\) | \(E_1^q\) | \(E_1^{\dot q}\) | \(E_{\mathrm{roll}10}\) | \(E_{\mathrm{roll}50}\) |
|------|---------|-----------|------------------|-------------------------|-------------------------|
| M0 恒等 | **0.947** | **0.500** | 1.159 | **1.322** | **1.243** |
| M1 独立二阶离散 | 1.524 | 2.101 | 0.938 | 5.39 | \(1.02\times 10^5\) |
| M2 跨关节线性 | 1.522 | 2.101 | 0.934 | 18.6 | \(3.57\times 10^8\) |
| M3 \(\sin/\cos\) 构型基 | 1.520 | 2.101 | 0.927 | 18.7 | \(6.71\times 10^8\) |

G1 阈值：\(0.75\times 0.947=0.710\)。三者均约 1.52，远未过门。

## 读数（不改门、不改模型）

1. **速度通道有一点线性信息，但不够当结构优势。**  
   \(E_1^{\dot q}\)：M0 = 1.159，M1–M3 ≈ 0.93。跨关节（M2）和 \(\sin q,\cos q\)（M3）相对 M1 几乎没有再压缩。这与 X0C 的 `servo_signal_insufficient` 一致：\((e_q,e_v)\) 对下一步运动只有弱相关。

2. **一步误差被预注册的 Euler 位置更新拖垮。**  
   M1–M3 共享
   \[
   \hat q_{t+1}=q_t+\Delta t_{\mathrm{control}}\,\dot q_t.
   \]
   一次 native `take_action` 内部是 TOPP + 多次 PhysX substep，\(\Delta t_{\mathrm{control}}\) 是整段控制窗，不是单步积分器步长。因此 Euler 把 \(q\) 推得过远：\(E_1^q\approx 2.10\)，而恒等复制 \(q_t\) 只需 \(0.50\)。三者 \(E_1^q\) 完全相同，说明一步失败主要来自这条冻结的 \(q\) 运动学，而不是缺跨关节或构型基。

3. **滚动发散。** M1 的 H50 已 \(10^5\)；M2/M3 到 \(10^8\)。恒等 H50 仍约 1.24。低阶线性闭环在开环目标、闭环状态 rollout 下不稳定。

4. **这否掉的是预注册族，不是“一切结构”。**  
   已否：独立/耦合线性 \(\dot q\) + Euler \(q\) + 少量 \(\sin/\cos\)。  
   **未**在本格检验：对 \(q_{t+1}\) 也做线性拟合、更短内部时钟、非线性伺服、neural residual。按预注册，本族 STOP，不在见曲线后改 M1–M3 定义来补救。

Numpy 植物（真实 M1 离散系统）单测 **G0–G3 可通过**，说明门限在匹配植物上非空。正式结论以 **robotwin** 为准。

## 账本（冻结）

```text
RTWX-X0C = FAIL / servo_signal_insufficient
RTWX-X0D = FAIL / low_order_structure_insufficient
           STOP tested low-order closed-loop family  （M1–M3 不再改）
X0D1     = LOCKED
TASK-XL  = LOCKED
R10      = LOCKED
NEXT     = RTWX-X0E 新族 / direct native-window Δstate
```

已定位：失败主要是 **错误的外部时钟 / Euler 积分假设**，不是“物理完全无结构”。速度通道仍有弱信息（\(E_1^{\dot q}:1.159\to 0.927\sim 0.938\)）；跨关节线性与 \(\sin q,\cos q\) 不是瓶颈。  
**未证明**所有结构方法无效。下一格必须是新族，不是 X0D 补丁。

## 复现

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json PYTHONPATH=/root/APR-WM \
  /root/miniconda3/envs/Robotwin/bin/python -m aprwm_v0 rtwx-x0d \
  --output /root/APR-WM/runs/rtwx_x0d --backend robotwin --robotwin-repo /root/RoboTwin
```

单测：

```bash
PYTHONPATH=/root/APR-WM /root/miniconda3/envs/Robotwin/bin/python -m pytest tests/test_rtwx_x0d.py -q
```
