# RTWX-X0C 报告 — 原生 qpos 闭环伺服输入诊断

日期：2026-08-27  
状态：**已跑完**；**`rtwx_x0c_passed=false`**；pattern **`servo_signal_insufficient`**  
预注册：`REPORT/REG/RTWX/RTWX0C_PREREG.md`  
产物：`runs/rtwx_x0c/{run.json,header.json,run.header.txt,metrics.json,logs.npz,run.log}`  
不做：网络；\(\phi\) 拟合；容量；TASK-XL；RGB；R10；`set_qf`/`get_qf`/指令力矩/`qacc` 缓冲/滞后扫描

## 一句话

在冻结规模（train/val/test = 48/24/24，每段 120 次 native `take_action(..., action_type="qpos")`）上，**仪器侧 G0 通过**（force 驱动、目标与 \(K_p,K_d\) 可读、Pinocchio \(\tau_{\mathrm{ID}}\) 可算、有效数据接触率 0）。  
**G1 未过**：留出集 M2 \(E_{\ddot q}=0.869>0.75\)，且仅 **58%** 臂自由度 \(E_j<0.9\)（门限 80%）。  
因此 **不能** 声称「位置/速度误差已构成稳定有效控制器输入，足以支撑结构化自由空间动力学预测」。X0C1 **未解锁**。

本单元是 **oracle 仿真器状态 / native-qpos 闭环诊断**，**不是**官方 RoboTwin 观测基准，**不是**力传感器级辨识。\(\tau_{\mathrm{ID}}\) 是由观测运动 + 刚体模型得到的**所需广义力**。

## 门限（收集前冻结，未见曲线后改）

| 门 | 结果 | 细节 |
|----|------|------|
| G0 | **通过** | native qpos；drive mode=`force`；\(K_p=1000,K_d=200\) 全关节记录；接触率 0（有效段）；激励非退化 |
| G1 | **未过** | test M2 \(E_{\ddot q}=0.869\)；\(7/12\) 自由度 \(E_j<0.9\) |
| G2 | 未评（G1 失败） | 记录 Kp/Kd 的 M3 \(E_{\ddot q}\sim 1.6\times 10^5\)（远差于恒等）；不得把记录刚度当成已校准力矩增益 |
| G3 | 空过（未用 M4 主张 G2） | M4 test \(E_{\ddot q}\sim 224\)，不可辨识为物理参数 |

Pattern：`servo_signal_insufficient`  
`rtwx_x0c_passed=false`；`unlocks_x0c1=false`；`capacity_claim=false`

## 测试集主指标（臂 12 自由度，夹爪排除）

| 模型 | \(E_{\ddot q}\) | \(E_{\mathrm{roll}10}\) | \(E_\tau\) vs \(\tau_{\mathrm{ID}}\) |
|------|-----------------|-------------------------|--------------------------------------|
| M0 恒等 | 1.000 | 4.75 | — |
| M1 \(a\,e_q\) | 1.000 | — | — |
| M2 \(a e_q+b e_v\) | **0.869** | 4.56 | — |
| M3 记录 PD + \(M^{-1}(\hat\tau-h)\) | \(1.62\times 10^5\) | 发散 | 74.2（\(\rho_\tau=-0.24\)） |
| M4 对角非负拟合 | 224 | 发散 | 1.02（\(\rho_\tau\approx 0\)） |

M2 略优于恒等，但未达预注册 G1。M3 的记录 \(K_p,K_d\) **不能**当作与 \(\tau_{\mathrm{ID}}\) 同单位的力矩；RoboTwin 内部仍可对被动项做重力补偿（本单元未调用 `set_qf`）。

Numpy 后端（已知对角 PD+刚体植物）单测 **G1/G2 可通过**，说明门限在真植物上非空；正式科学结论以 **robotwin** 为准。

## 账本

```text
RTWX-X0/X0R/X0S/X0F = SAPIEN-in-RoboTwin 诊断（非 native qpos）
RTWX-X0C           = RAN / servo_signal_insufficient（首个 native-qpos 单元格）
X0C1               = LOCKED
TASK-XL / R10      = LOCKED
```

## 复现

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json PYTHONPATH=/root/APR-WM \
  /root/miniconda3/envs/Robotwin/bin/python -m aprwm_v0 rtwx-x0c \
  --output /root/APR-WM/runs/rtwx_x0c --backend robotwin --robotwin-repo /root/RoboTwin
```

单测（numpy，允许更小 \(n\)）：

```bash
PYTHONPATH=/root/APR-WM /root/miniconda3/envs/Robotwin/bin/python -m pytest tests/test_rtwx_x0c.py -q
```

环境短烟（**不是**正式 \(n\)，勿写入 `runs/rtwx_x0c`）：`--smoke --output runs/rtwx_x0c_smoke`。
