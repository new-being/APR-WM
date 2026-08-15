# Data collection next steps
_Exported on 2026/8/15 at GMT+8 22:08:49 from Cursor (3.16.17)_

---

**User**

@APR-WM进度.md  @APR-WM思路.md 参考这两篇文档，
进行下一步的数据采集

---

**Cursor**

先读两篇文档，确认当前进度和下一步该采什么数据。

进度文档很长，我接着看结尾和思路文档里“下一阶段”相关部分。

当前阻塞是 ManiSkill Drawer 官方资产（0/25）。下一步就是安全下载这批资产，为 Mode-A C0 采集解锁。我先检查下载脚本和本机状态。

本机没有 `.venv-maniskill` 和已下载资产。按文档先重建隔离环境，再尝试官方资产采集。

UCSD 经代理 TLS 失败、直连超时；HF 上没有 PartNet 资产。接下来查国内可达的官方同源镜像，并重建环境。

代理能连上，但 UCSD 的 TLS 仍失败。接着确认 C0 采集代码是否已实现，并重建环境做能做的准备。

按文档，下一步是 Drawer Mode-A C0 数据采集；当前卡在官方资产。先重建环境，并加上离线导入通道（本机 UCSD TLS 仍不通）。

环境安装还在进行；同时准备离线资产导入，并实现 C0 采集骨架。

数据采集需要在本机执行安装与下载命令，正在请求批准以继续。

---

**User**

可以切，而且我认为**现阶段切到 robosuite / MuJoCo 比继续卡在 ManiSkill Drawer 更合理**。但科学上要明确：这不是“换一个等价数据源继续 R1-MS”，而是把真实性阶段改造成一个新的 **R1-MJ/RS realism bridge**。原来的 `R1-MS0 = infrastructure block` 保留在实验记录里，不删除、不算失败。

当前 robosuite 最新 release 是 **v1.5.2**；它直接基于 MuJoCo，并提供 `Door`、`Wipe`、`ToolHang`、`NutAssembly` 等 manipulation 环境。`Door` 尤其适合接替 ManiSkill Drawer：单机械臂、单主要铰链，而且官方 `DoorObject` 本身就允许覆盖 hinge 的 `frictionloss` 和 `damping`。

我建议把新阶段定义为：

\[
\boxed{
R1\text{-MJ/RS}:
\text{Frozen R0.6 under MuJoCo dynamics, robot contact, and task diversity}
}
\]

而不是立即训练任何 policy 或视觉网络。

---

# 一、为什么现在适合切 MuJoCo / robosuite

ManiSkill 当前阻塞在外部 PartNet 资产，而 APR-WM 这一步真正需要验证的不是视觉泛化，而是：

\[
\text{generalized dynamics}
+
\text{contact}
+
\text{geometry}
+
\text{model revision}
\]

robosuite 正好允许把这些因素逐个打开。

更重要的是，MuJoCo 的动力学接口和你已经验证成功的 R0.1 force-space formulation 非常匹配。MuJoCo 明确给出

\[
M(q)\dot v+c(q,v)=
\tau+J^\top f
\]

其中：

- `qpos`：\(q\)
- `qvel`：\(v\)
- `qfrc_bias`：\(c(q,v)\)
- inertia matrix：\(M(q)\)
- `qfrc_actuator`
- `qfrc_applied`
- `qfrc_passive`
- constraint/contact force

都可以直接获得。

所以原来的 generalized-force residual 可以自然改成：

\[
\boxed{
r_{\tau,j}
=
\left[
M(q)\ddot q+
c(q,\dot q)
-
\tau_{\mathrm{known}}
-
\tau_{\mathrm{constraint}}
-
\tau_{\mathrm{nominal-passive}}
\right]_j
}
\]

这里 \(j\) 是 door hinge DOF。

一个非常重要的规则是：

> **不能把 truth simulator 的完整 `qfrc_passive` 喂给 APR-WM。**

否则你人为注入的 damping / drag 已经被 MuJoCo 算出来了，residual 会被直接消掉，形成 label leakage。

因此以后应该区分：

\[
\boxed{
\text{truth simulator}
\neq
\text{learner nominal physics}
}
\]

---

# 二、建议把 R1 重构成四阶段

我建议严格按：

\[
\boxed{
R1\text{-MJ0}
\rightarrow
R1\text{-RS0}
\rightarrow
R1\text{-RS1}
\rightarrow
R1\text{-RS2}
}
\]

进行。

其中：

| 阶段 | 环境 | 主要问题 | Robot contact |
|---|---|---|---|
| R1-MJ0 | 原生 MuJoCo 单 hinge | force interface 是否闭合 | 无 |
| R1-RS0 | robosuite Door | robosuite asset 是否闭合 | 无 |
| R1-RS1 | Door + controlled misspecification | R0.6 revision 能否迁移 | 无 |
| R1-RS2 | Door + Panda | robot contact 下是否仍成立 | 有 |
| 后续 | Wipe / ToolHang | persistent contact / multi-object | 更复杂 |

**不要直接从 Panda 拉门开始。**

否则如果失败，你无法判断是：

- MuJoCo API 适配错；
- door inertia 错；
- contact force accounting 错；
- controller 错；
- 还是 R0.6 真正失效。

---

# 三、R1-MJ0：MuJoCo force-space closure

这是新的最低层 preflight。

直接构造一个极小 MJCF：

```text
world
 └── door/body
      └── hinge joint
```

只有：

\[
q,\dot q,\ddot q,\tau
\]

不要机器人，不要 contact，不要 latch。

使用三类 trajectory：

1. sinusoidal torque
2. chirp / frequency sweep
3. random piecewise-constant torque

每条 10 s。

开发 seeds：

\[
8201,\;8211,\;8221
\]

每 seed × 3 trajectories：

\[
3\times3=9
\]

个基础 cell。

这一阶段只验证：

\[
M(q)\ddot q+c(q,\dot q)-\tau
\approx0
\]

以及：

- snapshot restore
- deterministic replay
- DOF indexing
- H2/H4/H8/H32 plumbing
- generalized-force extraction
- parameter tangent
- passivity calculation

建议 gate：

\[
\frac{\operatorname{RMSE}(r_\tau)}
{\operatorname{RMS}(\tau)+\epsilon}
<10^{-4}
\]

并要求所有 9 cell individually close。

这是 infrastructure gate，不算正式科学结果。

---

# 四、一个必须立即修改的 MuJoCo API 问题

如果直接使用当前 MuJoCo，尤其需要注意这一点。

截至 2026 年 7 月 27 日，stable MuJoCo 已到 **3.11.0**。MuJoCo 3.11 删除了旧的 `mjData.qM` inertia storage，改成 CSR `mjData.M`；当前 API 应通过新的 `mj_fullM(model, data, dst)` 或 `mj_mulM` 使用 mass matrix。

所以如果 R0.6 adapter 里面还有类似：

```python
data.qM
```

或者旧版：

```python
mujoco.mj_fullM(model, M, data.qM)
```

不要直接搬。

新的 adapter 应抽象成类似：

```text
get_mass_matrix(model, data)
get_bias_force(data)
get_constraint_force(data)
get_actuator_force(data)
```

以后 SAPIEN / MuJoCo 后端都通过统一接口。

对于正式实验，我建议：

```text
robosuite == 1.5.2
MuJoCo == 一个通过完整 smoke 的固定版本
```

可以先测试 3.11.0，但**版本一旦通过 RS0 就冻结**，正式 held-out experiment 不升级 physics engine。

---

# 五、R1-RS0：robosuite Door Mode-A C0

这是旧 ManiSkill Drawer 9-cell gate 的真正替代品。

robosuite 官方 `Door` 是单臂 door-opening environment，而且 `DoorObject` 明确暴露 hinge friction 和 damping override。

但这一阶段**不让 Panda 操作门**。

加载：

```text
Door
Panda present
robot fixed / inactive
use_latch = False
```

直接向 door hinge 施加 generalized torque。

这样验证：

\[
\boxed{
\text{R0.6 force-space formulation}
\rightarrow
\text{robosuite Door asset}
}
\]

---

## 三个 C0 physics profiles

不要只测一个 Door。

建议建立：

| Profile | friction | damping | learner |
|---|---:|---:|---|
| P0 | nominal | nominal | known |
| P1 | nominal × 0.5 | nominal × 0.5 | known |
| P2 | nominal × 2 | nominal × 2 | known |

注意这三个仍然全部是 **C0 adequate physics**：

环境参数变了，但 learner 使用对应正确参数。

目的不是制造 misspecification，而是验证：

\[
\text{closure is robust across parameter scale}
\]

固定：

\[
\boxed{
3\ profiles
\times
3\ seeds
=
9\ cells
}
\]

seeds：

\[
8301,\;8311,\;8321
\]

继续沿用原来的原则：

\[
\boxed{
\text{all 9 cells individually close before aggregation}
}
\]

建议 gate：

\[
NRMSE_\tau<10^{-3}
\]

\[
C0\ false\ revision\le1\%
\]

\[
H32\ validity\ coverage\ge95\%
\]

以及：

\[
\boxed{
\text{no C0 cell may be rescued by aggregation}
}
\]

如果这个阶段失败，仍然只能排查：

\[
\text{DOF mapping}
\rightarrow
M(q)
\rightarrow
gravity/bias
\rightarrow
passive force
\rightarrow
constraint accounting
\]

不能 tuning R0.6。

---

# 六、R1-RS1：真正的 model-revision scientific experiment

RS0 通过后，才开始测试 R0.6。

这一阶段仍然**不使用 robot contact**。

建立五个 regime。

### C0：adequate

完全 nominal。

用于测 false revision。

---

### C1-L：已知结构，中等强度

采用 frozen R0.6 已存在的 dissipative operator，例如：

\[
r_\tau=-0.12|v|v
\]

这是你在 R0.x 已经反复使用过的 operator。

---

### C1-H：同结构 OOD coefficient

例如：

\[
r_\tau=-0.24|v|v
\]

检验：

\[
\text{operator generalization}
\neq
\text{memorized coefficient}
\]

---

### CNEG：anti-damping negative control

\[
r_\tau=+0.12|v|v
\]

这是非常重要的 control。

R0.5P 已说明 active `abs_v_v` 会产生能量注入和长期 instability。

所以 frozen R0.6 应当：

\[
\boxed{
\text{detect it statistically}
\quad\text{but reject it physically}
}
\]

即：

\[
\text{high evidence}
\not\Rightarrow
\text{accept}
\]

这是检验你当前论文最高层原则最干净的实验之一。

---

### C2：outside-library

建议直接启用：

```text
use_latch=True
```

或者引入明显的 discontinuous/contact structural effect。

目的不是要求 APR-WM recovery，而是要求：

\[
\boxed{
\text{unknown rejection}
}
\]

避免 forced wrong explanation。

---

# 七、RS1 正式矩阵

使用新的 held-out seeds：

\[
9101,\;9111,\;9121,\;9131,\;9141
\]

每个：

- 5 regimes
- 8 probe trajectories

因此：

\[
\boxed{
5\ seeds
\times
5\ regimes
\times
8
=
200\ trajectories
}
\]

每个 trajectory 建议 10 s。

其中 probe bank 固定为：

```text
2 × low-frequency sine
2 × high-frequency sine
2 × chirp
2 × piecewise random
```

probe amplitude 在正式运行前冻结。

不能根据某个 C1 condition 单独调 amplitude。

---

# 八、RS1 的预注册指标

这里建议沿用 R0.6 的核心指标，而不是创造一套新指标。

### C0

要求：

\[
false\ revision\le1\%
\]

---

### C1

要求：

\[
exact\ operator\ recovery\ge85\%
\]

coefficient relative error：

\[
\frac{|\hat\alpha-\alpha|}
{|\alpha|}
\le10\%
\]

至少 85% accepted cases 满足。

---

### Physical safety

CNEG：

\[
\boxed{
physical\ acceptance=0
}
\]

至少不允许有长期 unstable accepted revision。

总 accepted stability：

\[
\boxed{\ge99\%}
\]

---

### C2

要求：

\[
unknown\ rejection\ge90\%
\]

\[
forced\ wrong\ revision\le5\%
\]

---

### Rollout

仍然：

```text
H2 / H4 / H8 = acceptance-visible
H32 = blind evaluation only
```

主要报告：

\[
RMSE_{H32}^{physics}
\]

\[
RMSE_{H32}^{no-revision}
\]

\[
RMSE_{H32}^{R0.6}
\]

以及 paired gain：

\[
G_i=
RMSE_i^{no-revision}
-
RMSE_i^{R0.6}
\]

我会把 scientific success 定义为：

\[
\boxed{
\operatorname{mean}(G)>0
}
\]

且 paired 95% CI lower bound \(>0\)。

这比人为规定“必须提升 10%”更合理。

---

# 九、R1-RS2：真正加入 Panda contact

只有 RS1 通过才进入这里。

使用：

```text
robosuite Door
robot = Panda
controller = OSC
camera = off
policy = none
```

robosuite 本身提供 operational-space controller；环境默认接口也支持 20 Hz 控制频率。内部物理仿真实际会在每个控制动作之间执行多次 MuJoCo simulation steps，因此你的 dynamics logger 应该挂在**simulation timestep**，而不能只记录 `env.step()` 返回的 control-rate observation。

设计三个固定 scripted manipulation：

\[
A_1=\text{slow pull}
\]

\[
A_2=\text{fast pull}
\]

\[
A_3=\text{pull-release}
\]

不要 RL。

不要 imitation policy。

不要 diffusion policy。

这些动作只是 deterministic excitation。

---

# 十、contact 下 residual 要重新定义

MuJoCo 的完整 equation 是：

\[
M\ddot q+c
=
\tau
+
J^\top f
\]

所以如果机器人正在拉门，而你仍然使用：

\[
r_\tau=M\ddot q+c-\tau
\]

那么：

\[
J^\top f
\]

会全部被错误解释成 structural residual。

MuJoCo 本身明确将 contact / constraints 作为 \(J^\top f\) 放入动力学方程。

因此 RS2 第一阶段必须使用：

\[
\boxed{
r_{\tau,\mathrm{intrinsic}}
=
M\ddot q+c
-
\tau_{\mathrm{known}}
-
J^\top f_{\mathrm{robot-contact}}
-
\tau_{\mathrm{nominal-passive}}
}
\]

也就是说：

> **第一版 RS2 使用 oracle contact force。**

这不是最终目标，但非常重要。

因为此时科学问题只是：

\[
\boxed{
\text{Does revision survive robot-contact excitation?}
}
\]

还不是：

\[
\boxed{
\text{Can APR-WM infer contact dynamics itself?}
}
\]

后者应当是后续 R2。

---

# 十一、RS2 正式实验矩阵

我建议去掉 CNEG，把它留在 RS1 做安全 falsification。

RS2 使用四个 regime：

```text
C0
C1-L
C1-H
C2-latch
```

然后：

\[
5\ seeds
\times
4\ regimes
\times
3\ action\ scripts
\times
3\ repetitions
\]

得到：

\[
\boxed{180\ episodes}
\]

每 episode 约 15 s。

主要比较：

```text
Physics
No-revision
Frozen R0.6
Oracle correct revision
```

不训练任何新网络。

---

# 十二、RS2 gate

建议：

\[
C0\ false\ revision\le2\%
\]

因为 contact 环境允许比 direct-drive 略宽。

仍要求：

\[
accepted\ stability\ge99\%
\]

C1：

\[
operator\ recovery\ge80\%
\]

以及最重要的 H32：

\[
\boxed{
95\%\,CI
\left(
RMSE^{no-revision}_{H32}
-
RMSE^{R0.6}_{H32}
\right)
>0
}
\]

C2：

\[
unknown\ rejection\ge90\%
\]

\[
wrong\ explicit\ revision\le5\%
\]

---

# 十三、数据具体保存成什么

建议继续 HDF5。

但把数据严格拆成：

```text
/metadata
/raw_truth
/learner_visible
/decision
/evaluation
```

这是这次非常重要的设计。

### metadata

保存：

```text
robosuite_version
mujoco_version
APR-WM git commit
XML SHA256
seed
task
robot
controller
model_timestep
control_timestep
physics_profile
regime
```

### raw_truth

仅用于离线审计：

```text
qpos
qvel
qacc
qfrc_bias
qfrc_passive_truth
qfrc_actuator
qfrc_applied_truth
qfrc_constraint
contact info
true hidden operator
true coefficient
```

### learner_visible

APR-WM 真正能看：

```text
q
qdot
qddot
known control
known nominal physics
oracle contact force        # RS2 only
nominal metadata
```

**不能出现：**

```text
true hidden damping
true hidden drag coefficient
truth qfrc_passive decomposition
true model class
```

### decision

保存：

```text
trigger score
r_parallel
r_perp
candidate ranking
candidate coefficient
passivity score
utility score
selected candidate
accepted / rejected
fallback
```

### evaluation

保存：

```text
H2 RMSE
H4 RMSE
H8 RMSE
H32 RMSE
stability
energy change
revision gain
oracle utility
```

这样可以完全避免未来被质疑：

> “是不是直接从 simulator internal state 读到了 ground-truth dynamics？”

---

# 十四、到底需要多少数据

其实很少。

这不是 diffusion policy training dataset，而是 system-ID / revision evaluation dataset。

假设一个 simulation-step record 保存约：

\[
100\sim150
\]

个 `float64`。

那么大概：

\[
0.8\sim1.2\text{ KB / simulation step}
\]

RS1：

\[
200\times10\text{ s}
\]

RS2：

\[
180\times15\text{ s}
\]

如果实际 runtime：

\[
dt=0.002s
\]

则大约：

\[
N_{RS1}=1.0M
\]

\[
N_{RS2}=1.35M
\]

加上 RS0，整个 Door 阶段大约：

\[
\boxed{2.5\sim3M\ simulation\ steps}
\]

所以完整 raw dynamics：

\[
\boxed{\approx3\sim5\text{ GB}}
\]

就已经很宽裕。

如果只保存 target hinge + 必要 robot/contact quantities：

\[
\boxed{<1\text{ GB}}
\]

完全可能。

因此没必要找几十 GB / 几百 GB 的 robosuite demonstration dataset。

**当前阶段应该自己在线生成数据。**

---

# 十五、Door 通过以后再扩展什么

我建议顺序不是 PickPlace，而是：

\[
\boxed{
Door
\rightarrow
Wipe
\rightarrow
ToolHang/NutAssembly
}
\]

因为科学复杂度是逐步增加的。

`Door` 是：

\[
\text{1 articulated DOF + contact}
\]

`Wipe` 官方也是 robosuite 的单臂 manipulation task，它会引入持续表面接触和摩擦。

然后 `ToolHang` / `NutAssembly` 增加：

\[
\text{multi-object}
+
\text{geometry constraints}
+
\text{contact topology changes}
\]

robosuite 当前文档中这些环境均存在。

所以真实性链会变成：

\[
\boxed{
\begin{aligned}
R0.6 &: \text{SAPIEN single articulated dynamics}\\
R1\text{-MJ0} &: \text{MuJoCo force closure}\\
R1\text{-RS0} &: \text{robosuite Door closure}\\
R1\text{-RS1} &: \text{Door structural revision}\\
R1\text{-RS2} &: \text{robot-contact revision}\\
R1\text{-RS3} &: \text{persistent contact / multi-object}\\
R2 &: \text{contact itself becomes learned interaction structure}
\end{aligned}
}
\]

---

# 十六、我建议现在实际执行到哪里

现在不要一次把 RS2 全做完。

新的严格解锁顺序应该改成：

\[
\boxed{
\text{MJ0}
\rightarrow
\text{RS0 C0}
\rightarrow
\text{RS1}
\rightarrow
\text{RS2}
}
\]

**第一轮实际工作量只做两件事：**

1. 建 MuJoCo backend adapter，完成 **R1-MJ0 9-cell closure**；
2. 加载 robosuite `Door`，完成新的 **R1-RS0 3 physics profiles × 3 seeds = 9-cell Mode-A C0**。

只有这两个都通过，才跑 200-trajectory RS1 scientific matrix。

这样既绕过 UCSD 资产问题，又没有牺牲你前面一直保持得很好的 gate discipline。

我会把当前项目状态更新理解为：

```text
R1-MS0:
    frozen = true
    status = infrastructure_block
    scientific_no_go = false

R1-MJ/RS:
    selected_as_next_realism_bridge = true

R1-MJ0:
    unlocked = true

R1-RS0:
    locked_until_MJ0_passes = true

R1-RS1:
    locked = true

R1-RS2:
    locked = true

Wipe / ToolHang:
    locked = true
```

从科学性、工程可控性和你当前 APR-WM 的发展脉络来看，我认为这比继续解决 ManiSkill asset 下载问题更值得优先做。尤其是 **RS1 的 dissipative truth / anti-damping negative control / latch unknown 三者同时存在时，可以非常直接地检验 R0.6 最核心的命题：**

\[
\boxed{
\text{statistically explanatory}
\not\Rightarrow
\text{physically admissible}
}
\]

以及真正想证明的：

\[
\boxed{
\text{physically admissible}
+
\text{predictively useful}
\Rightarrow
\text{safe model self-revision}
}
\]

这会比单纯换一个 Drawer asset benchmark 更有科学信息量。

---

**Cursor**

按你的方案切到 R1-MJ/RS：保留 R1-MS0 为 infrastructure block，先做 MJ0 九格闭合，再做 RS0。开始核对仓库状态并落地实现。

仓库里已有 R1-MJ/RS 骨架。我先核对实现完整度、运行结果和下一步缺口。

MJ0 已通过；RS0 卡在 MuJoCo 3.11 移除 `qM` 后 robosuite OSC 仍访问旧 API。正在修复兼容层并重跑 RS0 采集。

Door 环境已能创建。接着验证力残差路径，并跑完整 RS0 九格采集。

根因是 robosuite 的 `MjData` 在类创建时固化属性；需在兼容层同时打补丁到 wrapper，并修正 seed / 底层 model 解包。

环境已通，但 hinge 残差 NRMSE≈0.13，未达闭合门限。正在诊断力平衡。

残差主因是流体黏性与未固定的机器人。将关闭 viscosity、运动学冻结 Panda，并设非零 nominal friction/damping。

MuJoCo 的 `frictionloss` 进的是 `qfrc_constraint` 而非 `qfrc_passive`。因此 learner 只建模 damping，摩擦由 constraint 项吸收。

RS0 九格全部通过。正在恢复 R1-MS0 记录并更新进度文档。

已按你的方案切到 **R1-MJ/RS**，第一轮两级 infrastructure gate 都过了。

### 状态
| 阶段 | 结果 |
|------|------|
| **R1-MS0** | 保留为 `infrastructure_block`（已恢复 `REPORT/R1_MS_PREFLIGHT_REPORT.md`），非 scientific no-go |
| **R1-MJ0** | **通过** — 9/9，最大 NRMSE `8.21e-11` |
| **R1-RS0** | **通过** — 9/9，最大 NRMSE `2.79e-4` |
| **R1-RS1** | 已具备解锁资格，**尚未跑** |
| **R1-RS2 / Wipe** | 仍锁定 |

### 关键实现点
- 冻结：`robosuite==1.5.2`，`mujoco==3.11.0`
- 修了 MuJoCo 3.11 删除 `qM` 后 robosuite OSC 的兼容层
- Door Mode-A：Panda **运动学冻结**，直接激励 `Door_hinge`
- **不读** truth `qfrc_passive`；MuJoCo 的 `frictionloss` 走 `qfrc_constraint`，learner 只显式建模 damping
- 数据按 `raw_truth / learner_visible / decision / evaluation` 写入 HDF5

### 产物
- `runs/r1_mj0/closure/`
- `runs/r1_rs0/c0/`
- [R1_MJRS_MJ0_RS0_REPORT.md](REPORT/R1_MJRS_MJ0_RS0_REPORT.md)

下一步按协议应是 **R1-RS1**（C0 / C1-L / C1-H / CNEG / C2-latch，200 trajectories），仍不碰 robot contact。需要的话我可以直接开做。

---

**User**

可以，**现在应直接做 R1-RS1**。根据已上传报告，两个前置 gate 都满足解锁条件，而且 RS0 的 accounting 已经处理了最容易造成伪结果的 `qfrc_passive / qfrc_constraint` 问题。

我建议在运行前把下面版本作为 **R1-RS1 正式 preregistration** 冻结，之后不根据结果修改。

## 1. 科学问题

R1-RS1 不再是 infrastructure test，而是第一个新的 scientific transfer：

\[
\boxed{
\text{Does frozen R0.6 retain model-inadequacy detection,
structure selection, physical rejection and useful revision
under robosuite/MuJoCo dynamics?}
}
\]

也就是同时检验四件事：

\[
\text{detect}
\rightarrow
\text{identify}
\rightarrow
\text{physically filter}
\rightarrow
\text{improve rollout}
\]

此阶段仍然严格排除 robot contact，避免把 contact accounting 混进 revision science。

---

# 2. 全部冻结项

固定：

| 项目 | RS1 设置 |
|---|---|
| robosuite | `1.5.2` |
| MuJoCo | `3.11.0` |
| Environment | `Door` |
| Robot | Panda present, kinematically frozen |
| Door drive | direct generalized torque on `Door_hinge` |
| Base profile | **RS0 P0** |
| frictionloss | `0.10` |
| damping | `0.10` |
| simulation dt | `0.002 s` |
| episode length | `10 s` |
| learner constraint accounting | `include_constraint=True` |
| truth `qfrc_passive` | raw_truth only |
| R0.6 parameters | frozen |
| operator library | frozen |
| proposal ranking | frozen |
| tangent thresholds | frozen |
| acceptance thresholds | frozen |
| passivity/admissibility | frozen |
| utility gate | frozen |
| H2/H4/H8 | acceptance-visible |
| H32 | **blind evaluation only** |

尤其禁止：

\[
\boxed{
\text{adding MuJoCo-specific operators after seeing RS1 results}
}
\]

否则 RS1 就不再是 frozen-transfer experiment。

---

# 3. 五个 regime

建议就按之前约定的五组，不再扩展。

| Regime | Truth | 预期 |
|---|---|---|
| C0 | adequate P0 | 不 revision |
| C1-L | \(r_\tau=-0.12|v|v\) | recovery |
| C1-H | \(r_\tau=-0.24|v|v\) | coefficient OOD recovery |
| CNEG | \(r_\tau=+0.12|v|v\) | 有证据，但必须 physical reject |
| C2-latch | latch structural dynamics | unknown / reject wrong explanation |

这里最重要的是 CNEG。

它不是普通 negative sample，而是直接 falsify：

\[
\text{statistical evidence}
\Rightarrow
\text{revision}
\]

这个错误命题。

理想表现应是：

\[
\boxed{
\text{trigger}=1,\quad
\text{candidate evidence high},\quad
\text{physical admissibility}=0,\quad
\text{accept}=0
}
\]

换言之，R0.6 不能靠“根本没有发现 CNEG”来通过。

---

# 4. Hidden-force 实现

C1/CNEG 的 hidden operator 可以注入 truth simulator：

\[
\tau_{\mathrm{hidden}}
=
\alpha |v|v
\]

其中

\[
\alpha\in\{-0.12,-0.24,+0.12\}.
\]

但必须保持数据边界：

\[
\tau_{\mathrm{hidden}}
\in raw\_truth
\]

且：

\[
\tau_{\mathrm{hidden}}
\notin learner\_visible.
\]

learner 仍只知道：

\[
(q,\dot q,\ddot q,\tau_{\mathrm{command}},
\text{nominal physics})
\]

然后从 generalized-force residual 自己发现 mismatch。

否则会发生 intervention leakage。

---

# 5. C2-latch 要额外小心

这里我建议比之前定义得更严格。

`use_latch=True` 产生的是 **constraint/contact-mediated structural effect**，它非常适合作为 outside-library case，但不能把：

```text
unsupported
```

自动算成：

```text
correct unknown rejection
```

否则会出现：

\[
\boxed{
\text{validity rejection}
\neq
\text{model-class rejection}
}
\]

所以 C2 必须同时记录：

\[
p(\text{modeled})
\]

\[
p(\text{boundary})
\]

\[
p(\text{unsupported})
\]

\[
p(\text{trigger})
\]

以及：

\[
p(\text{unknown}\mid
\text{triggered and evaluable})
\]

这样能够区分三种失败：

\[
\text{did not observe informative regime}
\]

\[
\text{observed but did not trigger}
\]

\[
\text{triggered but explained with wrong model}
\]

这和 V5 中你已经得到的经验一致：**trigger miss 和 rejection failure 必须分开。**

---

# 6. Probe bank

正式使用固定 8 个 probe。

不需要 active probe selection，否则又同时测试 V5 的 experiment-design mechanism。

建议：

| Probe | 类型 |
|---|---|
| P1 | low-frequency sine #1 |
| P2 | low-frequency sine #2 |
| P3 | high-frequency sine #1 |
| P4 | high-frequency sine #2 |
| P5 | upward chirp |
| P6 | downward chirp |
| P7 | slow piecewise-random |
| P8 | fast piecewise-random |

**幅值直接沿用已经通过 RS0 closure 的安全 excitation scale**，不要为了 RS1 单独重新优化 amplitude。

频率、phase、piecewise dwell time 在正式运行前写到 config。

random probe 的随机序列必须由 episode seed 唯一决定。

---

# 7. 正式矩阵

使用 previously reserved held-out seeds：

\[
\boxed{
9101,\;9111,\;9121,\;9131,\;9141
}
\]

于是：

\[
5\ seeds
\times
5\ regimes
\times
8\ probes
=
\boxed{200\ trajectories}
\]

每条 10 s：

\[
200\times10s
\]

对应：

\[
5000
\]

simulation steps / trajectory，因此约：

\[
\boxed{10^6\ simulation\ steps}
\]

规模很小。

---

# 8. 允许一次 plumbing smoke，但不能看科学指标调参数

正式 200 条以前，我允许一个很窄的非科学 smoke：

\[
1\ dev\ seed\times5\ regimes\times1\ probe
\]

即 5 trajectories。

例如使用：

```text
seed = 8901
```

只检查：

- 是否 crash；
- C2 latch 是否真的激活；
- HDF5 schema；
- hidden force 是否没有进入 `learner_visible`；
- H32 是否确实 blind；
- deterministic replay；
- NaN / indexing。

**不能根据这 5 条的 AUROC、recovery、RMSE 去改 threshold。**

否则它就变成 development data。

---

# 9. RS1 primary gates

我建议正式冻结为以下标准。

### C0 — specificity

\[
\boxed{
false\ revision\le1\%
}
\]

同时要求 RS0-style closure 没有明显退化。

---

### C1-L / C1-H — discovery

合并主要指标：

\[
\boxed{
exact\ operator\ recovery\ge85\%
}
\]

同时分别报告 C1-L 和 C1-H，不能只报 pooled。

coefficient accuracy：

\[
\boxed{
\frac{|\hat\alpha-\alpha|}{|\alpha|}
\le10\%
}
\]

要求至少 85% 的 correct accepted revisions 达到该范围。

这样 C1-H 真正检验的是：

\[
\boxed{
\text{operator identity generalization}
+
\text{coefficient estimation}
}
\]

而不是记住 \(-0.12\)。

---

# 10. CNEG 应成为硬 safety gate

这里我甚至建议比 aggregate stability 更严格：

\[
\boxed{
CNEG\ accepted\ physical\ revision=0
}
\]

因为 passivity 在 R0.6 已经是硬结构约束。

如果：

```text
1 / 40
```

CNEG trajectory 最终接受了正 `abs_v_v`，

我会判：

\[
\boxed{\text{RS1 safety regression}}
\]

即使 aggregate RMSE 很漂亮也不能通过。

但同时必须报告：

\[
trigger\ rate
\]

和 candidate proposal / selection rate。

因为：

```text
CNEG trigger = 0
accept = 0
```

并不能证明 physical admissibility gate 有效。

真正强证据是：

```text
trigger high
proposal/select active operator
passivity rejects it
```

---

# 11. C2 要防止“什么都不做也能过”

之前简单写：

\[
unknown\ rejection\ge90\%
\]

存在一个漏洞：

> 如果模型完全不 trigger，也可能看起来没有 forced wrong revision。

因此这次建议至少同时报告两个不同 denominator。

条件 rejection：

\[
R_{\mathrm{cond}}
=
P(
\text{unknown}
\mid
\text{triggered,evaluable}
)
\]

要求：

\[
\boxed{
R_{\mathrm{cond}}\ge90\%
}
\]

forced wrong：

\[
\boxed{
P(\text{wrong explicit revision}\mid
\text{triggered,evaluable})
\le5\%
}
\]

此外必须单独报告：

\[
\boxed{
C2\ trigger\ coverage
}
\]

不建议现在凭空给 trigger coverage 再设一个新的高阈值；先把它作为 **non-vacuity diagnostic**。

如果最终：

```text
unknown rejection = 100%
trigger coverage = 8%
```

则不能声称 C2 discovery/rejection 已经成功，只能说：

> once triggered, rejection is reliable, but detection coverage is insufficient.

这个解释更干净。

---

# 12. 长期 stability

继续沿用 **R0.6 已冻结的 stability predicate**，不要在 MuJoCo 阶段重新定义“stable”。

要求：

\[
\boxed{
accepted\ revision\ stability\ge99\%
}
\]

但 CNEG 的 0-acceptance hard gate 优先级更高。

即：

\[
\text{CNEG failure}
\Rightarrow
\text{RS1 fail}
\]

不能被其它 case 的 100% stability 平均掉。

---

# 13. H32 是 RS1 最关键 scientific endpoint

H2/H4/H8 仍然允许参与 acceptance。

但是：

\[
\boxed{H32\text{ cannot influence any decision}}
\]

正式 pipeline 必须是：

```text
observation
    ↓
trigger
    ↓
proposal
    ↓
selection
    ↓
H2/H4/H8 validation
    ↓
physical admissibility
    ↓
utility gate
    ↓
ACCEPT / REJECT
    ↓
freeze decision
    ↓
H32 blind rollout
```

绝对不能：

```text
H32 → acceptance
```

---

# 14. H32 utility 的统计单位必须是 seed，而不是 trajectory

这是这次实验一个值得提前修正的统计问题。

你有 80 条 C1 trajectories：

\[
5\ seeds\times2\ C1\times8\ probes.
\]

但不能把这 80 条当成 80 个独立样本去构造 CI，因为同一个 seed 下共享 simulator realization / configuration。

先计算：

\[
G_{s,r,p}
=
RMSE^{no-revision}_{H32}
-
RMSE^{R0.6}_{H32}
\]

然后每个 seed 内聚合：

\[
\bar G_s
=
\frac1{16}
\sum_{r,p}G_{s,r,p}
\]

最终统计单位为：

\[
\boxed{n=5\ seeds}
\]

主要 utility gate：

\[
\boxed{
\operatorname{mean}_s(\bar G_s)>0
}
\]

并报告 seed-level paired 95% CI。

若沿用你此前 5-seed 的统计习惯，可以用 \(t_4\) CI。

不要使用 trajectory-level 80-sample CI，否则显著性会被伪放大。

---

# 15. 建议同时保存 decomposition funnel

RS1 最有价值的不只是最终 recovery。

建议每个 regime 输出：

```text
total
↓
valid / evaluable
↓
triggered
↓
truth in top-3
↓
correctly selected
↓
statistically accepted
↓
physically admissible
↓
utility accepted
↓
H32 useful
```

这样如果失败，可以立即知道瓶颈从 V4/V5/V6/R0.6 的哪一层重新出现。

例如：

\[
\text{proposal recall}=98\%
\]

但：

\[
\text{selection}=70\%
\]

就是 V4-type failure。

或者：

\[
selection=95\%,\quad acceptance=30\%
\]

就是 V5/V6 validation-power failure。

或者：

\[
short\ acceptance=90\%,\quad H32<0
\]

则重新出现 R0.3/R0.4 long-horizon utility failure。

这会让 RS1 的科学信息量远高于只给一个最终 RMSE。

---

# 16. 最终 RS1 判定逻辑

我建议预注册成：

\[
\boxed{
RS1\_GO =
C0_{\mathrm{specific}}
\land
C1_{\mathrm{recover}}
\land
CNEG_{\mathrm{safe}}
\land
C2_{\mathrm{reject}}
\land
H32_{\mathrm{stable}}
\land
H32_{\mathrm{useful}}
}
\]

其中核心硬 gate 可以写成：

| Gate | 标准 |
|---|---:|
| C0 false revision | \(\le1\%\) |
| C1 exact recovery | \(\ge85\%\) |
| C1 coefficient ≤10% error | \(\ge85\%\) of correct accepted |
| CNEG accepted active revision | **0** |
| C2 conditional unknown rejection | \(\ge90\%\) |
| C2 conditional forced-wrong | \(\le5\%\) |
| accepted H32 stability | \(\ge99\%\) |
| C1 seed-level H32 paired gain | mean \(>0\)，并报告 95% CI |

这里我**不建议强制把 95% CI lower bound \(>0\) 设成唯一 GO gate**。只有 5 个 held-out seeds 时这个条件容易把“有效但统计 power 不足”和“没有 utility”混在一起。

更严谨的是：

\[
\boxed{
\text{mean paired gain}>0
}
\]

作为 GO 条件，同时把：

\[
CI_{95\%}
\]

作为 evidence strength 报告。

如果 lower bound 也 \(>0\)，那就是更强的 confirmatory success。

---

# 17. RS1 通过后才能解锁什么

若 RS1_GO：

\[
\boxed{
R1\text{-RS2 unlocked}
}
\]

然后才允许：

\[
\text{Panda motion}
+
\text{robot-door contact}
+
\text{oracle }J^\top f
\]

如果 RS1 fail：

**不要直接 tuning R0.6。**

先按 funnel 分类：

\[
\boxed{
\begin{aligned}
\text{closure failure}&\rightarrow\text{interface}\\
\text{trigger failure}&\rightarrow\text{inadequacy detection}\\
\text{proposal failure}&\rightarrow\text{library/representation}\\
\text{selection failure}&\rightarrow\text{discrimination}\\
\text{acceptance failure}&\rightarrow\text{validation}\\
\text{CNEG failure}&\rightarrow\text{physical admissibility}\\
\text{H32 failure}&\rightarrow\text{long-horizon utility}
\end{aligned}
}
\]

这会保持整个 APR-WM 从 V0 到 R1 的因果诊断逻辑一致。

**所以现在可以直接开 RS1。** 我建议把上述 protocol 写入 `REPORT/R1_RS1_PREREG.md` 后锁住 config，再跑 5-trajectory plumbing smoke，随后一次性执行 200 trajectories。此时 RS1 才真正开始检验 R0.6 从 SAPIEN 到 MuJoCo/robosuite 的科学迁移，而不是继续做 infrastructure closure。

---

**Cursor**

按你的 preregistration 冻结 RS1：先写协议与配置，再做 5 条 plumbing smoke，最后一次性跑 200 条。

开始落地：先写 prereg 与核心实现，Door 采样范围会按 `[0,0.4]` 关节限位适配。

实现会拆成几个模块来写，避免单文件过大。

定位到关键泄漏：`qfrc_applied` 含 hidden，却被当成 learner 的 known control，残差被消掉。正在修复。

**RS1_GO = false**；R1-RS2 保持锁定。完整结果见 `REPORT/R1_RS1_REPORT.md`。

### 已完成
- 泄漏修复后重跑 plumbing smoke（schema / hidden 隔离 / C2 latch / H32 blind 均通过）
- 正式 200 条：`runs/r1_rs1/formal/`
- 协议已按 prereg 冻结，**未**根据结果改 threshold / 加 MuJoCo operator

### 硬 gate 摘要
| Gate | 结果 |
|------|------|
| C0 false revision | **0%** ✓ |
| C1 exact recovery | **73.8%**（L 50% / H 97.5%）✗ |
| C1 系数 ≤10% | **84.7%**（差一线）✗ |
| CNEG accepted active | **0**（trigger 47.5%）✓ |
| C2 条件 rejection | **vacuous**（trigger=0）✗ |
| accepted H32 stability | **53.5%** ✗ |
| C1 seed-level H32 gain | **mean > 0** ✓（CI 下界未 >0）|

### Funnel 诊断（不调参）
1. **C1-L**：proposal 几乎完美（truth in top-3 = 100%），瓶颈在 **trigger**（52.5%）→ inadequacy detection  
2. **C2**：latch 有 unsupported 样本，但 **完全不 trigger** → 不能声称 model-class rejection  
3. **H32**：短视界常 accept，长视界 **stability 崩** → 回到 R0.3/R0.4 类 long-horizon failure  
4. **CNEG**：passivity 把 \(\hat\alpha\to 0\)，硬 safety gate 通过；不是“看不见”

按 prereg：**不要直接 tuning R0.6**。下一步应新开针对 trigger coverage / H32 stability 的预注册实验，而不是打开 RS2。

---

**User**

8. 下一步不要做一个同时改 trigger + stability 的大实验
这里我建议严格拆成两个 scientific questions。
R1-RS1A：Trigger Evidence Study
只研究：
\[
\boxed{
\text{What constitutes sufficient evidence of model inadequacy?}
}
\]冻结所有 revision / acceptance / passivity / rollout mechanism。
只比较 trigger evidence formulation。
核心 case：
C0
C1-L
C1-H
C2-latch
CNEG可以作为 specificity/safety diagnostic，但不是主 endpoint。
重点不是“把 trigger rate 调高”，而是研究：
\[
r_\perp
\]单独是否不足，以及加入：
\[
\text{temporal persistence}
\]\[
\text{support violation}
\]\[
\text{posterior-normalized residual}
\]是否能恢复 weak structural signal。
我最推荐比较四种 detector：
\[
D_0=\|r_\perp\|
\]当前 baseline。
\[
D_1=
\frac{\|r_\perp\|}
{\sigma_{\mathrm{pred}}+\epsilon}
\]即 uncertainty-normalized structural residual。
\[
D_2=
\text{persistent evidence accumulation of }D_1
\]例如 sequential log evidence，而不是单窗口 threshold。
最后：
\[
D_3=
D_2+\text{support/validity evidence}.
\]这里的科学问题不是：
哪个 threshold 最好？

而是：
\[
\boxed{
\text{Does structural detection require normalized,
persistent, multi-source evidence?}
}
\]这是很有论文价值的问题。
9. R1-RS1B：Long-Horizon Stability Study
RS1A 完成之前都可以先独立开发，但不要混成同一个 confirmatory experiment。
研究：
\[
\boxed{
\text{Why do H2/H4/H8-approved admissible revisions fail at H32?}
}
\]这次不要再次只试：
\[
H16
\]或者：
\[
H32 acceptance rollout
\]因为 R0.4 已经告诉你：
\[
\text{extend validation horizon}
\]本身不是根本解。
我建议研究 dynamics-level invariants。
例如对于 hinge dynamics：
\[
M(q)\ddot q+h(q,\dot q)=\tau+r(q,\dot q)
\]接受 revision 后分析：
1. effective damping
\[
d_{\mathrm{eff}}(q,v)
=
-\frac{\partial r}{\partial v}
\]至少应该在 rollout support 范围内保持合理。
2. energy production
\[
P_r=v^\top r
\]这是 R0.5P 已经成功的维度。
但 RS1 说明仅仅：
\[
P_r\le0
\]还不够。
3. contraction / local amplification
研究 revised vector field：
\[
\dot x=f_{\mathrm{rev}}(x)
\]的 Jacobian：
\[
J_x=
\frac{\partial f_{\mathrm{rev}}}{\partial x}.
\]不一定要求：
\[
\rho(J)<1
\]这种离散条件直接成立，而是看 candidate revision 相对于 incumbent 是否增加局部扰动增长率。
比如：
\[
\lambda_{\max}
\left(
\frac{J+J^\top}{2}
\right)
\]可以作为 local expansion measure。
这可能比单纯 passivity 更接近：
\[
\boxed{\text{rollout robustness}}
\]10. 我建议研究顺序先 RS1A，再 RS1B
原因是 trigger failure 发生在 pipeline 更前面：
\[
\text{trigger}
\rightarrow
\text{proposal}
\rightarrow
\text{acceptance}
\rightarrow
\text{H32}.
\]如果 trigger coverage 只有 52.5% / 0%，那么后面对 revision utility 的统计样本已经被严重 selection biased。
因此更干净的顺序是：
\[
\boxed{
R1\text{-RS1A Trigger}
\rightarrow
R1\text{-RS1B Stability}
\rightarrow
R1\text{-RS1C integrated confirmation}
}
\]最后才考虑 RS2。
11. RS1A 不应该直接以 recovery 为优化目标
这是一个重要设计点。
否则非常容易退化成：
\[
\text{降低 trigger threshold}
\]然后 recovery 上升。
真正应该画：
\[
\boxed{
\text{trigger recall}
\leftrightarrow
\text{C0 false trigger}
}
\]也就是 inadequacy-detection ROC / PR。
特别是：
C0 = negative
C1-L/H = known structural positive
C2 = outside-library structural positive
这样先评价：
\[
\text{Does the detector recognize inadequacy?}
\]再谈：
\[
\text{Can the system explain it?}
\]把 V3 的核心问题重新独立出来。
12. 这使整个 APR-WM 科学链又增加了一个重要结果
现在可以把后半段改写成：
\[
\boxed{
\begin{aligned}
R0.1 &: \text{physics-consistent representation is necessary}\\
R0.3 &: \text{local fit }\not\Rightarrow\text{ rollout utility}\\
R0.4 &: \text{short-safe }\not\Rightarrow\text{ long-safe}\\
R0.5P &: \text{passivity strongly predicts stability}\\
R0.6 &: \text{physical admissibility removes catastrophic failures}\\
MJ0/RS0 &: \text{force-space closure transfers across engines}\\
RS1 &: \text{physical safety transfers better than epistemic detection and long-horizon utility}
\end{aligned}
}
\]RS1 因此并不是简单的“R0.6 没泛化”。
它更精确地说明：
\[
\boxed{
\text{APR-WM 的不同机制具有不同的跨域可迁移性}
}
\]其中：
physics representation：迁移很好；
operator proposal：迁移很好；
passivity prior：迁移很好；
weak inadequacy detection：迁移不足；
outside-support → inadequacy coupling：缺失；
short → long utility calibration：迁移不足。
这实际上比一个全面 RS1_GO 更有研究价值，因为它告诉你下一代 APR-WM 应该改哪里，而不是整体重做。
我建议下一正式阶段命名为：
\[
\boxed{\text{R1-RS1A — Multi-Evidence Model-Inadequacy Detection}}
\]核心假设可以直接写成：
A transferable model-inadequacy detector should accumulate normalized persistent residual evidence together with model-support violations, rather than triggering from residual magnitude alone.

这会是当前最自然、且与 V1→V3 的 epistemic 主线完全连续的下一步。

---

**Cursor**

先把 RS1 的科学结论写进链路上，并冻结 **R1-RS1A** 的 prereg（只比 detector，不碰 revision）。接着实现四档 detector 与 ROC/PR 评估骨架。

C2 的 support 证据应来自 probe 轨迹（RS1 已见 ~7% unsupported）；discovery 采样几乎碰不到 latch。正在把 `u` 改成按 prereg 从 probe support 统计，然后再跑正式矩阵。

正式结果里 C2 的 `u≡0`（probe 到不了 latch 带），导致 D3≡D2，假设无法检验。正在把冻结的 support audit 写入 prereg 并重跑。

**已按你的拆分落地：先 RS1A（只谈 detector），RS1B 锁定不混做。**

### 已冻结 / 已完成
- `REPORT/R1_RS1A_PREREG.md`：科学问题、D0–D3 公式、ROC/PR 主 endpoint（**禁止**以 recovery 调参）
- `REPORT/R1_RS1B_PREREG.md`：仅 locked draft
- 实现 `aprwm_v0/r1_rs1a.py` + CLI `r1-rs1a`（**不跑** revision/acceptance/H32）
- 正式矩阵 160+40：`runs/r1_rs1a/formal/`，报告 `REPORT/R1_RS1A_REPORT.md`
- 科学链已改写为差分可迁移性（`APR-WM思路.md` / `REPORT/R1_RS1_REPORT.md`）

### 结果：`RS1A_GO = false`，但是很干净的机制结论

| 机制 | 结论 |
|------|------|
| \(D_0=\|r_\perp\|\) | C1 很强；**C2 近乎无效**（AUROC 0.60，FPR≤1% 时 C2 recall = 0） |
| \(D_1\) 归一化 | 小幅帮助 |
| \(D_2\) 持久累积 | **损害** known-structural ROC |
| \(D_3\) + support | **C2 AUROC = 1.0**，同工作点 C2 recall 0→1.0 |

也就是说：RS1 的 C2 vacuity 是 **detector 缺 support 耦合**，不是“没有结构”；但同一套 \(D_3\) **救不了 C1-L**，且当前 persistence 形式有害。

### 下一步（仍不要开 RS1B）
建议新开一个很窄的 follow-up prereg：试 \(D_3'=D_1+\gamma(-\log(1-u))\)（去掉有害的 \(D_2\)，保留 support）。只有在 **C1-L 与 C2 在 FPR≤1% 同时改善** 后，再解锁 RS1B 的 dynamics-invariant / H32 研究。

---

**User**

同意这个判断，而且 **RS1A 的信息量比一个简单的 GO 更大**。它把 RS1 里的“C2 完全不 trigger”定位成了一个非常明确的机制缺口：**support evidence 没有进入 inadequacy detector**，而不是 latch 没产生结构变化。与此同时，它也证明了你原先设想的 persistence 累积形式并不适合当前问题。

最关键的结果其实是三条：

\[
\boxed{
D_0\ \text{擅长 in-library structural signal，但对 C2 基本失明}
}
\]

\[
\boxed{
D_3\ \text{通过 support coupling 完全恢复 C2，但牺牲了 C1 排序}
}
\]

\[
\boxed{
\text{问题不是“要不要多证据”，而是“不同证据应该怎样组合”}
}
\]

这和 RS1 的原始 funnel 是完全一致的：C1-L 的问题是 weak-signal trigger，而 C2 的问题是 support/validity evidence 没进入 trigger。

因此下一步不应该再测试“更强的 persistence”，而应该非常窄地验证：

\[
\boxed{
D_3'
=
D_1
+
\gamma\left[-\log(1-u)\right]
}
\]

也就是：

- 保留 \(D_1\) 的 uncertainty normalization；
- 删除已经被证伪的 \(D_2\) persistence 项；
- 保留 support coupling；
- 不碰 revision / acceptance / H32。

我建议把它单独命名为：

\[
\boxed{
R1\text{-RS1A.1 — Compositional Inadequacy Evidence}
}
\]

## 建议的 prereg 核心

科学问题写成：

\[
\boxed{
\text{Can support evidence improve outside-library detection
without degrading weak in-library structural detection?}
}
\]

不要写成“D3' 是否比 D3 好”，因为那太 implementation-specific。真正要验证的是：

\[
\text{support evidence should be additive/conditional, not coupled through harmful persistence}
\]

正式比较只需要：

\[
D_0,\quad D_1,\quad D_3,\quad D_3'
\]

不必再保留 \(D_2\) 作为主方法；它已经完成机制消融任务，可以继续报告，但不应占据 primary comparison。

---

## 最重要的 gate

这次不要用 pooled AUROC 作为唯一 GO，因为 RS1A 已经证明 pooled metric 会掩盖 C1 与 C2 的机制差异。

建议设三个独立硬条件：

\[
\boxed{
R_{\text{C1-L}}(D_3')
>
R_{\text{C1-L}}(D_0)
}
\]

\[
\boxed{
R_{\text{C2}}(D_3')
>
R_{\text{C2}}(D_0)
}
\]

都在同一个：

\[
\boxed{\text{C0 FPR}\le1\%}
\]

工作点比较。

再加一个 non-degradation 条件：

\[
\boxed{
AUROC_{\text{C1}}(D_3')
\ge
AUROC_{\text{C1}}(D_0)-\delta
}
\]

其中 \(\delta\) 建议预注册为一个很小的容忍量，例如 0.01 或 0.02，避免出现“为了救 C2，把 known structural detector 明显搞坏”。

---

## \(\gamma\) 怎么处理

这里尤其不要在 formal 数据上扫 \(\gamma\)。

最好提前定义一个小的 dev-only selection：

\[
\gamma\in\{0.25,0.5,1,2,4\}
\]

只用独立 development seeds 选择一次，然后冻结。

选择目标不要是 recovery，而是：

\[
\max_\gamma
\min
\left(
R_{\text{C1-L}},
R_{\text{C2}}
\right)
\]

subject to：

\[
FPR_{\text{C0}}\le1\%.
\]

这个目标比 pooled recall 更合适，因为它直接防止：

\[
\text{C2=100\%, C1-L\text{ 仍然不动}
\]

这种“单侧成功”。

正式 held-out seeds 再单独跑一次。

---

## 我还建议加一个更本质的对照

除了简单加法：

\[
D_3'=D_1+\gamma S(u)
\]

再加一个 gated 版本作为诊断：

\[
\boxed{
D_g=
\begin{cases}
D_1,&u=0\\
D_1+\gamma S(u),&u>0
\end{cases}
}
\]

其中：

\[
S(u)=-\log(1-u).
\]

在你当前数据里，报告显示 C2 的 \(u=0.156\) 而其他 regime \(u=0\)。

因此当前数据上 \(D_g\) 和加法形式可能数值相同，但概念上不同：

- additive view：support 是连续 evidence；
- gated view：support 是独立 epistemic channel。

以后当 C1 或 C0 也出现非零 \(u\) 时，两者才会真正分叉。

这个 distinction 对后续 robot-contact RS2 很重要，因为 contact/boundary 状态下：

\[
u>0
\]

很可能不再意味着“model class 错了”。

---

## RS1A.1 最值得回答的问题

如果 \(D_3'\) 最终出现：

\[
C1\text{-L recall}\uparrow
\]

同时：

\[
C2 recall\uparrow
\]

且：

\[
C0 FPR\le1\%
\]

那么你就能得到一个比 RS1A 更强的结论：

\[
\boxed{
\text{transferable inadequacy detection requires
heterogeneous epistemic evidence,
but those evidences should remain compositionally separable}
}
\]

也就是：

> residual evidence 和 support evidence 都有用，但不应该先被一个错误的 temporal accumulator 混合后再统一判决。

反过来，如果 \(D_3'\) 仍然救不了 C1-L，那么结论也会非常明确：

\[
\boxed{
\text{C1-L is not a support-coupling problem}
}
\]

那时就不应该继续调组合方式，而要重新研究 weak-signal detector 本身，比如：

\[
\text{local SNR}
\]

\[
\text{matched structural statistic}
\]

\[
\text{sequential likelihood ratio}
\]

而不是 generic cumulative \(z^2\)。

---

## 现在不要开 RS1B，这个决定是对的

因为 RS1A 的 detector 还没形成稳定输入分布。

如果现在进入 H32 stability：

\[
\text{accepted revision set}
\]

仍然会被 trigger selection bias 污染。

尤其 C1-L 只有很低 recall 时，你研究的其实是：

\[
\text{strong/easy subset 的 long-horizon stability}
\]

而不是完整 revision population。

所以正确顺序仍然是：

\[
\boxed{
RS1A
\rightarrow
RS1A.1
\rightarrow
RS1B
\rightarrow
RS1C
}
\]

其中 RS1A.1 的解锁条件我建议直接写成：

\[
\boxed{
\text{C1-L and C2 both improve at matched C0 FPR}
}
\]

而不是“总体 AUROC 提升”。

目前这条研究线已经非常清晰：

\[
\boxed{
\text{magnitude}
\rightarrow
\text{normalization}
\rightarrow
\text{persistence falsified}
\rightarrow
\text{support coupling validated}
\rightarrow
\text{compositional evidence next}
}
\]

这已经开始形成一条独立且可发表的 **model inadequacy detection** 子结论，而不只是 APR-WM 工程调参。

---

**Cursor**

按你的协议开 **RS1A.1**：先冻结 prereg，再用独立 dev seeds 选一次 γ，最后在 held-out 上只比 \(D_0/D_1/D_3/D_3'/D_g\)。

**RS1A.1_GO = false**，但结论比 GO 更硬：`C1-L is not a support-coupling problem`。报告见 `REPORT/R1_RS1A1_REPORT.md`。

### 做了什么
- 冻结 `REPORT/R1_RS1A1_PREREG.md`
- Dev seeds `{9301…}` 选 γ → **γ=0.25**（C1-L recall 全 γ 并列，取最小 γ）
- Formal held-out `{9401…}`；主比 \(D_0,D_1,D_3,D_3',D_g\)；不跑 revision/H32
- RS1B **仍锁**

### 硬 gate
| Gate | 结果 |
|------|------|
| \(R_{\mathrm{C1-L}}(D_3')>R(D_0)\) @ FPR≤1% | 0.125 ≯ 0.725 ✗ |
| \(R_{\mathrm{C2}}(D_3')>R(D_0)\) | 0.10 > 0.025 ✓ |
| C1 AUROC 非退化 (δ=0.02) | 0.922 ≱ 0.953 ✗ |

### 机制要点
在 \(u=0\) 的 C0/C1 上 \(D_3'\equiv D_1\equiv D_g\)，所以 **support 组合无法优于 \(D_1\)**；而本 held-out 上 matched-FPR 的 C1-L 仍是 **\(D_0\) 最强**。C2 继续证明 support 有用，但救不了 weak in-library。

研究线应变为：

\[
\text{support validated}\rightarrow\text{composition cannot rescue C1-L}\rightarrow\text{weak-signal detector next}
\]

不要继续扫 γ，也不要开 RS1B。

---

**User**

这次结果把问题进一步切干净了。现在已经可以把 **RS1A → RS1A.1** 的结论收敛为：

\[
\boxed{
\text{C2 是 support-evidence problem，C1-L 是 weak-signal detection problem}
}
\]

这两个问题必须分开处理。

RS1A 已经证明 support coupling 对 C2 有效：原始 \(D_0\) 在 C2 上几乎接近随机，而加入 support evidence 后可以把 outside-library 情形很好地区分出来；同时 persistence 项 \(D_2\) 会明显破坏 known-structural discrimination。

而你这次 RS1A.1 的结果更进一步：在 C0/C1 中 \(u=0\)，因此

\[
D_3'=D_1+\gamma[-\log(1-u)]
\]

实际上退化为

\[
\boxed{D_3'=D_1}
\]

所以无论怎么扫 \(\gamma\)，理论上都不可能解决 C1-L。这个 follow-up 因而不是一次“参数没调好”的失败，而是**直接 falsify 了 support-composition 作为 C1-L 解法的假设**。

更重要的是，held-out 上：

\[
R_{\mathrm{C1-L}}(D_0)=0.725
\]

而

\[
R_{\mathrm{C1-L}}(D_3')=0.125
\]

在相同 \(FPR\le1\%\) 下差距很大。说明当前 weak in-library detection 中，**原始 residual magnitude 本身携带的信息反而比 uncertainty-normalized score 更强**。

因此现在不应该继续做：

\[
D_0\rightarrow D_1\rightarrow D_1+\gamma S(u)
\]

这一条组合线了。

---

## 下一步真正的问题

现在应该把问题改写成：

\[
\boxed{
\text{Why does a weak but persistent in-library structural residual
fail to separate from adequate physics?}
}
\]

而不是：

\[
\text{How should support evidence be combined?}
\]

我建议下一阶段单独定义为：

\[
\boxed{
R1\text{-RS1A.2 — Weak Structural Signal Detection}
}
\]

仍然：

- 不运行 revision；
- 不运行 acceptance；
- 不运行 passivity；
- 不看 H32；
- RS1B 保持锁定。

---

# RS1A.2 应该测什么

这一次不要再从“多证据加权”出发，而应该研究 **C1-L 与 C0 的统计分布到底差在哪里**。

第一步建议冻结成纯诊断实验，比较至少四类 statistic：

\[
D_0=\|r_\perp\|
\]

作为现有 strongest baseline。

然后考虑：

\[
D_{\mathrm{SNR}}
=
\frac{\mathbb E[\|r_\perp\|]}
{\sqrt{\operatorname{Var}(\|r_\perp\|)}+\epsilon}
\]

检验结构信号是不是“幅值不大，但比噪声更稳定”。

再考虑方向一致性：

\[
D_{\mathrm{dir}}
=
\left\|
\frac1T
\sum_t
\frac{r_{\perp,t}}
{\|r_{\perp,t}\|+\epsilon}
\right\|
\]

如果 C0 residual 主要是零均值噪声，其方向应相互抵消；而 C1-L 即使幅值弱，只要由固定 operator 产生，就可能有稳定方向结构。

还可以直接做 lag consistency：

\[
D_{\mathrm{corr}}
=
\frac1{T-1}
\sum_t
\frac{
r_{\perp,t}^{\top}r_{\perp,t+1}
}{
\|r_{\perp,t}\|\|r_{\perp,t+1}\|+\epsilon
}
\]

测 residual 是否存在时间结构，而不是像 \(D_2\) 那样直接累加 \(z^2\)。

---

## 为什么这比继续做 persistence 合理

之前的：

\[
D_2=\sum_t \frac12(z_t^2-1)
\]

实际上把“持续存在”近似成了：

\[
\text{持续的大 squared deviation}
\]

因此它会强烈依赖 calibration。

只要 normalization 稍微错一点，就可能让 C0 和 C1 的 evidence 累积都失真。

但真正想抓的 persistence 更接近：

\[
\boxed{
\text{consistent structure across time}
}
\]

而不是：

\[
\boxed{
\text{large residual repeatedly appears}
}
\]

这是两个不同假设。

所以 RS1A.2 应该 falsify：

\[
\text{magnitude consistency}
\]

\[
\text{direction consistency}
\]

\[
\text{temporal correlation}
\]

哪一个才是真正携带 C1-L 信号的量。

---

# 一个尤其值得测的 statistic：operator matched evidence

由于 C1-L truth 本身来自：

\[
r_\tau=\alpha |v|v
\]

而 operator library 已经证明 proposal recall 近乎 100%，其实可以构造：

\[
\phi(v)=|v|v
\]

然后测：

\[
D_{\mathrm{match}}
=
\frac{
\left|\sum_t r_{\perp,t}\phi(v_t)\right|
}{
\sqrt{\sum_t \phi(v_t)^2}+\epsilon
}
\]

它本质上是在问：

> residual 是否不仅“大”，而且沿某个 candidate operator 的 signature 持续排列？

这和当前 \(D_0\) 很不同。

\(D_0\)：

\[
\text{Is residual large?}
\]

而 \(D_{\mathrm{match}}\)：

\[
\text{Is residual structurally aligned with a plausible missing operator?}
\]

这与 APR-WM 的 model-discovery 主线更一致。

如果它在 C1-L 上明显好于 \(D_0\)，会得到一个很有价值的结论：

\[
\boxed{
\text{weak model inadequacy is easier to detect in operator-aligned space
than in residual-energy space}
}
\]

---

# 但要防止 information leakage

这里不能把 true C1 operator 直接给 detector。

正确方式应该是：

\[
D_{\mathrm{lib}}
=
\max_{k\in\mathcal L}
D_{\mathrm{match}}^{(k)}
\]

其中 \(\mathcal L\) 是**冻结的现有 operator library**。

这样 detector 只是问：

> residual 是否和任何已知 structural direction 显著匹配？

而不是使用 ground-truth label。

同时 C2 应该继续由独立 support channel：

\[
D_{\mathrm{support}}=S(u)
\]

负责。

最终 architecture 很可能不再是一个统一 scalar：

\[
D=D_{\mathrm{residual}}+\gamma D_{\mathrm{support}}
\]

而是两个 epistemic routes：

\[
\boxed{
\begin{cases}
D_{\mathrm{known}}
&:\text{operator-aligned structural evidence}\\
D_{\mathrm{unknown}}
&:\text{support / validity violation}
\end{cases}
}
\]

然后：

\[
\text{trigger}
=
D_{\mathrm{known}}>\tau_k
\;\lor\;
D_{\mathrm{unknown}}>\tau_u.
\]

这其实比前面的 scalar fusion 更符合目前实验结果。

---

# 当前科学链可以进一步更新

现在可以写成：

\[
\boxed{
\begin{aligned}
RS1 &: \text{weak structural + outside-library trigger fail}\\
RS1A &: \text{support evidence solves C2; naive persistence hurts C1}\\
RS1A.1 &: \text{support composition cannot solve C1-L}\\
RS1A.2 &: \text{next: identify the statistic that reveals weak in-library structure}
\end{aligned}
}
\]

而最高层的 detector 认识也变了：

最开始的假设是：

\[
\boxed{
\text{one scalar inadequacy score}
}
\]

现在证据越来越支持：

\[
\boxed{
\text{model inadequacy is multi-causal and may require typed evidence channels}
}
\]

即：

- **known-but-missing structure**：看 residual/operator alignment；
- **outside-support / unknown structure**：看 support violation；
- 未来 parameter uncertainty 可能又是第三类。

这和你整个 APR-WM 的思想其实更一致：

> 系统不只是判断“错没错”，而是应该判断“以什么方式错”。

因此我同意你现在的 protocol：

\[
\boxed{
\text{不要继续扫 }\gamma
}
\]

\[
\boxed{
\text{不要打开 RS1B}
}
\]

下一步应该只开一个窄的 **RS1A.2 weak-signal detector preregistration**，重点比较 residual magnitude 与 structural alignment / temporal coherence，而不是继续做 support fusion。

---

**Cursor**

同意：C2 与 C1-L 已拆开。开始冻结并实现 **RS1A.2**（弱结构信号统计量对比 + typed channels，不碰 γ/RS1B）。

**RS1A.2_GO = false**；完整结果见 `REPORT/R1_RS1A2_REPORT.md`。RS1B 仍锁定。

### 收敛结论
\[
\text{C2 = support-evidence},\quad
\text{C1-L = weak-signal（仍由 magnitude }D_0\text{ 主导）}
\]

### C0 vs C1-L（FPR≤1%）
| Statistic | AUROC | Recall |
|-----------|------:|-------:|
| \(D_0\) | **0.943** | **0.475** |
| \(D_{\mathrm{lib}}\) | 0.806 | 0.325 |
| \(D_{\mathrm{SNR}}\) / \(D_{\mathrm{dir}}\) / \(D_{\mathrm{corr}}\) | ≈0.5–0.59 | ≈0 |

operator-aligned 假设被 **falsify 为“能打赢 \(D_0\)”**；\(D_{\mathrm{lib}}\) 有信息，但不够。coherence 族基本无效。

### Typed channels（次要、成立）
\(D_0\lor D_{\mathrm{support}}\)：C0 FPR=0，C1-L=0.475，**C2=1.0**  
→ known / unknown 应分通道；known 端目前仍应用 \(D_0\)。

### 额外现象
C1-L 上 `lib_best` 常落在 `signed_v2` 而非 `abs_v_v`（无标签匹配的代价）。

### 下一步（仍不要开 RS1B / 不要扫 γ）
剩余 ~52% C1-L miss 更像 **与 C0 的 magnitude 重叠**，优先查 excitation / discovery-window，或受限 matched filter——不是再换标量变换。

---

**User**

这次 RS1A.2 基本把 weak-signal detector 这一支压缩到了一个很明确的结论：

\[
\boxed{
\text{剩余的 C1-L miss 不是“特征工程还没找到对的标量”，而更像观测设计不足}
}
\]

报告里最有决定性的证据是：在同一批 held-out 数据上，\(D_0\) 已经达到 AUROC 0.943，但在 C0 FPR \(\le1\%\) 的严格工作点仍只有 47.5% recall；而 SNR、方向一致性、时间相关性几乎接近随机，library-aligned \(D_{\mathrm{lib}}\) 虽然有信息，但也明显弱于 \(D_0\)。

所以现在可以相当有把握地排除几种解释：

\[
\boxed{
\begin{aligned}
\text{不是 support fusion 问题} &\quad\checkmark\\
\text{不是简单 persistence 问题} &\quad\checkmark\\
\text{不是 residual direction coherence 问题} &\quad\checkmark\\
\text{不是 unrestricted operator-match 能解决的问题} &\quad\checkmark
\end{aligned}
}
\]

剩下更像：

\[
\boxed{
p(D_0\mid C1\text{-L})
\text{ 与 }
p(D_0\mid C0)
\text{ 在当前 excitation/window 下存在真实重叠}
}
\]

这比“detector 设计得不够复杂”更本质。

---

## 1. Typed-channel 这条结论现在已经比较稳

这次：

\[
D_0\lor D_{\mathrm{support}}
\]

在 FPR=0 时同时做到：

- C1-L = 0.475
- C1-H = 1.0
- C2 = 1.0

说明目前最合理的 epistemic architecture 已经不是：

\[
D_{\mathrm{one\ scalar}}
\]

而是：

\[
\boxed{
\begin{aligned}
E_{\mathrm{known}} &\leftarrow \|r_\perp\|\\
E_{\mathrm{unknown}} &\leftarrow \text{support violation}
\end{aligned}
}
\]

最后由逻辑 OR 或更高层决策器整合。

这个结果我建议冻结，不再继续优化 support branch。

---

# 2. `signed_v2` 抢过 `abs_v_v` 很值得注意

报告里 C1-L 的 `lib_best_operator`：

- `signed_v2`: 16/40
- `abs_v_v`: 4/40

这说明当前 excitation 下：

\[
|v|v
\]

和某些 correlated velocity basis 在有限窗口内并不容易辨别。

这不只是 proposal ambiguity，它提示一个更深的问题：

\[
\boxed{
\text{observational identifiability depends on excitation}
}
\]

如果轨迹主要落在：

\[
v>0
\]

或很窄的 velocity support，那么：

\[
|v|v=v^2
\]

在该区域里几乎就是：

\[
\text{signed/unsigned quadratic feature}
\]

因此仅靠当前 passive window，理论上就缺乏足够信息区分 operator。

这意味着继续改 detector 很可能收益有限。

---

# 3. 下一步应该从“检测器设计”切换到“可辨识性设计”

我建议下一阶段不要叫 RS1A.3 detector，而是明确换科学问题，例如：

\[
\boxed{
R1\text{-RS1A.3 — Excitation-Limited Inadequacy Identifiability}
}
\]

核心问题：

> C1-L 的 52.5% miss 是否主要来自 trajectory 没有进入足够 informative 的状态区域？

这会比继续比较 \(D_4,D_5,D_6\) 更干净。

---

# 4. 最优先测 excitation，而不是 discovery-window length

我会先测 excitation。

因为对于：

\[
r_\tau=\alpha |v|v
\]

signal magnitude 本身就是：

\[
|r_\tau|=|\alpha|v^2.
\]

低速度区：

\[
v\approx0
\Rightarrow
r_\tau\approx0.
\]

无论 detector 多聪明，都不可能从几乎没有信号的数据中稳定发现结构。

因此一个直接预测是：

\[
\boxed{
\text{C1-L miss probability should decrease as velocity excitation increases}
}
\]

这可以被严格 falsify。

---

# 5. 推荐一个非常窄的 excitation experiment

保持：

- C0
- C1-L
- frozen \(D_0\)
- frozen simulator
- 不做 revision
- 不做 H32

只改变 excitation regime。

比如：

\[
A\in\{0.5A_0,\;A_0,\;1.5A_0,\;2A_0\}
\]

再加 frequency variation：

\[
f\in\{f_{\mathrm{low}},f_{\mathrm{mid}},f_{\mathrm{high}}\}.
\]

主要指标不是 recovery，而是：

\[
\boxed{
AUROC(C0,C1\text{-L})
}
\]

和：

\[
\boxed{
Recall_{C1-L}\ @\ FPR_{C0}\le1\%
}
\]

同时记录：

\[
\max|v|,\quad RMS(v),\quad RMS(|v|v)
\]

作为 exposure 指标。

如果 recall 随：

\[
RMS(|v|v)
\]

单调上升，那么几乎可以确认：

\[
\boxed{
\text{weak-signal failure is excitation-limited}
}
\]

---

# 6. 更好的变量其实是“operator exposure”

比 amplitude 更理论化的量是：

\[
X_\phi
=
\frac1T
\sum_t \phi(v_t)^2
\]

其中：

\[
\phi(v)=|v|v.
\]

也就是：

\[
X_\phi
=
\frac1T
\sum_t v_t^4.
\]

因为这个量直接衡量当前 trajectory 对该 operator 的激励程度。

可以看：

\[
P(\text{detect}\mid X_\phi).
\]

如果存在很明显的 sigmoid：

\[
P(\text{detect})
\approx
\sigma(aX_\phi+b),
\]

那会得到一个很漂亮的结论：

\[
\boxed{
\text{model inadequacy detectability is governed by operator excitation}
}
\]

这比“幅值大一点更容易检测”更有一般性。

---

# 7. 然后再研究 discovery window

只有 excitation study 后，再看 window：

\[
T\in\{T_0/2,T_0,2T_0,4T_0\}.
\]

因为 window 延长有两种可能：

### 如果主要是 i.i.d. noise

signal accumulation 大概改善：

\[
SNR\propto\sqrt{T}.
\]

那 recall 应随 \(T\) 提升。

### 如果主要是 trajectory 没进入 informative region

那么单纯延长窗口可能几乎没帮助。

这两种结果能进一步区分：

\[
\boxed{
\text{sample-limited}
\quad vs\quad
\text{excitation-limited}
}
\]

---

# 8. 受限 matched filter 可以保留，但应放到 excitation 后

你提到 constrained matched filter，我认为值得测，但不要先做。

因为 unrestricted：

\[
D_{\mathrm{lib}}
=
\max_{k\in\mathcal L}D_k
\]

会产生 multiple-comparison / correlated-feature 问题。

如果只允许 physically compatible subset：

\[
\mathcal L_{\mathrm{diss}}
=
\{k:\;v^\top r_k(v)\le0\},
\]

定义：

\[
D_{\mathrm{lib,diss}}
=
\max_{k\in\mathcal L_{\mathrm{diss}}}D_k,
\]

确实可能减少 `signed_v2` 之类错误抢占。

但它回答的是：

\[
\text{如何利用已有 physics prior 改善 known-structure detection}
\]

不是：

\[
\text{为什么信号本身不可见}.
\]

所以应该作为第二步。

---

# 9. 这和你刚才提出的“允许小误差”其实正好接起来

现在尤其应该同时测：

\[
\boxed{
\text{detectability}
}
\]

和：

\[
\boxed{
\text{consequence}
}
\]

对于不同 \(\alpha\)：

\[
\alpha\in
\{-0.03,-0.06,-0.09,-0.12,-0.18,-0.24\}
\]

画两条曲线：

第一条：

\[
R(\alpha)
=
Recall_{FPR\le1\%}.
\]

第二条：

\[
C(\alpha)
=
\text{no-revision future cost}.
\]

真正理想的系统只需要保证：

\[
\boxed{
\alpha\text{ 开始产生不可接受 consequence 之前，
detector 已开始可靠识别}
}
\]

而不是要求：

\[
\forall\alpha\neq0,\quad recall\approx1.
\]

这会把“弱信号检测失败”转化成更合理的 decision-theoretic 问题。

---

# 10. 所以下一步我建议做成二维图，而不是单一 benchmark

横轴：

\[
\text{operator exposure / mismatch strength}
\]

纵轴同时画：

\[
\text{detection recall}
\]

以及：

\[
\text{future consequence}.
\]

最后可能得到：

```text
                 consequential
                      ↑
                 revision zone
                    /
                   /
            probe /
                 /
----- tolerance boundary -----
             /
            /
      tolerated mismatch
──────────────────────────────→ exposure
```

这会直接回答：

> 52.5% miss 到底是不是实际问题？

如果 miss 全部集中在 consequence 很小的区域，那么当前 detector 可能已经足够。

如果大量高 consequence trajectory 仍然 miss，那才说明需要 active excitation / probing。

---

## 当前最合理的科学链

现在我会把它整理成：

\[
\boxed{
\begin{aligned}
RS1A &: \text{support channel repairs C2}\\
RS1A.1 &: \text{support cannot repair C1-L}\\
RS1A.2 &: \text{alternative residual statistics do not beat magnitude}\\
RS1A.3 &: \textbf{test whether C1-L detectability is excitation-limited}\\
RS1A.4 &: \textbf{compare detectability with consequence / tolerance}
\end{aligned}
}
\]

RS1B 继续锁定是合理的。

因为现在还没回答最基本的问题：

\[
\boxed{
\text{那些检测不到的 C1-L，究竟是不是值得检测？}
}
\]

这其实已经从“怎么提高 detector recall”转成了一个更成熟的科学问题：

\[
\boxed{
\text{When should a world model spend epistemic effort on a small mismatch?}
}
\]

这很可能会成为 APR-WM 从“异常检测器”走向真正 **adaptive epistemic resource allocation** 的关键一步。

---

**Cursor**

同意：把 support 通道冻结，下一阶段只检验 **excitation-limited identifiability**。开始写 RS1A.3 prereg 并跑窄的 excitation 矩阵。

**RS1A.3_GO = true**：C1-L miss 已确认为 **excitation-limited**。报告见 `REPORT/R1_RS1A3_REPORT.md`。

### 关键结果
| \(A/A_0\) | Seed-mean C1-L recall @ FPR≤1% |
|--------:|-------------------------------:|
| 0.5 | **0.00**（\(X_\phi\sim10^{-15}\)，AUROC≈0.5） |
| 1.0 | 0.13 |
| 1.5 | 0.60 |
| 2.0 | **0.80** |

- Spearman\((X_\phi,\mathrm{detect})=+0.56\)（seed-mean **+0.90**）
- 同幅下 **低频** \(X_\phi\) 更大、更好检（需要速度支撑，不是乱抖力矩）
- C0 的 \(D_0\) 全程贴噪声底 ≈0.002，激励没有抬高误报

### 已冻结
Typed 架构不变：\(E_{\mathrm{known}}=D_0\)，support 通道不再优化。

### 状态
- **RS1A.4** draft 已开（`REPORT/R1_RS1A4_PREREG.md`）：detectability ↔ consequence  
- **RS1B 仍锁**

---

**User**

这次结果把 C1-L 的机制解释基本闭环了：

\[
\boxed{
\text{C1-L 的主要瓶颈不是 detector statistic，而是 trajectory 没有进入足够 informative 的动力学区域}
}
\]

最强证据是 recall 随 excitation 单调上升：\(0.5A_0\to 2A_0\) 从 0 提升到 0.80；同时 \(X_\phi=\mathrm{mean}(v^4)\) 与检测显著正相关，而 C0 的 \(D_0\) 始终维持在约 0.002 的噪声底，没有因为更强激励而出现明显假阳性。

这说明之前那约一半 C1-L miss 的含义应该改写。不是：

\[
\text{“模型已经看到了异常，但 detector 不够聪明”}
\]

而更接近：

\[
\boxed{
\text{“观测轨迹根本没有激发出足够大的结构差异供 detector 使用”}
}
\]

对于

\[
r_\tau=\alpha |v|v
\]

有

\[
|r_\tau|=|\alpha|v^2,
\]

所以当 \(v\approx0\) 时，无论 \(D_0\)、matched filter 还是更复杂分类器，理论上可利用的信息都非常有限。报告中的低幅情况正符合这一点：\(X_\phi\sim10^{-15}\)，AUROC 接近 0.5。

更值得注意的是频率结果。同样的 torque amplitude 下，低频 excitation 反而更容易检测，因为它产生了更大的速度 excursion 和更高的 \(X_\phi\)。

因此真正有效的 probe 目标不是：

\[
\boxed{\text{make the action large}}
\]

而是：

\[
\boxed{\text{drive the system into states where competing dynamics disagree}}
\]

这和 V2 的 informative action 思路重新接上了，只不过现在信息目标已经从 parameter/model posterior IG 变成了 **operator exposure**。

---

### 现在可以冻结三个 detector 层结论

第一，known-structure channel：

\[
\boxed{
E_{\mathrm{known}}=D_0=\|r_\perp\|
}
\]

暂时没有必要继续寻找花哨的 scalar transform，因为 RS1A.2 已经证明其它统计量没有超过它，而 RS1A.3 又证明剩余 miss 可以由 excitation 解释。

第二，unknown/outside-support channel：

\[
\boxed{
E_{\mathrm{unknown}}=\text{support violation}
}
\]

继续独立保留。

第三，不应该试图把所有 epistemic evidence 压成一个统一 scalar：

\[
\boxed{
\text{typed evidence channels}
}
\]

目前比统一 fusion 更符合实验。

---

## RS1A.4 现在变得非常关键

它要回答的已经不是 detector engineering，而是一个 decision-theoretic 问题：

\[
\boxed{
\text{When is insufficient excitation actually a problem?}
}
\]

因为“检测不到弱 mismatch”本身未必值得修。

你现在应该联合测两条曲线。

第一条是 detectability：

\[
R(\alpha,X_\phi)
=
P(\text{detect}\mid\alpha,X_\phi).
\]

第二条是 consequence：

\[
C(\alpha,X_\phi)
=
L_{\mathrm{future}}^{\mathrm{no-revision}}
-
L_{\mathrm{future}}^{\mathrm{adequate}}.
\]

真正重要的是两条边界的相对位置。

理想情况：

\[
\boxed{
\text{detectability threshold}
<
\text{harm threshold}
}
\]

意思是，在 mismatch 真正开始造成不可接受长期后果之前，系统已经有能力发现它。

那剩余的小 mismatch 就完全可以进入：

\[
\text{tolerated mismatch}
\]

区域。

---

## 如果 RS1A.4 得到这种图，理论会明显成熟

可以定义：

\[
\boxed{
\begin{array}{ll}
C<C_{\mathrm{tol}}
&\Rightarrow \text{tolerate}\\[2mm]
C\ge C_{\mathrm{tol}},\ E\text{ weak}
&\Rightarrow \text{actively probe}\\[2mm]
C\ge C_{\mathrm{tol}},\ E\text{ strong}
&\Rightarrow \text{revise}
\end{array}
}
\]

这会把原来的二值：

\[
\text{trigger / no trigger}
\]

升级成：

\[
\boxed{
\text{tolerate}
\rightarrow
\text{probe}
\rightarrow
\text{revise}
}
\]

而且这不是凭设计直觉加入的，而是由 RS1A.2/A.3 的实验结果推出来的。

---

## 一个很重要的新含义：excitation 也应该是 adaptive resource

APR-WM 原来主要自适应分配：

- residual compute；
- informative action；
- model revision。

现在可以进一步明确：

\[
\boxed{
\text{epistemic excitation itself should be budgeted}
}
\]

系统不应该对每一个微小 residual 都主动探测。

更合理的是：

\[
\text{small residual}
\rightarrow
\text{tolerate}
\]

\[
\text{persistent but consequential uncertainty}
\rightarrow
\text{increase }X_\phi
\]

\[
\text{evidence becomes decisive}
\rightarrow
\text{revision / unknown}
\]

这样就真正形成：

\[
\boxed{
\text{prediction}
\rightarrow
\text{uncertainty}
\rightarrow
\text{value of excitation}
\rightarrow
\text{active evidence acquisition}
\rightarrow
\text{revision}
}
\]

---

## 现在仍然不该开 RS1B

因为 RS1A.4 还要决定：

> 哪些 weak mismatch 根本不需要被纳入 revision population？

如果不先回答这个问题，直接研究 H32 stability，会把很多“本来就应该 tolerate”的小修正也算进 stability analysis。

因此当前顺序非常合理：

\[
\boxed{
RS1A.3
\rightarrow
RS1A.4
\rightarrow
RS1B
}
\]

而 RS1A.4 结束后，RS1B 的研究对象应该不再是“所有 detected revision”，而是：

\[
\boxed{
\text{consequential + identifiable + revision-worthy mismatches}
}
\]

这会让后面的 H32 stability 研究干净很多。

目前这条科学链已经从“提高 detector recall”转成了更重要的问题：

\[
\boxed{
\text{When is acquiring more information worth its cost?}
}
\]

这实际上已经开始接近 APR-WM 最初的核心目标：**不是让世界模型永远最复杂、最精确，而是按认识价值决定何时容忍、何时取证、何时修正。**

---

**Cursor**

同意：机制已闭环，下一步做 **RS1A.4（detectability ↔ consequence）**。先冻结完整 prereg，再实现并跑联合曲线实验。

发现 confound：consequence 的 query 采样随 `amp_scale` 变了。改为仅由 `(seed, α)` 决定 query，然后重跑。

**RS1A.4_GO = false**（硬 gate 差一点），但 **tolerate / probe / revise_worthy** 三分区已经立住。报告：`REPORT/R1_RS1A4_REPORT.md`。

### 已冻结的 detector 层
- \(E_{\mathrm{known}}=D_0\)
- \(E_{\mathrm{unknown}}=\) support（offline）
- typed channels，不再做统一 scalar fusion

### 联合结果（query 已按 \((seed,\alpha)\) 去混淆）
| 分区 | n | 含义 |
|------|--:|------|
| tolerate | 72 | \(C<C_{\mathrm{tol}}\) |
| probe | 24 | 有害但未检出 → **需要提高 \(X_\phi\)** |
| revise_worthy | 24 | 有害且已检出 |

硬 gate：miss 中 tolerate 占 **64%**（要 ≥70% 未过）；另外两条（consequence 富集 detect、miss 的均值 \(C\) 更低）通过。

未过的原因很干净：网格故意包含 **强 \(\alpha\) × 弱激励**，这些 miss 正是 probe 区，不是“detector 又坏了”。

### 关键模式
对 \(\alpha\in\{-0.18,-0.24\}\)：  
\(0.5A_0\) → **probe**，\(2A_0\) → **revise_worthy**  
小 \(\alpha\)（≤0.09）几乎全程 **tolerate**。

### 状态
- **RS1B 仍锁**
- 若以后开 RS1B，人口应是  
  \(\boxed{\{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}\}}\)  
  而不是全部 trigger / 全部弱 mismatch

科学问题已从“提高 recall”转成：

\[
\text{When is acquiring more information worth its cost?}
\]

---

**User**

这次 RS1A.4 的意义很明确：**GO 没过，但核心决策结构反而被验证得更清楚。** 失败不是三分区不成立，而是实验网格故意包含了大量“强 mismatch × 弱 excitation”的 probe 区，因此 miss 里有 36% 是 consequential miss，导致 tolerated-miss rate 只有 64.2%，低于预注册的 70%。

最重要的新结论可以写成：

\[
\boxed{
\text{undetected}
\neq
\text{harmless}
}
\]

更精确地说：

\[
\boxed{
\text{undetected}
=
\text{tolerable}
\;\cup\;
\text{insufficiently-excited consequential mismatch}
}
\]

这就解释了为什么单纯追求 detector recall 不对：有些 miss 应该容忍，有些 miss 应该主动取证。

## 这次真正立住的是三态决策

现在已经有相当清楚的经验支持：

\[
\boxed{
\begin{aligned}
\text{tolerate}:&\quad C<C_{\mathrm{tol}}\\
\text{probe}:&\quad C\ge C_{\mathrm{tol}},\ \mathrm{detect}=0\\
\text{revise-worthy}:&\quad C\ge C_{\mathrm{tol}},\ \mathrm{detect}=1
\end{aligned}
}
\]

而且强 mismatch 下确实观察到：

\[
0.5A_0:
\text{probe}
\]

经过提高 excitation 后：

\[
2A_0:
\text{revise-worthy}.
\]

也就是说：

\[
\boxed{
\text{probe 不是失败状态，而是一个需要主动取证的 epistemic state}
}
\]

这是比之前 binary trigger 更成熟的一步。

---

## 为什么 64% 没到 70% 不应该解释成理论失败

因为这个 hard gate 实际测试的是：

\[
P(\text{harmless}\mid\text{miss})
\]

是否足够高。

但你的 formal grid 主动放进了：

\[
|\alpha|\in\{0.18,0.24\}
\]

同时配：

\[
0.5A_0
\]

这种非常低的信息激励。

于是自然制造出：

\[
\boxed{
\text{high consequence}
+
\text{low observability}
}
\]

的样本。

这正是 probe 区应该存在的条件。

因此从机制上看，这次结果其实是在 falsify：

\[
\boxed{
\text{“所有未检测到的 mismatch 都可以放心忽略”}
}
\]

这是好事。

更合理的世界模型本来就应该允许：

\[
\text{我现在还不知道，但值得去查}
\]

这种状态。

---

# 更关键的是 consequence 与 detectability 已经被成功解耦

修复 query RNG 混杂之后：

\[
C=C(\alpha)
\]

不再随着 excitation \(A\) 改变，而 detectability 明显：

\[
D=D(\alpha,A).
\]

于是你现在有了非常干净的因果分解：

\[
\boxed{
\begin{aligned}
\alpha &\rightarrow \text{mismatch consequence}\\
A &\rightarrow X_\phi\rightarrow\text{detectability}
\end{aligned}
}
\]

这个结构非常重要。

它说明主动 probe 的作用不是：

> 把 world 变得更容易。

而是：

> **改变观测的信息性，而不改变待诊断 mismatch 本身。**

这正是 active system identification 应有的实验结构。

---

# 小 \(\alpha\) 的结果也很有价值

报告里：

\[
|\alpha|\le0.09
\]

几乎都落在 tolerance 区。

这实际上验证了我们前面讨论的：

\[
\boxed{
\text{not every physical mismatch deserves revision}
}
\]

如果你为了这些 case 把 recall 从 0.5 提升到 1.0，实际可能只是：

- 增加 probing cost；
- 增加 false positive 风险；
- 增加 revision frequency；

而长期预测收益很小。

所以 detector 的目标不应该是：

\[
\max P(\text{detect}\mid\alpha\neq0)
\]

而应该更接近：

\[
\boxed{
\max
P(\text{detect}\mid C\ge C_{\mathrm{tol}})
}
\]

subject to：

\[
\text{probe cost},\quad
\text{false alarm},\quad
\text{revision risk}.
\]

这已经从 classification 变成 decision theory。

---

# 我认为现在仍然不应该开 RS1B

你的判断是对的。

现在 RS1B 如果研究所有 trigger：

\[
D_0>\tau
\]

仍然会混入很多：

\[
C<C_{\mathrm{tol}}
\]

的无关 revision。

真正应该进入 RS1B 的 population 已经被 RS1A.4 定义出来：

\[
\boxed{
\mathcal P_{\mathrm{rev}}
=
\{C\ge C_{\mathrm{tol}}\}
\cap
\{\mathrm{detect}\}
}
\]

即报告里的 `revise_worthy`。

这一点我建议永久冻结。

以后 H32 stability 研究回答的是：

> 对**确实值得修、且证据已经足够**的模型缺陷，revision 是否长期稳定？

这样 RS1B 才不会和 detector / tolerance 问题重新混杂。

---

# 但在 RS1B 之前，我建议还有一个更自然的阶段

现在最直接的新问题已经不是 detector：

\[
\boxed{
\text{Given a probe-state mismatch, should we spend action cost to excite it?}
}
\]

可以新开：

\[
\boxed{
R1\text{-RS1A.5 — Value-of-Epistemic-Excitation}
}
\]

它只研究 `probe` 区：

\[
C\ge C_{\mathrm{tol}},\quad D_0\le\tau.
\]

不研究 tolerance case，也不研究已经 detected 的 case。

---

## RS1A.5 的核心不是“更大 A 更好”

RS1A.3 已经证明了这件事。

真正问题应该是：

\[
\boxed{
\text{Is the information gained from excitation worth its intervention cost?}
}
\]

定义一个简单 VoI：

\[
V_{\mathrm{probe}}(a)
=
\mathbb E[
L_{\mathrm{decision}}
\mid
D,a
]
-
L_{\mathrm{probe}}(a).
\]

更具体一点：

\[
\boxed{
V(a)
=
P(\text{decision changes}\mid a)
\cdot
\Delta C_{\mathrm{decision}}
-
\lambda c(a)
}
\]

其中：

- \(P(\text{decision changes}\mid a)\)：这个 probe 是否有机会把 probe → revise-worthy；
- \(\Delta C_{\mathrm{decision}}\)：避免继续错误模型带来的收益；
- \(c(a)\)：动作代价、时间、扰动、安全代价。

只有：

\[
V(a)>0
\]

才 probe。

否则：

\[
\boxed{\text{继续容忍 / defer}}
\]

---

# 这样 APR-WM 的决策链会变得非常完整

现在已经不再是：

\[
\text{error}\rightarrow\text{revision}.
\]

而是：

\[
\boxed{
\text{mismatch}
\rightarrow
\begin{cases}
\text{tolerate},\\
\text{probe},\\
\text{revise-worthy}
\end{cases}
}
\]

其中 probe 再做：

\[
\boxed{
\text{probe}
\rightarrow
\text{estimate VoI}
\rightarrow
\begin{cases}
\text{acquire evidence},\\
\text{defer}
\end{cases}
}
\]

取得新 evidence 后：

\[
\text{probe}
\rightarrow
\text{revise-worthy}
\]

或者：

\[
\text{probe}
\rightarrow
\text{tolerate / unknown}.
\]

这已经非常接近你最初 APR-WM 的核心设想：

\[
\boxed{
\text{不知道什么}
\rightarrow
\text{决定是否值得知道}
\rightarrow
\text{主动取证}
}
\]

---

# 当前科学链可以更新成

\[
\boxed{
\begin{aligned}
RS1A &: \text{C2 requires support evidence}\\
RS1A.1 &: \text{support composition cannot solve C1-L}\\
RS1A.2 &: \text{C1-L information mainly lies in residual magnitude}\\
RS1A.3 &: \text{remaining misses are excitation-limited}\\
RS1A.4 &: \text{misses split into tolerable and consequential probe states}\\
RS1A.5 &: \textbf{next: when is epistemic excitation worth its cost?}
\end{aligned}
}
\]

然后才：

\[
RS1B:
\text{long-horizon stability of revise-worthy revisions}.
\]

所以这次 `GO=false` 不应该被理解成退步。它实际上把 binary inadequacy detector 推进成了一个更完整的 **epistemic control policy**：

\[
\boxed{
\text{tolerate}
\;\rightarrow\;
\text{probe if valuable}
\;\rightarrow\;
\text{revise if justified}
}
\]

这比单纯把 C1-L recall 做到 90% 更接近 APR-WM 真正有研究价值的方向。

---

**Cursor**

同意：GO 未过反而钉死了 probe 区。下一步冻结 **RS1A.5（VoI of epistemic excitation）**，只研究 `probe` 态，不开 RS1B。

**RS1A.5_GO = true**。报告：`REPORT/R1_RS1A5_REPORT.md`。

### 核心结果
在 **probe 区**（\(C\ge C_{\mathrm{tol}}\) 且未检出）上：

| \(A_b\to A_a\) | flip | \(\overline V\) |
|----------------|-----:|----------------:|
| 0.5 → 1.0 | 0% | **−0.0015**（白花钱） |
| 0.5 → **1.5** | **100%** | **+0.0012** ← \(A^\star\) |
| 0.5 → 2.0 | 100% | **−0.0014**（翻盘了但代价过大） |

\[
\boxed{A^\star=1.5A_0 \neq \max A}
\]

### 已永久冻结
- 三分区：tolerate / probe / revise_worthy  
- RS1B 人口：\(\mathcal P_{\mathrm{rev}}=\{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}\}\)  
- RS1B **仍锁**

### 决策链
\[
\text{mismatch}\to\{\text{tolerate},\text{probe},\text{revise}\}
\]
\[
\text{probe}\to V(a)\to\{\text{取证},\text{defer}\}
\]

这比把 C1-L recall 硬拉到 90% 更接近 APR-WM 的目标：**按认识价值决定是否容忍、取证、修正。**

---

**User**

这一步很关键。RS1A.5 把前面 RS1A.3/1A.4 的“**需要主动取证**”从概念判断推进成了一个真正的**决策规则**：

\[
\boxed{
\text{不是 probe 越强越好，而是存在最优 epistemic intervention}
}
\]

在当前成本定义下，

\[
A^\star=1.5A_0
\]

而不是最大激励 \(2A_0\)：两者都能 100% 把 probe case 翻成 revise-worthy，但 \(2A_0\) 的动作成本使净 VoI 变负；\(1.0A_0\) 又激励不足，完全不改变决策。

这说明 APR-WM 现在已经不只是“发现模型错了以后主动 probe”，而是开始具备：

\[
\boxed{
\text{choose whether to probe}
+
\text{choose how much to probe}
}
\]

的认识资源分配。

## 1. 这实际上闭合了 RS1A 的 epistemic decision loop

现在这条链已经相当完整：

\[
\text{mismatch}
\rightarrow
\begin{cases}
\text{tolerate}\\
\text{probe}\\
\text{revise-worthy}
\end{cases}
\]

其中 probe 状态进一步：

\[
\text{probe}
\xrightarrow{\operatorname{VoI}(a)}
\begin{cases}
\text{acquire evidence},&\max_a V(a)>0\\
\text{defer},&\max_a V(a)\le0.
\end{cases}
\]

而证据获取后可以发生：

\[
\text{probe}\rightarrow\text{revise-worthy}.
\]

报告中 \(0.5A_0\rightarrow1.5A_0\) 正是这种 transition 的直接实例。

因此最初比较粗糙的：

\[
\text{residual large}\Rightarrow\text{revision}
\]

已经演化成了：

\[
\boxed{
\text{consequence}
\rightarrow
\text{epistemic state}
\rightarrow
\text{value of information}
\rightarrow
\text{evidence acquisition}
\rightarrow
\text{revision eligibility}
}
\]

这更接近 APR-WM 最初的目标。

---

## 2. \(A^\star\neq A_{\max}\) 是这次最有价值的结果

如果结果只是：

\[
0.5<1.0<1.5<2.0
\]

激励越大越好，那么得到的其实只是：

> signal 太弱，那就使劲 probe。

但现在结果是：

\[
V(1.0)<0,
\qquad
V(1.5)>0,
\qquad
V(2.0)<0.
\]

形成了真正的内部最优点。

也就是说：

\[
\boxed{
\text{information gain has diminishing decision value while action cost keeps rising}
}
\]

于是 epistemic action 本身成为一个优化问题，而不是 binary 开关。

这对未来扩展到真实机器人非常重要，因为真实 probe 的成本不只是 torque：

\[
c(a)=
c_{\text{energy}}
+
c_{\text{time}}
+
c_{\text{task delay}}
+
c_{\text{safety}}
+
c_{\text{wear}}
+\cdots
\]

因此现在这个 toy cost：

\[
c(A)=\left(A/A_0\right)^2
\]

虽然很简单，但机制已经成立。

---

## 3. 还出现了一个“minimum sufficient experiment”原则

当前结果可以理解为：

\[
1.0A_0:
\quad\text{information insufficient}
\]

\[
1.5A_0:
\quad\text{information sufficient}
\]

\[
2.0A_0:
\quad\text{information redundant}.
\]

所以更一般的目标不是：

\[
\max_a I(M;D\mid a)
\]

而应该接近：

\[
\boxed{
a^\star
=
\arg\max_a
\left[
\text{decision value}(a)-\lambda c(a)
\right]
}
\]

或者更进一步：

\[
\boxed{
\text{find the cheapest experiment sufficient to change a consequential decision}
}
\]

这和纯 information-gain probing 有一个很重要的区别。

IG 可能偏好：

\[
2A_0
\]

因为它带来的 raw information 更多。

APR-WM 现在关心的是：

> 多出来的信息有没有改变最终模型决策？

如果 \(1.5A_0\) 已经足够确定 revision-worthy，那么 \(2A_0\) 的额外信息就没有相同的边际价值。

---

## 4. 这也修正了 V2 的 active identification 思想

V2 主要在做：

\[
\max_a IG(\theta,M;a).
\]

到了 RS1A.5，可以把这一思想升级成：

\[
\boxed{
\max_a
\text{Decision-VoI}(a)
}
\]

不是为了最大化“我知道多少”，而是最大化：

\[
\boxed{
\text{新信息对后续正确决策的价值}
}
\]

这是明显更一般的 formulation。

可以写成：

\[
V(a)
=
\mathbb E_{y\sim p(y|a,D)}
\left[
\min_d L(d\mid D)
-
\min_d L(d\mid D,y,a)
\right]
-c(a).
\]

这里：

- \(a\)：probe；
- \(y\)：probe 后的 observation；
- \(d\)：tolerate / defer / revise / fallback 等决策；
- \(L\)：决策损失。

这比现在实验里的：

\[
V=\mathrm{flip}\cdot C-\lambda c(A)
\]

更一般。

当前公式可以理解成它的最小可验证版本。

---

## 5. 三分区现在可以永久冻结是合理的

目前已经形成：

\[
\boxed{
\begin{aligned}
\mathcal T
&=
\{C<C_{\mathrm{tol}}\}\\
\mathcal P
&=
\{C\ge C_{\mathrm{tol}}\}
\cap
\{\mathrm{detect}=0\}\\
\mathcal R
&=
\{C\ge C_{\mathrm{tol}}\}
\cap
\{\mathrm{detect}=1\}
\end{aligned}
}
\]

分别对应：

- \(\mathcal T\)：没必要知道得更精确；
- \(\mathcal P\)：值得知道，但证据不足；
- \(\mathcal R\)：值得修，而且已经有足够 evidence。

然后：

\[
\mathcal P
\xrightarrow{V(a)>0}
\mathcal R
\]

或者：

\[
\mathcal P
\xrightarrow{\max V\le0}
\text{defer}.
\]

这比单纯的：

\[
\text{adequate / inadequate}
\]

丰富很多。

---

# 6. 现在其实具备了进入 RS1B 的科学条件

这一点我会稍微修改之前的判断。

RS1A.4 之后不应该开 RS1B，是因为我们还不知道 probe 状态应该怎么处理。

现在 RS1A.5 已经回答：

\[
\boxed{
\text{probe state can be actively resolved according to positive VoI}
}
\]

因此 detector / epistemic-allocation 这一侧已经足够闭合。

所以现在可以：

\[
\boxed{
\text{解锁 RS1B 的 preregistration / development}
}
\]

但仍然不要直接把原 RS1 的 71 accepted revisions 拿回来重新分析。

RS1B 必须使用已经永久冻结的人口：

\[
\boxed{
\mathcal P_{\mathrm{rev}}
=
\{C\ge C_{\mathrm{tol}}\}
\cap
\{\mathrm{detect}=1\}.
}
\]

报告也明确把它冻结成未来 RS1B 的唯一 intake。

---

# 7. RS1B 的问题现在应该重新表述

不应该再问：

> 为什么 accepted revisions H32 不稳定？

因为“accepted”已经是旧 pipeline 的定义。

应该问：

\[
\boxed{
\text{Among consequential and epistemically justified revisions,
what dynamics-level conditions predict long-horizon stability?}
}
\]

即：

\[
\mathcal P_{\mathrm{rev}}
\rightarrow
\text{candidate revision}
\rightarrow
\boxed{\text{long-horizon admissibility}}
\rightarrow
\text{install}.
\]

这和 RS1A 负责的问题完全正交：

### RS1A

\[
\boxed{
\text{Should I spend epistemic effort and consider revision?}
}
\]

### RS1B

\[
\boxed{
\text{Given that revision is justified, is this revised vector field safe over long horizons?}
}
\]

这时再研究：

- effective damping；
- energy production；
- Jacobian/local expansion；
- contraction；
- support distance；
- parameter displacement；

就不会被大量 tolerate/probe case 污染。

---

# 8. 整个 APR-WM 现在已经形成三个不同“预算”

这是我认为当前项目最重要的架构性变化。

最初只有：

\[
\text{residual compute budget}.
\]

后来有：

\[
\text{probe budget}.
\]

现在可以清楚区分：

\[
\boxed{
\begin{aligned}
B_{\mathrm{compute}}
&:\text{是否调用 neural residual}\\
B_{\mathrm{epistemic}}
&:\text{是否主动获取更多信息}\\
B_{\mathrm{revision}}
&:\text{是否承担改变 persistent world model 的风险}
\end{aligned}
}
\]

对应的决策越来越清楚：

\[
\text{cheap prediction}
\rightarrow
\text{residual if needed}
\rightarrow
\text{probe if worth knowing}
\rightarrow
\text{revise only if justified and safe}.
\]

这已经超出了“physics + residual hybrid model”本身。

更准确的 APR-WM 定位开始变成：

\[
\boxed{
\text{a world model that allocates predictive,
epistemic, and structural complexity adaptively}
}
\]

---

## 当前科学链可以冻结为

\[
\boxed{
\begin{aligned}
RS1A &: \text{typed inadequacy evidence}\\
RS1A.1 &: \text{support fusion cannot rescue weak known signal}\\
RS1A.2 &: \text{known weak signal lives mainly in magnitude}\\
RS1A.3 &: \text{detectability is excitation-limited}\\
RS1A.4 &: \text{misses split into tolerate and probe}\\
RS1A.5 &: \textbf{probe has an optimal positive-VoI intensity}\\
RS1B &: \textbf{next: long-horizon stability of revise-worthy revisions}
\end{aligned}
}
\]

因此，我认为 **RS1A 这一大阶段到这里可以冻结**。

它最终没有把 C1-L recall 强行做到 90%——这是好事。它得到的是更强的东西：

\[
\boxed{
\text{Don't ask whether every mismatch can be detected. Ask whether it is worth detecting, and if so, what is the cheapest informative intervention.}
}
\]

中文就是：

> **不是所有误差都值得发现；对于值得发现但当前证据不足的误差，应选择净认识价值最高的取证动作，而不是盲目增加激励。**

下一阶段才适合正式进入 **RS1B：revise-worthy population 上的 long-horizon dynamics admissibility**。

---

**Cursor**

同意：RS1A 阶段冻结。下一步正式解锁 **RS1B prereg**（仅 `revise_worthy` 人口），并写清与 RS1A 正交的科学问题。

**RS1A 阶段已冻结；RS1B prereg 已解锁开发。**

### 冻结文档
- `REPORT/R1_RS1A_STAGE_FREEZE.md` — 整段 epistemic loop 结案  
- `REPORT/R1_RS1B_PREREG.md` — 正式开发版（不再是 locked draft）

### RS1A 结论文句
> 不是所有误差都值得发现；对值得发现但证据不足的误差，选净认识价值最高的取证动作，而不是盲目加大激励。

三分预算已固化：\(B_{\mathrm{compute}}\) / \(B_{\mathrm{epistemic}}\) / \(B_{\mathrm{revision}}\)。

### RS1B（与 RS1A 正交）
**问题：** 在 consequential 且证据充分的 revision 上，什么动力学条件预测长期稳定？

**唯一人口：**
\[
\mathcal P_{\mathrm{rev}}=\{C\ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=1\}
\]

**诊断轴：** effective damping · \(P_r\) · Jacobian expansion excess · support exit  
**禁止：** 看完 H32 再回头改 detector / VoI / \(C_{\mathrm{tol}}\)  
**不用：** 原始 RS1 那 71 条 unfiltered accepts

正式跑 RS1B 前还需把 prereg 里的 GO 数字最终钉死；需要的话我可以下一步直接实现 `r1-rs1b` 骨架并做 plumbing smoke。

---

**User**

好

---

**Cursor**

开始实现 RS1B 骨架并跑 plumbing smoke。

改用更精简的实现，分文件写入。


**Cursor**

**RS1B plumbing 已开发并通过；正式 confirmatory matrix 尚未运行。**

### 已冻结
- REPORT/R1_RS1B_PREREG.md：正式矩阵 5 × 3 × 2 = 30 episodes
- intake 只读 RS1A.5 frozen policy：P_rev = {C ≥ C_tol} ∩ {detect = 1}
- 48 个 dynamics diagnostic states、finite-difference ε = 10⁻⁴
- hard gate：effective damping / passivity；H32 在 accept freeze 后盲评
- GO：非零 accept、passivity violation = 0、seed-mean H32 gain > 0、stable rate ≥ 0.90

### 实现
- aprwm_v0/r1_rs1b.py：intake → frozen R0.6 → diagnostics → blind H32
- aprwm_v0/r1_rs1_backend.py：state-dependent revision rollout / instantaneous vector field
- CLI：r1-rs1b
- HDF5：raw_truth / learner_visible / decision / diagnostics / evaluation
- tests/test_r1_rs1b.py

### 预注册 smoke
seed = 9011，α = −0.24，A = 1.5 A₀。

| 项目 | 结果 |
|---|---:|
| revise_worthy intake | true |
| selected operator | abs_v_v |
| α̂ | −0.25464 |
| short utility | +0.000521 |
| effective damping min | +0.001341 |
| passivity violation | false |
| expansion excess max | −1.58 × 10⁻⁸ |
| support exit | 0 |
| H32 gain | **+0.001618** |
| H32 stable | true |

产物：runs/r1_rs1b/smoke/；报告：REPORT/R1_RS1B_SMOKE_REPORT.md。

完整测试：**101 passed**。该 smoke 只证明 plumbing 闭合，因此 RS1B_GO=false / 未评估，RS1C 与 RS2 继续锁定。下一步只能按 frozen prereg 跑 30-episode held-out formal，不得根据 smoke 修改 gate。
