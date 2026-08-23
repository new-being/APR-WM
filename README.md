# APR-WM V0：交互级 Physics–Residual Competition

这是面向 RTX 5070 8GB 的第一版可运行研究基线。V0 只回答一个问题：

> 在对象交互层面，自适应 gate 能否只在显式物理失效时调用 neural residual，并在精度、OOD、计算量与长时 rollout 稳定性之间取得更好的折中？

它刻意不包含视觉、slot、在线系统辨识、语言、动作先验或 MPC。输入直接使用 simulator ground-truth object state。

## V0 实验定义

每个对象状态为：

```text
[x, y, vx, vy, radius, mass, friction, material]
```

合成 2D 圆盘世界包含两种接触：

- 简单接触：显式 spring-damper physics 可以准确解释；
- 复杂接触：增加非线性法向响应和速度相关的切向黏滑，显式 expert 故意不知道这部分。

训练和评估使用 observed transitions；没有用模型自己的 counterfactual rollout 反向监督模型。OOD 同时改变物体数量、质量/摩擦范围、复杂材料比例和动作幅度。

四个 baseline 使用完全相同的数据：

1. `physics`：纯显式物理；
2. `neural`：纯 neural interaction GNN；
3. `residual`：physics + always-on residual；
4. `adaptive`：physics + interaction-level learned gate + residual 使用成本。

`adaptive` 先用 always-on residual warm-up，让 expert 学会物理模型缺失项；廉价 router 同时蒸馏 residual expert 的修正幅值需求（不使用 simulator 的材料/复杂度标签）。随后在每个世界的接触边上按 router 得分执行 top-budget straight-through 二值 gate，使训练时的前向计算与实际 sparse routing 一致，避免 tiny soft gate 与无限放大的 residual 相互抵消。默认预算为 60% 的接触边。评估同时报告 soft rollout 和预算约束的 sparse hard-routing rollout，后者只为被选中的 edge 执行 residual MLP。

## 本机安装

RTX 50 系列应使用包含 Blackwell 架构支持的 CUDA 12.8 或更新版 PyTorch wheel。当前机器已经有可直接使用的环境：

```bash
/home/dong/miniconda3/envs/EfficientWAM/bin/python -m aprwm_v0 doctor \
  --config configs/rtx5070_8gb.json
```

若需要独立环境，可使用系统默认 Python 3.13：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
```

先检查 wheel 是否原生包含 GPU 架构，并查看粗略显存预算：

```bash
python -m aprwm_v0 doctor --config configs/rtx5070_8gb.json
```

输出中的 `native_arch_supported` 应为 `true`。若为 `false`，不要继续训练，需换用更新的官方 CUDA wheel。

## 运行

快速验证完整链路：

```bash
python -m pytest
python -m aprwm_v0 suite --config configs/smoke.json
```

在 RTX 5070 8GB 上依次训练四个 baseline：

```bash
python -m aprwm_v0 suite --config configs/rtx5070_8gb.json
```

也可以只训练核心模型：

```bash
python -m aprwm_v0 train \
  --config configs/rtx5070_8gb.json \
  --model adaptive
```

重新评估 checkpoint：

```bash
python -m aprwm_v0 evaluate \
  --checkpoint runs/v0_rtx5070/adaptive/checkpoint.pt
```

## 输出与判据

结果写入 `runs/v0_rtx5070/`：

- 每个模型的 `checkpoint.pt`、`metrics.json`、`train_history.json`；
- 四模型横向对比 `suite_summary.csv`；
- 完整结果 `suite_metrics.json`。

关键指标：

- `one_step_*_rmse`：teacher-forced 单步预测；
- `rollout_state_rmse`：soft gate 自回归 rollout；
- `hard_rollout_state_rmse`：实际稀疏路由 rollout；
- `routing_auroc`：gate 区分简单/复杂接触的能力（只用于评估，不作监督）；
- `active_residual_edge_fraction`：预算实际选中的 contact edge 比例；
- `active_graph_edge_fraction`：实际调用 residual 的全部图 edge 比例；
- `learned_compute_fraction`：router + 被调用 residual expert 的解析 FLOPs proxy；
- `rollout_stable_fraction`：有限值且速度不爆炸的序列比例。

V0 成功的最低判据是：`adaptive` 明显优于 pure physics，在接近 always-on residual 的 ID/OOD rollout 误差下满足 `residual_budget_fraction`，且 `routing_auroc` 显著高于 0.5。精度不足时提高预算或延长 warm-up；算量过高时降低预算。

## 本机首轮结果

RTX 5070 Laptop 8GB、`torch 2.11.0+cu128`、seed 7、每个可学习模型 4,000 steps：

| 模型 | ID hard rollout RMSE | OOD hard rollout RMSE | OOD router AUROC | learned compute proxy |
|---|---:|---:|---:|---:|
| Pure Physics | 0.03938 | 0.06828 | 0.500 | 0% |
| Pure Neural GNN | 0.02761 | 0.06141 | 0.500 | 100% |
| Always-on Residual | 0.01146 | 0.03155 | 0.500 | 100% |
| Adaptive Budget Router | 0.01200 | 0.03175 | 0.962 | 11.0% |

所有 rollout stability 均为 1.0，记录到的最高 CUDA allocated memory 为 193 MiB。该结果只用于确认 V0 机制和工程链路成立；它是单 seed、合成环境和解析 FLOPs proxy，不能当成最终论文结论。

## V0.5 科学验证

V0.5 已补充 5 seeds、Random/heuristic/learned/oracle Pareto、固定 material 的连续复杂度、H=1–50 rollout、真正 conditional execution 的 CUDA latency，以及 router ablation。完整结论见 [REPORT/REP/R0/V05_REPORT.md](REPORT/REP/R0/V05_REPORT.md)。

最重要的修正是：learned router 的排序信号成立，但当前环境中 contact edge 本身只占全图约 5%，廉价 contact filtering 已能获得大部分解析计算节省；同时小 residual MLP 的 sparse GPU 路径比 dense 路径更慢。因此目前不能声称 89% 实际计算或延迟下降。

## V0.6 Compute-valid Adaptive Routing

V0.6 将统计稀疏性与真实计算收益分开验证。合成 benchmark 中 interaction density 与 conditional residual necessity 独立变化；同一 interaction 是否进入非线性 residual regime 只由当前 extension、relative velocity 与 friction 决定，不提供 object/edge ID。公平基线是已经执行解析 Stage-0 filtering 的 `contact_packed`，learned router 只在 active edge 内决定是否调用 residual expert。

```bash
python -m aprwm_v0 v06 \
  --output runs/v06/core \
  --device cuda \
  --full-matrix
```

矩阵覆盖 interaction density `20/50/80/100%`、conditional residual necessity `5/10/20/40%`、近似真实 FLOPs 为 `1×/2×/4×/8×` 的 residual expert、`linear/tiny/current` 三种 router，以及 batch `1/16/128/512`。`hard_sparse` 的 CUDA 计时包含 router、top-k 与 residual-row packing；Stage-0 active-edge packing 是两种方法的共享前置成本，不计入二者的增量比较。完整结论及局限见 [REPORT/REP/R0/V06_REPORT.md](REPORT/REP/R0/V06_REPORT.md)。

## V0.7 Utility-supervised Routing

V0.7 冻结 residual expert 后，以完全相同的 router、loss 和 10% top-budget 比较 `magnitude`、binary `necessity` 与 task-weighted `utility` supervision，并用 privileged oracle 给出分配上界。正式实验使用 5 seeds，只测 B16/8×、B128/8×、B512/1×、B512/8× 四个 slower/crossover/faster 代表点。CUDA 延迟采用逐样本交替 baseline/adaptive 的 paired timing，避免 GPU boost 和执行顺序污染边界结论。

```bash
python -m aprwm_v0 v07 \
  --output runs/v07/core \
  --device cuda \
  --seeds 13 23 33 43 53
```

8× expert 下，utility router 捕获 `96.78%` oracle utility，显著高于 magnitude 的 `94.24%`；但 1× expert 上差异的置信区间跨零，因此 utility supervision 不是无条件占优。完整配对统计、硬件边界与 cost-aware objective 的等价性见 [REPORT/REP/R0/V07_REPORT.md](REPORT/REP/R0/V07_REPORT.md)。

下一阶段固定为 unknown-physics V1：先隐藏 `(k,c)` 并维护参数 posterior，显式分离 model-class inadequacy 与 parameter-estimation error；实施边界和成功判据见 [REPORT/REG/R0/V1_PLAN.md](REPORT/REG/R0/V1_PLAN.md)。

## V1 Unknown-physics Posterior

V1 最小实验已实现：每条 trajectory 隐藏 spring stiffness/damping `(k,c)`，由 4/8/16/32 条 context interaction 构造 Bayesian parameter posterior；结构非线性由独立 latent model class 控制。Residual expert 只监督 oracle-parameter 下仍存在的 structural correction，从训练目标上禁止它直接吸收 parameter error。

```bash
python -m aprwm_v0 v1 \
  --output runs/v1/core \
  --device cuda \
  --seeds 13 23 33 43 53
```

五 seed 结果表明：adequate physics class 下 posterior calibration 正常，90% 区间覆盖率为 `90.55%`；存在结构失配时覆盖率跌至 `17.40%`，说明 misspecified posterior 会变得严重过度自信。Posterior-full router 将严格 residual misuse rate 从 state-only 的 `10.09%` 降至 `0.80%`，并捕获 `95.09%` structural utility；但由于 parameter bias 与 structural error 强烈抵消，它在 estimated physics 上的短期 force RMSE 反而更高。完整分析见 [REPORT/REP/R0/V1_REPORT.md](REPORT/REP/R0/V1_REPORT.md)。

## V2 Misspecification-aware Epistemic Control

V2 将 environment action `a_t` 与 internal compute decision `r_t` 分开。联合粒子 belief 维护 `p(θ,M|D)`；环境动作从 passive、velocity probe、amplitude probe 中选择，计算策略再依据 model-class belief、净预测效用和 residual cost 决定是否调用 structural residual。

```bash
python -m aprwm_v0 v2 \
  --output runs/v2/core \
  --device cuda \
  --seeds 13 23 33 43 53
```

低幅数据构造出 compensation trap：强制线性模型在 inadequate class 中学到 `+0.373` 的 pseudo-true stiffness bias，ID RMSE 仅 `0.0256`，intervention RMSE 却升至 `0.2194`。Misspecification-aware joint-IG 相对 parameter-IG 将 model AUROC 从 `0.9155` 提至 `0.9632`，intervention RMSE 从 `0.1949` 降至 `0.1740`，同时平均少用 `0.186` 次 probe；与固定两次 amplitude probe、但仍需推断模型类别的 diagnostic oracle 在 intervention RMSE 上无显著差异。直接提供真实 model class 的上界达到 `0.0491`，说明主要剩余瓶颈是类别判定，而非诊断动作选择。完整方法、配对统计和边界见 [REPORT/REP/R0/V2_REPORT.md](REPORT/REP/R0/V2_REPORT.md)。

## V3-min Tangent-triggered Model Revision

V3-min 初始只激活 linear physics。它在多观测窗口上把 residual 投影为 parameter-tangent 分量与正交分量，用持续的正交证据决定保持线性、激活 dormant cubic hypothesis，或在候选结构无法通过 held-out 检验时声明 `unknown`。第三种真值是候选扩展中不存在的 velocity-coupled discrepancy，用于测试开放集拒识。

```bash
python -m aprwm_v0 v3 \
  --output runs/v3/core \
  --device cuda \
  --seeds 101 111 121 131 141
```

五个未参与阈值开发的 seeds 上，tangent evidence 相对 residual magnitude 将 structural AUROC 从 `0.7709` 提至 `0.9957`，adequate false revision 从 `65.44%` 降至 `0.92%`，三态 model accuracy 从 `70.22%` 提至 `89.61%`；intervention RMSE 从 `0.1751` 降至 `0.1641`。但 unseen 类即使有 `89.08%` 被正确判为 unknown，其 RMSE 仍为 `0.2690`，远高于 oracle-family 的 `0.0325`。因此 V3-min 支持 tangent-triggered revision，同时明确证明 detection 不等于 model discovery。完整结果见 [REPORT/REP/R0/V3_REPORT.md](REPORT/REP/R0/V3_REPORT.md)。

## V4-min Residual-guided Active Operator Discovery

V4-min 将 `unknown` 后续处理拆成三个独立阶段：用 tangent-orthogonal residual 在八个结构算子中生成 top-3 proposal；执行候选预测分歧最大的诊断动作；最后通过独立 held-out 数据与复杂度成本决定是否永久接受单个 operator。Cubic 与 velocity-coupled 真值位于字典内，另有一种 thresholded oscillatory dynamics 完全不在字典中。

```bash
python -m aprwm_v0 v4 \
  --output runs/v4/core \
  --device cuda \
  --seeds 201 211 221 231 241
```

五个 held-out seeds 上，orthogonal proposal 的 top-3 recall 为 `98.99%`，主动诊断将 exact operator recovery 从 passive 的 `43.66%` 提至 `60.45%`。Oracle proposal 只额外提升 `0.38` 个百分点，而 oracle selection 达到 `100%`，说明当前瓶颈已经从 proposal 转移到 candidate discrimination 与 held-out acceptance。主动发现还将 residual fallback 从 `65.24%` 降至 `25.76%`，保持 outside-library rejection `90.80%`、adequate incorrect expansion `0%`。完整方法、oracle gap 与边界见 [REPORT/REP/R0/V4_REPORT.md](REPORT/REP/R0/V4_REPORT.md)。

## V5-min Noisy Sequential Hypothesis Discrimination

V5-min 冻结 V4 的八算子字典与 top-3 proposal，在受控 candidate-availability 条件下比较 raw disagreement、uncertainty normalization、posterior-weighted separation、固定证据预算、sequential stopping、LCB acceptance 与两个 oracle。正式矩阵覆盖 5 seeds 和观测噪声 `0/0.025/0.05/0.10/0.15`。

```bash
python -m aprwm_v0 v5 \
  --output runs/v5/core \
  --device cuda \
  --seeds 301 311 321 331 341
```

噪声 `0.05` 下，posterior-weighted 两次 probe 相对 normalized 两次 probe 将 selection 提高 `4.29` 个百分点、exact recovery 提高 `3.72` 个百分点；第三次 probe 将 exact recovery 进一步提高到 `79.42%`。Sequential 版本把平均 probe 从 `2.212` 降至 `0.906`，但 exact recovery 降至 `71.07%`，因此是 evidence-cost Pareto 点而非无条件优胜。噪声 `0.10` 时 weighted-3 selection 仍有 `73.06%`，但正确选择后的 acceptance 只剩 `30.53%`，证明高噪声瓶颈已经转移到 validation power。LCB 没有降低本已接近零的 adequate false expansion，却进一步损失 recovery。完整受控条件、配对区间、open-set 分解与局限见 [REPORT/REP/R0/V5_REPORT.md](REPORT/REP/R0/V5_REPORT.md)。

## V6 Adaptive Evidence Acquisition

V6 冻结 V5 的 operator library、受控 candidate coverage 和 posterior-weighted 三 probe selector，只研究模型 revision 的验证阶段。Acceptance 同时检验新结构是否优于旧显式 physics、是否优于 selection runner-up，并从 observed MSE 中扣除已知 observation variance；residual 仅作为拒绝或等待期间的预测缓冲。

```bash
python -m aprwm_v0 v6 \
  --output runs/v6/core \
  --device cuda \
  --seeds 401 411 421 431 441
```

V6 发现 V5 在 `σ=0.10` 的 acceptance collapse 部分来自固定 RMSE 阈值低于噪声地板。校准后，fixed-8 exact recovery 从 `3.20%` 恢复到 `28.42%`，但 false revision 同时升至 `3.41%`。双重 sequential BF-style 验证在全部噪声下的正式 mean false revision 均低于 `1%`；`σ=0.05` 时用 `16.17` 个平均 validation samples 达到 `45.20%` recovery，与 fixed-32 的 `44.99%` 无显著差异。`σ=0.10` 时它以 recovery 降至 `18.35%` 为代价，把 false revision 从 fixed-32 的 `1.66%` 降至 `0.47%`。这确立的是 validation cost、power 与 revision safety 的 Pareto 前沿，而非免费的 acceptance 提升。完整方法、配对统计和有效性边界见 [REPORT/REP/R0/V6_REPORT.md](REPORT/REP/R0/V6_REPORT.md)。

## 研究边界

这是验证机制的受控 toy environment，不是最终具身系统。V1–V6 已逐步加入 unknown-physics posterior、主动辨识、开放集 revision、operator discovery、noisy discrimination 与 adaptive validation，但仍未覆盖端到端视觉、长时闭环控制、真实机器人数据或能耗测量。V5/V6 的结构恢复还条件于受控 trigger 与 candidate availability；V6 的 BF 是 plug-in surrogate，也不提供理论上的 anytime-valid false-revision 保证。

## V6R0 Realism-bridge Preflight

V6R0 已加入 oracle-state SAPIEN 铰链后端、R0-C 三种 regime、counterfactual branch manifest、`H=1/4/8/16` rollout、RoboTwin capability gate 和 residual assimilation 指标。当前三开发种子运行被明确标记为 `R0-C headless SAPIEN hinge proxy`，没有冒充官方 RoboTwin Open Laptop。

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r0-doctor \
  --robotwin-repo /home/dong/Projects/RoboTwin

/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r0-smoke \
  --output runs/v6r0/smoke \
  --device cpu \
  --seeds 13 23 33 \
  --episodes-per-regime 24
```

预检没有通过扩大实验的门槛：pure physics 在全部 aggregate regime/horizon 上优于 no-revision residual 与 frozen V6；C1 exact recovery 仅 `1.39%`，C0 false revision 为 `2.78%`。原因是冻结的二维接口用 `[torque, qdot] -> qddot`，不能把 C0 表达成结构正确的 articulated physics，导致原生动力学残差与注入 C1 算子混叠。下一步必须先改成保留 `[q,qdot,torque]` 的 generalized-force residual，并验证 C0 residual 接近噪声地板；在此之前不进入正式 RoboTwin/R0-N/RGB。完整诊断、验收表和资产阻塞见 [REPORT/REP/R0/V6R0_PREFLIGHT_REPORT.md](REPORT/REP/R0/V6R0_PREFLIGHT_REPORT.md)。

## V6R0.1 Generalized-force Interface Validation

R0.1 保留 oracle local state `[q,qdot,torque]`，把残差改为

```text
r_tau = M(q) qdd + h(q,qdot) - tau_known
```

并显式清零 SAPIEN builder 后仍保留的 `0.05` joint friction 及 link damping。命令严格先跑 C0；三个开发 seed 全部通过 closure gate 后，才自动执行最小 C1 force-operator recovery，不运行完整 V6、C2 或 rollout：

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r01-closure \
  --output runs/v6r01/closure \
  --device cpu \
  --seeds 13 23 33 \
  --episodes 24 \
  --samples-per-episode 48
```

C0 clean force RMSE 为 `1.248e-7`，noise variance ratio 为 `1.041`，最大 `q/qdot` 相关性为 `0.031`，结构触发率为 `0%`。随后 C1 tangent AUROC、drag top-1/top-3 recovery 均为 `1.0`，估计系数 `-0.11984` 对应真值 `-0.12`，拟合后 residual 回到 `0.00198` 的噪声地板。这支持 prior no-go 的 interface diagnosis，但尚不支持 full-V6 false-revision、rollout 或 RoboTwin claim。完整边界见 [REPORT/REP/R0/V6R01_REPORT.md](REPORT/REP/R0/V6R01_REPORT.md)。

## V6R0.2 Frozen V6 Force-space Reintegration

R0.2 在 C0/C1 中恢复冻结的 tangent trigger、V4 operator proposal、V5 posterior-weighted selection、V6 dual sequential acceptance 及 residual assimilation。参数切空间使用 `J_theta`，结构算子独立使用 `[q,qdot]`，避免重新混淆物理坐标。

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r02 \
  --output runs/v6r02/trigger_gated \
  --device cpu \
  --seeds 13 23 33 \
  --episodes-per-regime 24
```

三个开发 seed 上，C0 false revision 为 `0%`，最大 aggregate V6–physics rollout gap 为 `1.88e-4`，通过 `0.002` non-inferiority margin。C1 proposal、selection、acceptance、exact recovery 与 assimilation 均为 `100%`；相对 no-revision 的 H1/H4/H8/H16 RMSE 改善分别为 `0.00204/0.00678/0.01170/0.02565`。结果支持完整 self-revision loop 在 force space 中稳定运行，但还不构成严格 `<1%` population safety 保证，也未覆盖 C2、joint-limit contact、R0-N 或官方 RoboTwin。完整诊断见 [REPORT/REP/R0/V6R02_REPORT.md](REPORT/REP/R0/V6R02_REPORT.md)。

## V6R0.3 Formal Held-out SAPIEN Bridge

R0.3 冻结 R0.2 全部机制和阈值，在 `1001/1011/1021/1031/1041` 五个 held-out seeds 上加入 history-dependent C2 与保留 PhysX 原生摩擦/阻尼的 R0-N。每个 rollout window 自动记录 joint-limit validity；超出当前模型支持域的 window 不进入 RMSE，但保留在 coverage 指标中。

```bash
/home/dong/miniconda3/envs/RoboTwin/bin/python -m aprwm_v0 v6r03 \
  --output runs/v6r03/formal \
  --device cpu \
  --seeds 1001 1011 1021 1031 1041 \
  --episodes-per-regime 24
```

正式结果为 overall no-go。C0 non-inferiority/false revision、C1 rollout/assimilation、C2 unknown rejection 与 validity coverage 均过 gate；C2 unknown rejection 为 `95.83%`，但无 history 输入的 fallback 在 H16 比 physics 更差。R0-N 中 `21.67%` episode 接受 revision，独立 one-step validation gain 为 `+0.00492`，但 H16 V6 RMSE 从 no-revision 的 `0.333` 恶化到 `2.032`，accepted revision 的 H16 gain 为 `-2.91`。这证明当前瓶颈是 rollout-safe acceptance 和 fallback utility，而不是 force-space closure。完整结果见 [REPORT/REP/R0/V6R03_REPORT.md](REPORT/REP/R0/V6R03_REPORT.md)。

## V6R0.4 Dynamical Acceptance + History-aware Fallback

R0.4 拆为两个独立实验。R0.4-A 的首批消融显示 `H={2,4,8}` short-rollout gate 可将 native H16 从 force-only 的 `1.948` 降至 `0.337`；但在策略固定后的全新确认 seeds 上，H16 为 `0.639`，仍差于 no-revision 的 `0.369`，accepted stability 也只有 `98.125%`，因此 revision gate 仍是 no-go。R0.4-B 则通过：在 delayed-torque C2 中，三步 action history 将 H16 从 physics 的 `0.01666` 降至 `0.00243`，而 12 维 memoryless fallback 恶化至 `0.04995`。这支持 state-closure 诊断，但不改变整体 RoboTwin no-go。完整协议、选择/确认分离及结果见 [REPORT/REP/R0/V6R04_REPORT.md](REPORT/REP/R0/V6R04_REPORT.md)。

## V6R0.5 Dynamical-safety Diagnostics

R0.5 不再修改 acceptance，而是回放 R0.4 revision episodes 并用 H16 outcome 做无阈值 AUROC 排序。one-step force-accepted revisions 中，selection/confirmation 分别有 `6/20` 与 `5/19` unsafe；正 revision 功率、负有效阻尼、Jacobian 谱半径的 AUROC 分别为 `0.988/1.000`、`0.917/1.000`、`0.833/1.000`，Mahalanobis 为 `0.738/0.871`，Euclidean 参数位移与 force gain 接近无效。当前失败主要来自正系数 `abs_v_v` 反阻尼 revision，但 operator identity 与安全标签高度混杂；严格 H2/H4/H8-accepted 子集也只有一个 H16 failure。因此结果支持 passivity-constrained R0.6 假设，却还不构成通用安全判据。完整边界见 [REPORT/REP/R0/V6R05_REPORT.md](REPORT/REP/R0/V6R05_REPORT.md)。

## V6R0.5P Operator-controlled Passivity Falsification

R0.5P 固定 `abs_v_v` operator，并在五个全新 seeds 上对称干预 `alpha=+-{0.15,0.30,0.45}`。冻结参数时 active/dissipative H16 instability 为 `95.83%/0%`，条件重拟合参数后为 `87.78%/0%`；power-violation fraction 对 instability 的 AUROC 为 `0.990/0.967`，且每个 seed 都保持 `>0.95`。在重拟合层，76 个 active 与 80 个 dissipative candidate 均通过 H2/H4/H8；其 H16 instability 分别为 `57.89%/0%`。但 dissipative candidate 仍常因过度耗散而产生负 utility，因此 passivity 是该算子族的稳定性结构条件，不是准确性或 acceptance 的充分条件。未实现 R0.6。完整结果见 [REPORT/REP/R0/V6R05P_REPORT.md](REPORT/REP/R0/V6R05P_REPORT.md)。

## V6R0.6 Passivity-feasible + Utility-accepted Revision

R0.6 将 acceptance 拆为物理可行性与预测效用两层：耗散算子在 `alpha<=0` 的结构保持空间中进行弱 posterior-regularized MAP 拟合，再用等权 H2/H4/H8 utility 决定是否持久化。在未使用过的 `7001–7041` seeds 上，native H16 从 no-revision 的 `0.32833` 降至 `0.23468`，accepted stability `100%`，接受率 `85%`，useful recall/precision 为 `93.68%/87.25%`；C0 false revision `0%`，C1 exact recovery `100%`。原始 short-rollout baseline 再次崩到 `0.99785`。全部预注册 gate 通过，因此轻量 SAPIEN realism gate 可以解除；但 utility 层并未优于 passivity-only (`0.23468` vs `0.23449`)，且 overdamping ratio 仍为 `1.98`。这使 RoboTwin task 实验变得合理，并不等于已经在 RoboTwin 成立。完整边界见 [REPORT/REP/R0/V6R06_REPORT.md](REPORT/REP/R0/V6R06_REPORT.md)。

## R1-MJ/RS MuJoCo–robosuite realism bridge

R1-MS0（ManiSkill Drawer）保留为 `infrastructure_block`，不算 scientific
no-go。下一阶段改为：

```text
R1-MJ0 → R1-RS0 → R1-RS1 → R1-RS2
```

基础两级已通过：

```bash
# 1) native MuJoCo 单 hinge 力空间闭合（9 cell）
.venv-robosuite/bin/python -m aprwm_v0 r1-mj0 \
  --output runs/r1_mj0/closure

# 2) MJ0 通过后才解锁 robosuite Door Mode-A C0（3 profiles × 3 seeds）
.venv-robosuite/bin/python -m aprwm_v0 r1-rs0 \
  --output runs/r1_rs0/c0 \
  --mj0-summary runs/r1_mj0/closure/summary.json

# 3) RS1B plumbing smoke：只接收 frozen revise_worthy population
.venv-robosuite/bin/python -m aprwm_v0 r1-rs1b --smoke \
  --output runs/r1_rs1b/smoke \
  --rs0-summary runs/r1_rs0/c0/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 4) RS2-C0：Panda OSC + oracle robot–Door J^T f 接触闭合
.venv-robosuite/bin/python -m aprwm_v0 r1-rs2-c0 \
  --output runs/r1_rs2/c0_v2 \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 5) RS2 Formal：冻结 RS1C 在 robot-contact 下的 held-out 180-episode
.venv-robosuite/bin/python -m aprwm_v0 r1-rs2 \
  --output runs/r1_rs2/formal \
  --rs2-c0-summary runs/r1_rs2/c0_v2/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 6) RS2A：切空间可见激励诊断（不改 RS2 决策）
.venv-robosuite/bin/python -m aprwm_v0 r1-rs2a \
  --output runs/r1_rs2a/formal \
  --rs2-formal-summary runs/r1_rs2/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 7) RS2B：后果可迁移性诊断（不改 C_tol / 不重开 detectability）
.venv-robosuite/bin/python -m aprwm_v0 r1-rs2b \
  --output runs/r1_rs2b/formal \
  --rs2a-summary runs/r1_rs2a/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 8) RS3A：learner-visible 跨干预结构证据校准（不改阈值 / 不开 RS3B）
.venv-robosuite/bin/python -m aprwm_v0 r1-rs3a \
  --output runs/r1_rs3a/formal \
  --rs2b-summary runs/r1_rs2b/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 9) RS3A.1：交叉拟合超额结构风险（最后一次无条件标量尝试；不开 RS3B）
.venv-robosuite/bin/python -m aprwm_v0 r1-rs3a1 \
  --output runs/r1_rs3a1/formal \
  --rs3a-summary runs/r1_rs3a/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 10) RS4A：交互几何条件化证据（LOIO；无 domain ID；不开 RS4B）
.venv-robosuite/bin/python -m aprwm_v0 r1-rs4a \
  --output runs/r1_rs4a/formal \
  --rs3a1-summary runs/r1_rs3a1/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 11) RS5A：干预索引校准 + 校准支持域外弃权（旁路结果；不开 RS5B）
.venv-robosuite/bin/python -m aprwm_v0 r1-rs5a \
  --output runs/r1_rs5a/formal \
  --rs4a-summary runs/r1_rs4a/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 12) R3-V7A：混合具身域情境化 p_struct（主线；不是旧 2D V7）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7a \
  --output runs/r3_v7a/formal \
  --rs5a-summary runs/r1_rs5a/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 13) R3-V7B：持续认识状态 vs V7A 静态摘要（不开 V7C / RS5B）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7b \
  --output runs/r3_v7b/formal \
  --v7a-summary runs/r3_v7a/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 14) R3-V7B.1：同一 GRU，IBS vs time-weighted BCE（不开 V7C）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7b1 \
  --output runs/r3_v7b1/formal \
  --v7b-summary runs/r3_v7b/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 15) R3-V7B.2：快慢认识状态 vs B2 与容量匹配 GRU（不开 V7C）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7b2 \
  --output runs/r3_v7b2/formal \
  --v7b1-summary runs/r3_v7b1/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 16) R3-V7B.3：匹配反事实证据充分性监督（同一 GRU；不开 V7C）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7b3 \
  --output runs/r3_v7b3/formal \
  --v7b2-summary runs/r3_v7b2/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 17) R3-V7C：oracle 接触 → 带噪本体感觉接触（B5 GRU；不开 V7D）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7c \
  --output runs/r3_v7c/formal \
  --v7b3-summary runs/r3_v7b3/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 18) R3-V7C.1：学习触觉表示 vs B5-S（不开 V7D）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7c1 \
  --output runs/r3_v7c1/formal \
  --v7c-summary runs/r3_v7c/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 19) R3-V7C.2：触觉场 / encoder / fusion 增量信息诊断（不开 V7D）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7c2 \
  --output runs/r3_v7c2/formal \
  --v7c1-summary runs/r3_v7c1/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 20) R3-V7C.3：触觉观测充分性消融（线性读出；不开 encoder / V7D）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7c3 \
  --output runs/r3_v7c3/formal \
  --v7c2-summary runs/r3_v7c2/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 21) R3-V7C.4：NSG 表示充分性（不开 B5 / V7D）
.venv-robosuite/bin/python -m aprwm_v0 r3-v7c4 \
  --output runs/r3_v7c4/formal \
  --v7c3-summary runs/r3_v7c3/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 22) R3-V7C.5：B5-S + Z_NSG 信念接入
.venv-robosuite/bin/python -m aprwm_v0 r3-v7c5 \
  --output runs/r3_v7c5/formal \
  --v7c4-summary runs/r3_v7c4/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json

# 23) R3-V7D：RGB 相对 h^S 的条件模态价值（冻结 encoder；需 EGL）
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-robosuite/bin/python -m aprwm_v0 r3-v7d \
  --output runs/r3_v7d/formal \
  --v7c5-summary runs/r3_v7c5/formal/summary.json \
  --rs1a5-summary runs/r1_rs1a5/formal/summary.json
```

冻结版本：`robosuite==1.5.2`，`mujoco==3.11.0`。

- **MJ0**：9/9 闭合，最大 NRMSE `8.21e-11`（门限 `1e-4`）
- **RS0**：9/9 闭合，最大 NRMSE `2.79e-4`（门限 `1e-3`）；Panda 运动学冻结，
  `frictionloss` 经 `qfrc_constraint` 计入，learner 只显式建模 damping，且
  **禁止**读入 truth `qfrc_passive`
- **RS1B plumbing**：预注册单格通过完整 intake → revision → dynamics filter →
  blind H32；选中 `abs_v_v`，H32 gain `+0.00162`，无 passivity violation，
  support exit `0`。这是非科学 smoke。
- **RS1B formal**：30-episode held-out，`RS1B_GO=false`。passivity 与 seed-mean H32 gain 通过；accepted H32-stable rate `0.80 < 0.90`。
- **RS1B.1 formal**：新 held-out 30-episode，`RS1B.1_GO=false`。support exit
  \(\not\Rightarrow\) harmful；未安装 support-exit hard filter。见
  [REPORT/REP/R1/R1_RS1B1_REPORT.md](REPORT/REP/R1/R1_RS1B1_REPORT.md)。
- **RS1B.2 formal**：targeted query 填满 B0–B3，`RS1B.2_GO=true`。support-risk 分支收束，不设 RS1B.3。见
  [REPORT/REP/R1/R1_RS1B2_REPORT.md](REPORT/REP/R1/R1_RS1B2_REPORT.md)、
  [REPORT/REP/R1/R1_RS1B_STAGE_FREEZE.md](REPORT/REP/R1/R1_RS1B_STAGE_FREEZE.md)。
- **RS1C formal**：held-out 100-episode，`RS1C_GO=true`。见
  [REPORT/REP/R1/R1_RS1C_REPORT.md](REPORT/REP/R1/R1_RS1C_REPORT.md)。不改写 `RS1B_GO`。
- **RS2-C0**：27/27 closure cells 通过；contact projection audit 最大
  `3.69e-16`，full/active residual NRMSE 最大 `1.89e-4`，false revision
  `0/27`。adapter 冻结为 \(s^\star=2.0\)，exposure 误差 `18.84%`。
  见 [REPORT/REP/R1/R1_RS2_C0_REPORT.md](REPORT/REP/R1/R1_RS2_C0_REPORT.md)。
- **RS2 Formal**：held-out 180-episode，`RS2_GO=false`。C0 特异性通过；VoI
  0 次 promote；\(\pi_{\mathrm{RS1C}}\) 0 install。不改写 `RS1C_GO`。见
  [REPORT/REP/R1/R1_RS2_REPORT.md](REPORT/REP/R1/R1_RS2_REPORT.md)。
- **RS2A**：90-episode 诊断，`RS2A_GO=true`。\(X_{\phi,\perp}\) 比 \(X_\phi\)
  更能跨 Mode-A/contact 预测 \(D_0\)；不改阈值、不打开 RS2B。见
  [REPORT/REP/R1/R1_RS2A_REPORT.md](REPORT/REP/R1/R1_RS2A_REPORT.md)。
- **RS2B**：20-episode 诊断，`RS2B_GO=true`。Mode-A \(C\) 看不见接触下 latch
  的 incumbent 损失；不改 \(C_{\mathrm{tol}}\)，不重开 RS2A。见
  [REPORT/REP/R1/R1_RS2B_REPORT.md](REPORT/REP/R1/R1_RS2B_REPORT.md)。
- **RS2 stage freeze**：physics transfers, epistemics do not。`RS2_GO` 保持
  false。见 [REPORT/REP/R1/R1_RS2_STAGE_FREEZE.md](REPORT/REP/R1/R1_RS2_STAGE_FREEZE.md)。
- **RS3A**：90-episode，`RS3A_GO=false`。learner-visible \(S_\perp\) 在
  Mode-A/contact 间没有校准 operating point（\(g(S_\perp)>g(D_0)\)）。
  不开 RS3B/C，不改 `RS2_GO`。见
  [REPORT/REP/R1/R1_RS3A_REPORT.md](REPORT/REP/R1/R1_RS3A_REPORT.md)。
- **RS3A.1**：90-episode，`RS3A.1_GO=false`。C0 上 \(E_{\mathrm{XR}}\approx0\)
  （不是 memorizer），但共享 \(\tau_E\) 与 C1 operating-point 均不运输。
  **invariant-scalar 搜索关闭。** 见
  [REPORT/REP/R1/R1_RS3A1_REPORT.md](REPORT/REP/R1/R1_RS3A1_REPORT.md)。
- **RS4A**：90-episode LOIO，`RS4A_GO=false`。几何条件化 \(p_z\) 在未见
  干预上 Brier 全面差于 \(D_0\)-only；Mode-A held-out 的 \(p\approx0\)。
  不开 RS4B。见 [REPORT/REP/R1/R1_RS4A_REPORT.md](REPORT/REP/R1/R1_RS4A_REPORT.md)。
- **RS5A**：90-episode，`RS5A_GO=true`。已知干预上 indexed Brier/ECE 优于
  global；`pull_push` 上 abstain FCR \(0/30\)，indexed/global 为 \(1\)。
  **旁路架构结果，不是下一主线。** 不开 RS5B / revision。见
  [REPORT/REP/R1/R1_RS5A_REPORT.md](REPORT/REP/R1/R1_RS5A_REPORT.md)。
- **Transport freeze**：physics 可迁移，认识语义依赖上下文。见
  [REPORT/REP/R1/R1_TRANSPORT_STAGE_FREEZE.md](REPORT/REP/R1/R1_TRANSPORT_STAGE_FREEZE.md)。
- **R3-V7A**：90-episode，`V7A_GO=true`。混合训练情境化 \(p_{\mathrm{struct}}\)
  held-out Brier \(0.164<0.209\)（\(D_0\)-only）；C0 FPR \(0.083\)；
  各族 AUROC \(\ge0.75\)。不开 RS5B。见
  [REPORT/REP/R3/R3_V7A_REPORT.md](REPORT/REP/R3/R3_V7A_REPORT.md)。
- **R3-V7B**：105-episode，`V7B_GO=false`。GRU 终局 Brier \(0.128<0.205\)（V7A
  静态）、中点 \(0.129<0.303\)；C0 逐步 FPR \(0.281>0.20\)。失败是瞬态进入校准，
  不是长期 hysteresis。见
  [REPORT/REP/R3/R3_V7B_REPORT.md](REPORT/REP/R3/R3_V7B_REPORT.md)。
- **R3-V7B.1**：105-episode，`V7B.1_GO=false`。同一 GRU 上 IBS 未胜过 BCE
  （IBS \(0.143>0.134\)）；C0 FPR \(0.341\)；中点/终局非劣通过。不开 V7C。见
  [REPORT/REP/R3/R3_V7B1_REPORT.md](REPORT/REP/R3/R3_V7B1_REPORT.md)。
- **R3-V7B.2**：105-episode，`V7B.2_GO=false`。快慢+门控对 B2 累积更好
  （终局 \(0.106<0.121\)），事件升幅 \(0.086<\eta=0.120\)；C0 FPR \(0.595\)，
  \(B_{C0}\) 差于 B2 与 wide GRU。不开 V7C。见
  [REPORT/REP/R3/R3_V7B2_REPORT.md](REPORT/REP/R3/R3_V7B2_REPORT.md)。
- **R3-V7B.3**：105-episode 匹配对，`V7B.3_GO=true`。B5 vs B2：\(B_{C0}\)
  \(0.219<0.293\)，C0 FPR \(0.006\)；Spearman \((p,w)=0.21\)；配对间隙随 \(w\)
  增大。同一 GRU。见
  [REPORT/REP/R3/R3_V7B3_REPORT.md](REPORT/REP/R3/R3_V7B3_REPORT.md)。
- **R3-V7C**：105-episode，`V7C_GO=true`。B5-S vs O 非劣（中点
  \(0.158\le0.170+0.02\)）；C0 FPR \(0.013\)；Spearman \(0.24\)；相对无接触
  通道 \(B_{C0}\) 与 C1 终局更好（混合 Brier 则 \(N<S\)）。见
  [REPORT/REP/R3/R3_V7C_REPORT.md](REPORT/REP/R3/R3_V7C_REPORT.md)、
  [REPORT/REP/R3/R3_V7_STAGE_FREEZE.md](REPORT/REP/R3/R3_V7_STAGE_FREEZE.md)。
- **R3-V7C.1**：105-episode，`V7C.1_GO=false`。H1–H3 过；H4 败
  （\(B_{C0}\) 差于无接触）。V7D 锁定。见
  [REPORT/REP/R3/R3_V7C1_REPORT.md](REPORT/REP/R3/R3_V7C1_REPORT.md)。
- **R3-V7C.2**：105-episode 诊断，locus **FIELD**。\(\Delta_X=0.001\le0.005\)。
  见 [REPORT/REP/R3/R3_V7C2_REPORT.md](REPORT/REP/R3/R3_V7C2_REPORT.md)。
- **R3-V7C.3**：105-episode 观测消融，`stop_tactile_branch=false`。
  \(\Delta_{\mathrm{NSG}}=0.057\)；剪切嵌套未增益。不开 encoder / V7D。
  见 [REPORT/REP/R3/R3_V7C3_REPORT.md](REPORT/REP/R3/R3_V7C3_REPORT.md)。
- **R3-V7C.4**：105-episode 表示充分性，冻结 `SUFFICIENT`。同 run
  \(\Delta_Z\approx\Delta_X=0.015\)；shuffle 几乎不降。不开 B5 / V7D。
  见 [REPORT/REP/R3/R3_V7C4_REPORT.md](REPORT/REP/R3/R3_V7C4_REPORT.md)。
- **R3-V7C.5**：105-episode 信念接入，`V7C.5_GO=false`。追加 \(Z_{NSG}\)
  相对 B5-S 使 BCE/Brier 变差；H5 时间门控失败。不开 V7C.6。
  见 [REPORT/REP/R3/R3_V7C5_REPORT.md](REPORT/REP/R3/R3_V7C5_REPORT.md)。
- **R3-V7D**：105-episode，`V7D_GO=false`。冻结 RGB encoder 接入 B5-S
  后 \(L(S)=0.441\)、\(L(S{+}Z)=0.488\)、shuffle \(0.470\)；H1/H4/H5 败。
  见 [REPORT/REP/R3/R3_V7D_REPORT.md](REPORT/REP/R3/R3_V7D_REPORT.md)。
- **R3-V7 Door 关闭**：无 V7E。\(h^{S}\) 已饱和当前 Door 结构错误判断。
  见 [REPORT/REP/R3/R3_V7_STAGE_FREEZE.md](REPORT/REP/R3/R3_V7_STAGE_FREEZE.md)、
  [REPORT/REP/R3/R3_V7_EVIDENCE_CHAIN.md](REPORT/REP/R3/R3_V7_EVIDENCE_CHAIN.md)。
- **R4-I0 v1**：`PASS=false`。\(\mu\) 泄漏进 \(F_n\)。
  见 [REPORT/REP/R4/R4_I0_REPORT.md](REPORT/REP/R4/R4_I0_REPORT.md)。
- **R4-I0 v2**：`PASS=true`。同 \(\mu\) / 平均 `solref`，左右刚度交换；
  475 帧宏观匹配，371 帧局部 \(p,\tau\) 翻转。不开 C0。
  见 [REPORT/REP/R4/R4_I0_V2_REPORT.md](REPORT/REP/R4/R4_I0_V2_REPORT.md)。
- **R4-I1**：`PASS=true`。轨迹划分线性探针；\(\mathrm{AUROC}(h^{S})=0.556\)，
  增益 \(0.440\)；canonicalize 诊断 \(\approx 0.50\)。I0 协议冻结。
  见 [REPORT/REP/R4/R4_I1_REPORT.md](REPORT/REP/R4/R4_I1_REPORT.md)。
- **R4-C0**：`GO=false`。\(\Delta_X=-0.015\le\varepsilon\)；宏观残差不产生
  \(w_t\)，故 \(e_t\equiv0\)。可观测 \(\neq\) 认识必要。不训 encoder。
  见 [REPORT/REP/R4/R4_C0_REPORT.md](REPORT/REP/R4/R4_C0_REPORT.md)、
  [REPORT/REP/R4/R4_STAGE_FREEZE.md](REPORT/REP/R4/R4_STAGE_FREEZE.md)。
- **R4-C1**：F0 过（held \(\mathbb{E}[C]=0.00118>10^{-3}\)）；F1 二值标签在 train 上饱和，未评估。
  `GO=false`，**不是** nuisance 停线。不训 encoder。
  见 [REPORT/REP/R4/R4_C1_REPORT.md](REPORT/REP/R4/R4_C1_REPORT.md)。
- **R4-C2**：`GO=false`，**R4 family stop**。\(\Delta_C\approx 0\)；
  \(L_C(h^{S})\approx L_C(h^{S},X)\approx 1.055\)。未来后果存在但触觉不能量化。
  见 [REPORT/REP/R4/R4_C2_REPORT.md](REPORT/REP/R4/R4_C2_REPORT.md)。
- **R4 family STOP**：四层分离已冻结。表示容量跟条件预测价值，不跟物理可观测性。
  见 [REPORT/REP/R4/R4_FAMILY_STOP.md](REPORT/REP/R4/R4_FAMILY_STOP.md)。
- **R5 接触族 PAUSE**（不是不可能定理）。I0 v1–v3 均未让三门共存；v3 已证明
  \(I(X;\lambda\mid h^{S})>0\not\Rightarrow I(X;C\mid h^{S})>0\)。无 neural。
  见 [REPORT/REP/R5/R5_CONTACT_FAMILY_PAUSE.md](REPORT/REP/R5/R5_CONTACT_FAMILY_PAUSE.md)。
- **R5-I0-SELFSTRESS**：新 family 的无学习 I0。\(D_h=0\)、\(X=\lambda\)，G_cons/G_xc 败（Spearman \(0.75\)）。**判决冻结**。
  见 [REPORT/REP/R5/R5_I0_SELFSTRESS_REPORT.md](REPORT/REP/R5/R5_I0_SELFSTRESS_REPORT.md)。
- **R5-I1-SELFSTRESS**：`GO=true`。不改 I0。\(X\) 对完整 \(Y^{\mathrm{future}}\) 的 ridge 条件增益 \(\Delta_Y/L_0=0.387\)。无 encoder。
  见 [REPORT/REP/R5/R5_I1_SELFSTRESS_REPORT.md](REPORT/REP/R5/R5_I1_SELFSTRESS_REPORT.md)。
- **R5-D0-preflight**：`PASS=true`。冻结 \(\mathcal A\) 与任务 \(J\) 下 oracle 动作排序随 \(\lambda\) 翻转（cons \(\to\) mid）。
  见 [REPORT/REP/R5/R5_D0_PREFLIGHT_REPORT.md](REPORT/REP/R5/R5_D0_PREFLIGHT_REPORT.md)。
- **R5-D0**：`GO=false`。self-stress **STOP**。否定的是冻结 ridge-then-\(J\) 上的
  **realized** \(\mathrm{VoI}_{\Pi}<0\)，不是 oracle \(\mathrm{VoI}\le 0\)。
  见 [REPORT/REP/R5/R5_D0_REPORT.md](REPORT/REP/R5/R5_D0_REPORT.md)、
  [REPORT/REP/R5/R5_SELFSTRESS_FAMILY_STOP.md](REPORT/REP/R5/R5_SELFSTRESS_FAMILY_STOP.md)。
- **R6-A0**：离线决策敏感误差审计（无训练）。D1/D2 成立；D3 \(R^2=0.114\)。
  \(L_Y\) 降 \(53\%\)，margin violation \(1\to3\)。不改 planner / WM。
  见 [REPORT/REP/R6/R6_A0_REPORT.md](REPORT/REP/R6/R6_A0_REPORT.md)。
- **R6-A1**：pairwise \(J\)-margin nested-LOO 不确定性审计。PX 三次错排均 \(z<1\)；
  \(\lambda=2\) 未触发自信错排否决。mid vs cons 在所有 \(\lambda\) 上都低置信。
  见 [REPORT/REP/R6/R6_A1_REPORT.md](REPORT/REP/R6/R6_A1_REPORT.md)。
- **R6-B0**：冻结弃权闭环。`POLICY_COLLAPSE=true`，\(\pi_B\equiv\pi_0\)，无新仿真。
  有害切换 2/2 撤回；有用切换 0 次提出。realization 支线 **STOP**。
  见 [REPORT/REP/R6/R6_B0_REPORT.md](REPORT/REP/R6/R6_B0_REPORT.md)、
  [REPORT/REP/R6/R6_REALIZATION_FREEZE.md](REPORT/REP/R6/R6_REALIZATION_FREEZE.md)。
- **R7**：switch certification。**R7-P0 `PASS=true`**；**R7-A0 `GO=true`**；
  **R7-P1 `PASS=true`**（饱和执行器；degree-2 LOO \(R^2=0.68\)）；
  **R7-A1 `GO=true`**（同一 \(f\) 与 \(L=\hat A-q\)；\(q\uparrow\)；recall
  0.273；无有害 commit）；
  **R7-A2 `GO=true`**（只换冻结三次样条；recall 0.909；precision 1；
  coverage 0.964；**A-series 冻结**）；
  **R7-B0 `GO=true`**（\(Y\)-only WM \(\to J\to\hat A\)；recall 0.70；
  precision 1；coverage 0.929；shuffle 零 commit）；
  **R7-B1 `shift_valid=false`**（冻结 \(f,q\)；\(F_{\max}:2\to 1.5\)；
  coverage 0.357；3 次有害 commit）；
  **R8-P0 `PASS=false`**（locus probe_cost）；
  **R8-P1 `PASS=false`**（四相自消除；G3 过、G1/G-return 不过）；
  **R8-P2 `PASS=false`**（阻尼回零；G-return/G3 过、G1 不过；open-loop
  probe family 冻结）；
  **R8-R0 `PASS=false`**（计费 probe→reset；G1/G-indep 过、G-reset 不过；
  同幕 family **STOP**）；
  **R9-P0 `PASS=true`**（摊销校准；\(K_{\min}=2\)）；
  **R9-A0 `GO=true`**（persistent \(b(\mathcal V)\)；\(\Delta K=0\)）；
  **R9-B0 `GO=false`**（被动撤销；policy-induced epistemic blindness）；
  **R9-B1-P0 `PASS=false`**（\(M=2\) 周期再验证；dwell 有界，invalid NetVoI 不过）；
  **论文级总账** + **VIS-X0–X3 / VIS-EXT0 PASS** + **CAP-X0 PASS** +
  **CAP-X1 PASS**（\(R_P(0)=1\) matched-family upper bound）+
  **CAP-X2 COMPLETE / `reference_failure`** + **PLAN-X0 PASS** +
  **PLAN-X1 FAIL / `anisotropy_no_value`**（H1 支持、H2 否定；PLAN-X2 未触发）+
  **PLAN-X1.5 FAIL / `iso_sufficient`**（mean+iso 足够；不开 diffusion）+
  **CAP-X3-P0 PASS** + **CAP-X3 PASS / `structured_adaptation_advantage`**
  （因果坐标超越同维降维；\(K_{90}^{\mathrm{phy}}=1\), \(K_{90}^{\mathrm{lat}}=2\), \(R_K=0.5\)）+
  **论文架构冻结**（三支柱；不开 X3B/X4/R10）+
  **RoboTwin-X0-P0 PASS**；官方 smoke **FAIL**（assets 已装；WSL 无 NVIDIA Vulkan，`SapienRenderer` 起不来）
  [REPORT/REP/PAPER/APRWM_PAPER_ARCHITECTURE.md](REPORT/REP/PAPER/APRWM_PAPER_ARCHITECTURE.md)、
  [REPORT/REP/RTWX/RTWX0_P0_REPORT.md](REPORT/REP/RTWX/RTWX0_P0_REPORT.md)、
  [REPORT/REP/PAPER/APRWM_R7_R9_SIMX_SYNTHESIS.md](REPORT/REP/PAPER/APRWM_R7_R9_SIMX_SYNTHESIS.md)、
  [REPORT/REP/CAPX/CAPX0_REPORT.md](REPORT/REP/CAPX/CAPX0_REPORT.md)、
  [REPORT/REP/CAPX/CAPX1_REPORT.md](REPORT/REP/CAPX/CAPX1_REPORT.md)、
  [REPORT/REP/CAPX/CAPX2_P0_REPORT.md](REPORT/REP/CAPX/CAPX2_P0_REPORT.md)、
  [REPORT/REP/CAPX/CAPX2_REPORT.md](REPORT/REP/CAPX/CAPX2_REPORT.md)、
  [REPORT/REP/PLANX/PLANX0_REPORT.md](REPORT/REP/PLANX/PLANX0_REPORT.md)、
  [REPORT/REP/PLANX/PLANX1_REPORT.md](REPORT/REP/PLANX/PLANX1_REPORT.md)、
  [REPORT/REP/PLANX/PLANX15_REPORT.md](REPORT/REP/PLANX/PLANX15_REPORT.md)、
  [REPORT/REG/PLANX/PLANX_PREREG.md](REPORT/REG/PLANX/PLANX_PREREG.md)、
  [REPORT/REP/CAPX/CAPX3_REPORT.md](REPORT/REP/CAPX/CAPX3_REPORT.md)、
  [REPORT/REG/CAPX/CAPX3_PREREG.md](REPORT/REG/CAPX/CAPX3_PREREG.md)、
  [REPORT/REG/CAPX/CAPX3_P0_PREREG.md](REPORT/REG/CAPX/CAPX3_P0_PREREG.md)。
  见 [REPORT/REP/R7/R7_A0_REPORT.md](REPORT/REP/R7/R7_A0_REPORT.md)、
  [REPORT/REP/R7/R7_A2_REPORT.md](REPORT/REP/R7/R7_A2_REPORT.md)、
  [REPORT/REP/R7/R7_B0_REPORT.md](REPORT/REP/R7/R7_B0_REPORT.md)、
  [REPORT/REP/R7/R7_B1_REPORT.md](REPORT/REP/R7/R7_B1_REPORT.md)、
  [REPORT/REP/R8/R8_P0_REPORT.md](REPORT/REP/R8/R8_P0_REPORT.md)、
  [REPORT/REP/R8/R8_P1_REPORT.md](REPORT/REP/R8/R8_P1_REPORT.md)、
  [REPORT/REP/R8/R8_P2_REPORT.md](REPORT/REP/R8/R8_P2_REPORT.md)、
  [REPORT/REP/R8/R8_R0_REPORT.md](REPORT/REP/R8/R8_R0_REPORT.md)、
  [REPORT/REP/R8/R8_SAME_HORIZON_STOP.md](REPORT/REP/R8/R8_SAME_HORIZON_STOP.md)、
  [REPORT/REP/R9/R9_P0_REPORT.md](REPORT/REP/R9/R9_P0_REPORT.md)、
  [REPORT/REP/R9/R9_A0_REPORT.md](REPORT/REP/R9/R9_A0_REPORT.md)、
  [REPORT/REP/R9/R9_B0_REPORT.md](REPORT/REP/R9/R9_B0_REPORT.md)、
  [REPORT/REP/R9/R9_B1_P0_REPORT.md](REPORT/REP/R9/R9_B1_P0_REPORT.md)、
  [REPORT/REP/R9/R9_TOY_FAMILY_STOP.md](REPORT/REP/R9/R9_TOY_FAMILY_STOP.md)、
  [REPORT/REG/R10/R10_PREREG.md](REPORT/REG/R10/R10_PREREG.md)、
  [REPORT/REG/SIMX/SIMX_PREREG.md](REPORT/REG/SIMX/SIMX_PREREG.md)、
  [REPORT/REG/SIMX/SIMX0_PREREG.md](REPORT/REG/SIMX/SIMX0_PREREG.md)、
  [REPORT/REG/SIMX/SIMX0_FORCE_ACCOUNTING.md](REPORT/REG/SIMX/SIMX0_FORCE_ACCOUNTING.md)、
  [REPORT/REP/SIMX/SIMX0_REPORT.md](REPORT/REP/SIMX/SIMX0_REPORT.md)、
  [REPORT/REG/SIMX/SIMX1_PREREG.md](REPORT/REG/SIMX/SIMX1_PREREG.md)、
  [REPORT/REP/SIMX/SIMX1_REPORT.md](REPORT/REP/SIMX/SIMX1_REPORT.md)、
  [REPORT/REG/SIMX/SIMX2_PREREG.md](REPORT/REG/SIMX/SIMX2_PREREG.md)、
  [REPORT/REP/SIMX/SIMX2_REPORT.md](REPORT/REP/SIMX/SIMX2_REPORT.md)、
  [REPORT/REG/SIMX/SIMX3_PREREG.md](REPORT/REG/SIMX/SIMX3_PREREG.md)、
  [REPORT/REP/SIMX/SIMX3_REPORT.md](REPORT/REP/SIMX/SIMX3_REPORT.md)、
  [REPORT/REP/SIMX/SIMX_FAMILY_STOP.md](REPORT/REP/SIMX/SIMX_FAMILY_STOP.md)、
  [REPORT/REG/REALLOG/REALLOG_A0_PREREG.md](REPORT/REG/REALLOG/REALLOG_A0_PREREG.md)、
  [REPORT/REP/REALLOG/REALLOG_A0_REPORT.md](REPORT/REP/REALLOG/REALLOG_A0_REPORT.md)、
  [REPORT/REP/PAPER/APRWM_R7_R9_SIMX_SYNTHESIS.md](REPORT/REP/PAPER/APRWM_R7_R9_SIMX_SYNTHESIS.md)、
  [REPORT/REG/VISX/VISX_PREREG.md](REPORT/REG/VISX/VISX_PREREG.md)、
  [REPORT/REG/VISX/VISX0_PREREG.md](REPORT/REG/VISX/VISX0_PREREG.md)、
  [REPORT/REG/VISX/VISX1_PREREG.md](REPORT/REG/VISX/VISX1_PREREG.md)、
  [REPORT/REP/VISX/VISX0_REPORT.md](REPORT/REP/VISX/VISX0_REPORT.md)、
  [REPORT/REP/VISX/VISX1_REPORT.md](REPORT/REP/VISX/VISX1_REPORT.md)、
  [REPORT/REP/VISX/VISX2_REPORT.md](REPORT/REP/VISX/VISX2_REPORT.md)、
  [REPORT/REG/VISX/VISX2_PREREG.md](REPORT/REG/VISX/VISX2_PREREG.md)、
  [REPORT/REG/VISX/VISX3_PREREG.md](REPORT/REG/VISX/VISX3_PREREG.md)、
  [REPORT/REP/VISX/VISX3_REPORT.md](REPORT/REP/VISX/VISX3_REPORT.md)、
  [REPORT/REG/VISX/VISEXT0_PREREG.md](REPORT/REG/VISX/VISEXT0_PREREG.md)。

基础桥说明见 [REPORT/REP/R1/R1_MJRS_MJ0_RS0_REPORT.md](REPORT/REP/R1/R1_MJRS_MJ0_RS0_REPORT.md)，RS1B
见 [REPORT/REG/R1/R1_RS1B_PREREG.md](REPORT/REG/R1/R1_RS1B_PREREG.md)、
[REPORT/REP/R1/R1_RS1B_SMOKE_REPORT.md](REPORT/REP/R1/R1_RS1B_SMOKE_REPORT.md)、
[REPORT/REP/R1/R1_RS1B_REPORT.md](REPORT/REP/R1/R1_RS1B_REPORT.md)。
