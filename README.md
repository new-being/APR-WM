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

V0.5 已补充 5 seeds、Random/heuristic/learned/oracle Pareto、固定 material 的连续复杂度、H=1–50 rollout、真正 conditional execution 的 CUDA latency，以及 router ablation。完整结论见 [V05_REPORT.md](V05_REPORT.md)。

最重要的修正是：learned router 的排序信号成立，但当前环境中 contact edge 本身只占全图约 5%，廉价 contact filtering 已能获得大部分解析计算节省；同时小 residual MLP 的 sparse GPU 路径比 dense 路径更慢。因此目前不能声称 89% 实际计算或延迟下降。

## V0.6 Compute-valid Adaptive Routing

V0.6 将统计稀疏性与真实计算收益分开验证。合成 benchmark 中 interaction density 与 conditional residual necessity 独立变化；同一 interaction 是否进入非线性 residual regime 只由当前 extension、relative velocity 与 friction 决定，不提供 object/edge ID。公平基线是已经执行解析 Stage-0 filtering 的 `contact_packed`，learned router 只在 active edge 内决定是否调用 residual expert。

```bash
python -m aprwm_v0 v06 \
  --output runs/v06/core \
  --device cuda \
  --full-matrix
```

矩阵覆盖 interaction density `20/50/80/100%`、conditional residual necessity `5/10/20/40%`、近似真实 FLOPs 为 `1×/2×/4×/8×` 的 residual expert、`linear/tiny/current` 三种 router，以及 batch `1/16/128/512`。`hard_sparse` 的 CUDA 计时包含 router、top-k 与 residual-row packing；Stage-0 active-edge packing 是两种方法的共享前置成本，不计入二者的增量比较。完整结论及局限见 [V06_REPORT.md](V06_REPORT.md)。

## V0.7 Utility-supervised Routing

V0.7 冻结 residual expert 后，以完全相同的 router、loss 和 10% top-budget 比较 `magnitude`、binary `necessity` 与 task-weighted `utility` supervision，并用 privileged oracle 给出分配上界。正式实验使用 5 seeds，只测 B16/8×、B128/8×、B512/1×、B512/8× 四个 slower/crossover/faster 代表点。CUDA 延迟采用逐样本交替 baseline/adaptive 的 paired timing，避免 GPU boost 和执行顺序污染边界结论。

```bash
python -m aprwm_v0 v07 \
  --output runs/v07/core \
  --device cuda \
  --seeds 13 23 33 43 53
```

8× expert 下，utility router 捕获 `96.78%` oracle utility，显著高于 magnitude 的 `94.24%`；但 1× expert 上差异的置信区间跨零，因此 utility supervision 不是无条件占优。完整配对统计、硬件边界与 cost-aware objective 的等价性见 [V07_REPORT.md](V07_REPORT.md)。

下一阶段固定为 unknown-physics V1：先隐藏 `(k,c)` 并维护参数 posterior，显式分离 model-class inadequacy 与 parameter-estimation error；实施边界和成功判据见 [V1_PLAN.md](V1_PLAN.md)。

## V1 Unknown-physics Posterior

V1 最小实验已实现：每条 trajectory 隐藏 spring stiffness/damping `(k,c)`，由 4/8/16/32 条 context interaction 构造 Bayesian parameter posterior；结构非线性由独立 latent model class 控制。Residual expert 只监督 oracle-parameter 下仍存在的 structural correction，从训练目标上禁止它直接吸收 parameter error。

```bash
python -m aprwm_v0 v1 \
  --output runs/v1/core \
  --device cuda \
  --seeds 13 23 33 43 53
```

五 seed 结果表明：adequate physics class 下 posterior calibration 正常，90% 区间覆盖率为 `90.55%`；存在结构失配时覆盖率跌至 `17.40%`，说明 misspecified posterior 会变得严重过度自信。Posterior-full router 将严格 residual misuse rate 从 state-only 的 `10.09%` 降至 `0.80%`，并捕获 `95.09%` structural utility；但由于 parameter bias 与 structural error 强烈抵消，它在 estimated physics 上的短期 force RMSE 反而更高。完整分析见 [V1_REPORT.md](V1_REPORT.md)。

## V2 Misspecification-aware Epistemic Control

V2 将 environment action `a_t` 与 internal compute decision `r_t` 分开。联合粒子 belief 维护 `p(θ,M|D)`；环境动作从 passive、velocity probe、amplitude probe 中选择，计算策略再依据 model-class belief、净预测效用和 residual cost 决定是否调用 structural residual。

```bash
python -m aprwm_v0 v2 \
  --output runs/v2/core \
  --device cuda \
  --seeds 13 23 33 43 53
```

低幅数据构造出 compensation trap：强制线性模型在 inadequate class 中学到 `+0.373` 的 pseudo-true stiffness bias，ID RMSE 仅 `0.0256`，intervention RMSE 却升至 `0.2194`。Misspecification-aware joint-IG 相对 parameter-IG 将 model AUROC 从 `0.9155` 提至 `0.9632`，intervention RMSE 从 `0.1949` 降至 `0.1740`，同时平均少用 `0.186` 次 probe；与固定两次 amplitude probe、但仍需推断模型类别的 diagnostic oracle 在 intervention RMSE 上无显著差异。直接提供真实 model class 的上界达到 `0.0491`，说明主要剩余瓶颈是类别判定，而非诊断动作选择。完整方法、配对统计和边界见 [V2_REPORT.md](V2_REPORT.md)。

## V3-min Tangent-triggered Model Revision

V3-min 初始只激活 linear physics。它在多观测窗口上把 residual 投影为 parameter-tangent 分量与正交分量，用持续的正交证据决定保持线性、激活 dormant cubic hypothesis，或在候选结构无法通过 held-out 检验时声明 `unknown`。第三种真值是候选扩展中不存在的 velocity-coupled discrepancy，用于测试开放集拒识。

```bash
python -m aprwm_v0 v3 \
  --output runs/v3/core \
  --device cuda \
  --seeds 101 111 121 131 141
```

五个未参与阈值开发的 seeds 上，tangent evidence 相对 residual magnitude 将 structural AUROC 从 `0.7709` 提至 `0.9957`，adequate false revision 从 `65.44%` 降至 `0.92%`，三态 model accuracy 从 `70.22%` 提至 `89.61%`；intervention RMSE 从 `0.1751` 降至 `0.1641`。但 unseen 类即使有 `89.08%` 被正确判为 unknown，其 RMSE 仍为 `0.2690`，远高于 oracle-family 的 `0.0325`。因此 V3-min 支持 tangent-triggered revision，同时明确证明 detection 不等于 model discovery。完整结果见 [V3_REPORT.md](V3_REPORT.md)。

## V4-min Residual-guided Active Operator Discovery

V4-min 将 `unknown` 后续处理拆成三个独立阶段：用 tangent-orthogonal residual 在八个结构算子中生成 top-3 proposal；执行候选预测分歧最大的诊断动作；最后通过独立 held-out 数据与复杂度成本决定是否永久接受单个 operator。Cubic 与 velocity-coupled 真值位于字典内，另有一种 thresholded oscillatory dynamics 完全不在字典中。

```bash
python -m aprwm_v0 v4 \
  --output runs/v4/core \
  --device cuda \
  --seeds 201 211 221 231 241
```

五个 held-out seeds 上，orthogonal proposal 的 top-3 recall 为 `98.99%`，主动诊断将 exact operator recovery 从 passive 的 `43.66%` 提至 `60.45%`。Oracle proposal 只额外提升 `0.38` 个百分点，而 oracle selection 达到 `100%`，说明当前瓶颈已经从 proposal 转移到 candidate discrimination 与 held-out acceptance。主动发现还将 residual fallback 从 `65.24%` 降至 `25.76%`，保持 outside-library rejection `90.80%`、adequate incorrect expansion `0%`。完整方法、oracle gap 与边界见 [V4_REPORT.md](V4_REPORT.md)。

## 研究边界

这是验证机制的受控 toy environment，不是最终具身系统。`material` 是 V0 的 oracle 物理属性；V1 才应把它改为带不确定性的 interaction belief / parameter posterior。V0.7 已补充 routing 的五 seed 统计，但仍需端到端 rollout、energy profiling，以及 unknown-physics 下 model inadequacy 与 parameter error 的显式分离。
