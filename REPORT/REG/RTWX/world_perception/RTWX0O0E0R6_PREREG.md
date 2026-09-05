# RTWX-O0E0R6 预注册 — Finite-Step Alternating Center–Axis Inference

日期：2026-08-31  
状态：**已冻结 / RAN**  
依赖：O0E0R5=`visibility_reference_joint_failure`；\(S_0\) / visibility LUT / B2 objective+search **冻结**  
**唯一变更**：inference schedule（**不**改 LUT、**不**改 \(S(n)\)、**不**训练、**不** persistent surface）

**禁止**：\(\lambda D_{ho}\) 加权 score（仅 diagnostic）；迭代至收敛；新采集 formal test；开 O0E1 / O1。

## 科学假设

\[
\boxed{
H_{\mathrm{alt}}:
\text{R5 failure 来自 candidate-specific center 与 axis 同步自洽；
有限步解耦更新可恢复正确 basin。}
}
\]

失败机制（R5 已实证）：

\[
n'\rightarrow \hat p_{\mathrm{vis}}(n')\rightarrow S(n';\hat p_{\mathrm{vis}}(n')) < S(n^*;\hat p_{\mathrm{vis}}(n^*)).
\]

本格原则：

\[
\boxed{
\text{不要让每个 axis candidate 在搜索中自由获得自己的 center。}
}
\]

## 冻结组件

| 组件 | 状态 |
|------|------|
| \(S_0\) natural UNet | 冻结（R5 权重 / cache） |
| `visibility_lut.npz` | 冻结（R5 产物） |
| B2 \((h,r)\) + Fibonacci 162 → Top-8 → \(5°/3°/1°\) | 冻结 |
| Controlled pose / 128² / dual camera | 冻结 |

## 数据

| 项 | 值 |
|----|-----|
| Formal test | **复用** R5 cache seed **37605**；n=200；**无新采集** |
| 输入 | `runs/rtwx_o0e0r5/cache/test/` + `visibility_lut.npz` |

同帧 paired 比较三支 inference branch。

## Inference branches（冻结）

### A0 — Joint baseline（R5 复现）

\[
n\rightarrow \hat p_{\mathrm{vis}}(n)\rightarrow S(n;\hat p_{\mathrm{vis}}(n)).
\]

\[
\hat n_0=\arg\min_n S(n;\hat p_{\mathrm{vis}}(n)),\quad \hat p_0=\hat p_{\mathrm{vis}}(\hat n_0).
\]

Sanity：应复现 R5 formal vis **~16.6°** median。

### A1 — Center-first（2-step，不收敛）

**冻结初始化**（独立于当前 B2 candidate loop）：

\[
n^{(0)} \equiv n_{\mathrm{const}}
\]

\(n_{\mathrm{const}}\)：P0 train GT axis 归一化和（与 O0E0R0–R5 相同 B0 constant）。

| Step | 操作 |
|------|------|
| 1 | \(p^{(0)}=\hat p_{\mathrm{vis}}(n^{(0)})\)（dual-view 等权 fuse） |
| 2 | \(n^{(1)}=\arg\min_n S(n;p^{(0)})\)（**固定** \(p^{(0)}\)，标准 B2 at center） |
| 3 | \(p^{(1)}=\hat p_{\mathrm{vis}}(n^{(1)})\) |
| 4 | \(n^{(2)}=\arg\min_n S(n;p^{(1)})\)（**固定** \(p^{(1)}\)） |

输出：\((\hat p,\hat n)=(p^{(1)}, n^{(2)})\)。**禁止**第三轮。

### A2 — Consensus center

1. 在 **A0 初筛** Fibonacci 162 上取 score 最低 **Top-8** 候选 \(\{n_k\}_{k=1}^8\)（与 B2 Top-8 **同一集合**，deterministic）。
2. \(\bar p=\operatorname{median}_k \hat p_{\mathrm{vis}}(n_k)\)（coordinate-wise median）。
3. \(\hat n=\arg\min_n S(n;\bar p)\)（固定 \(\bar p\)，完整 B2 refine）。
4. \(\hat p=\bar p\)（formal position 报告用 \(\bar p\)；不再 per-candidate 更新）。

## Formal gates（per branch，不变）

Axis：median \(\le15^\circ\)，P90 \(\le30^\circ\)。  
Position：median \(\le5\) cm，\(E_p\le0.20\)。

**Primary outcome branch**：A1 与 A2 中 **较好** median axis 者；unlock 需 **任一** branch 双门 PASS。

## Diagnostics（非门）

### D\_ho — Head/observer center disagreement

对 Fibonacci 候选 \(n\)：

\[
D_{ho}(n)=\|\hat p_h(n)-\hat p_o(n)\|.
\]

报告：

- median / P90 \(D_{ho}(\hat n^{GT})\) vs \(D_{ho}(n')\)（oracle 邻域 vs 全体候选）
- \(\operatorname{AUROC}(-D_{ho}(n),\ \text{axis correct within }15^\circ)\)（**仅 diagnostic**）
- A0 wrong-top-1 vs correct-neighborhood \(D_{ho}\) 分布

**禁止**在本格将 \(D_{ho}\) 写入 \(S_{\mathrm{joint}}\)。

### D\_paired — A0 vs A1 vs A2

同帧 \(\Delta e_{\mathrm{axis}}\)、\(\Delta e_p\)；P90 tail 是否收缩。

## L1 contingency

若 R5 cache 缺失：STOP（不 silent re-collect）。  
可选 sanity：G\_I1 oracle-axis **≤1 行日志** 确认 LUT 未变（非 formal claim）。

## Patterns

| Pattern | 条件 |
|--------|------|
| `joint_inference_supported` | A1 **或** A2 formal axis+position 均 PASS |
| `joint_inference_insufficient` | A1 与 A2 均 ¬G1 |
| `visibility_reference_joint_failure` | A0 复现 R5 级 FAIL（sanity tag） |

O0E1 **LOCKED** 除非 `joint_inference_supported`。

## 解释树（预冻结）

| 结果 | 结论 |
|------|------|
| A1/A2 PASS | blocker = inference coupling；非 observation/model capacity |
| 仍 FAIL + 大 P90 | 证据链支持进入 persistent surface / multi-frame reference（R7） |
| A2 ≫ A1 | consensus 可分离错误 candidate center |
| A1 ≫ A2 | 需 axis-informed center update，非 blind median |

## 后续顺序（本格不跑）

```text
R6: alternating / consensus inference  [本格]
R7: persistent surface (仅 R6 FAIL)
O0E1: multi-symmetry-class confirmation
O1: persistent dynamic belief
```

## 实现备忘

- `aprwm_v0/rtwx_o0e0r6.py`
- 复用 `geometry/visibility_reference.py`、`estimate_axis_b2_at_center` / `estimate_axis_b2_center_fn`
- `runs/rtwx_o0e0r6/`：summary + per_frame branch outputs
