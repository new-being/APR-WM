# RTWX-O0HYB0 预注册 — Prior-Anchored Hybrid State Update

日期：2026-08-31  
状态：**已冻结 / RAN**  
依赖：REL0/REL0A；R6 A1；REL0 frozen ICP；visibility LUT  
**禁止**：改 ICP/B2/A1；解锁 O0E1/O1

## 架构假设

\[
\boxed{
\hat n_0=n_{\mathrm{const}},\quad
\hat p_0=p_{\mathrm{vis}}(n_{\mathrm{const}}),\quad
\hat s_1^{rel}=\hat\Delta T\cdot\hat s_0,\quad
\hat s_1^{abs}=\mathrm{R6\text{-}A1}(O_1),\quad
\hat s_1=
\begin{cases}
\hat s_1^{rel},& q_{\mathrm{rel}}\ge\tau^*\\
\hat s_1^{abs},& q_{\mathrm{rel}}<\tau^*
\end{cases}
}
\]

**one-step** \(t_0\to t_1\)；near-upright 初态 regime。

## 数据

| 项 | 值 |
|----|-----|
| Calibration seed | **37607** |
| Formal seed | **37608** |
| 每 split × regime | **N=100** |
| 总计 | 4 regimes × 100 × 2 = **800** pairs |

初态：\(\beta_0\in\{0°,2°,5°\}\) 等权；\(\gamma_0\) 8 octants。

### 2×2 factorial

| | object static | object move |
|--|---------------|-------------|
| **camera small** | H00 | H01 |
| **camera large** | H10 | H11 |

Camera small：\(\Delta\theta_C\in\{0°,5°\}\)，\(\|\Delta p_C\|\le2\) cm。  
Camera large：\(\Delta\theta_C\in\{20°,30°\}\)，\(\|\Delta p_C\|\in\{4,6\}\) cm。  
Object move：\(\Delta\beta\in\{10,20,35\}°\)，\(\|\Delta p_O\|\in\{0,3,6\}\) cm，\(\Delta\gamma\) 8 octants。

## L1 — Prior anchor（calibration）

\[
\mathrm{median}\,e_{n_0}\le5°,\ P90\le10°;\quad
\mathrm{median}\,e_{p_0}\le5\text{ cm},\ E_{p_0}\le0.20
\]

FAIL → `prior_anchor_failure` → STOP。

## Estimators（冻结）

- **B0** absolute：R6 A1 \(n^{(1)}\) + \(p^{(0)}\)
- **B1** relative：REL0 ICP；传播用 **estimated** \(\hat p_0,\hat n_0\)（非 GT）
- **B_oracle** evaluator-only ceiling
- **B_hyb** non-oracle router

## Router

\[
q_{\mathrm{rel}}=\min(\rho_{0\to1},\rho_{1\to0}),\quad
\tau^*=\arg\max_\tau[\mathrm{TPR}-\mathrm{FPR}]
\]

calibration only；tie → **最大** \(\tau\)；formal **冻结**。

## Gates

- **L2** oracle hybrid：formal 四 regime **分别** axis \(\le15°/30°\) + position gate；任一 FAIL → `hybrid_complementarity_insufficient`
- **L3** formal hybrid：同上四 regime 全 PASS → `hybrid_state_update_supported`

## Patterns

```text
prior_anchor_failure
hybrid_complementarity_insufficient
hybrid_router_failure
hybrid_state_update_insufficient
hybrid_state_update_supported
```

Unlock：`unlocks_hybrid_world_state_branch=true`；`unlocks_o0e1=false`。

## 实现

```text
aprwm_v0/rtwx_o0hyb0.py
aprwm_v0/geometry/relative_validity.py
runs/rtwx_o0hyb0/
```

CLI：`rtwx-o0hyb0`（不暴露 \(\tau\)/ICP 超参）
