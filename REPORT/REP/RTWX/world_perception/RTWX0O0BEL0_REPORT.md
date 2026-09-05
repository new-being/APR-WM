# RTWX-O0BEL0 报告 — Relative Belief / Re-anchor Observability Audit

日期：2026-09-01  
状态：**正式冻结** `relative_belief_observable`（tag：`trigger_calibration_failed`）  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0BEL0_PREREG.md`  
产物：`runs/rtwx_o0bel0/`；cal **37610** / formal **37611**；100+100 seq × \(T=64\)

## 一句话

\[
\boxed{
\texttt{relative\_belief\_observable}
\quad+\quad
\texttt{trigger\_calibration\_failed}
}
\]

Shadow \(U_t\) **超过纯时间 baseline** 预测未来 4 步失效；**保守 first-trigger 阈值无法校准**。  
**不解锁** `O0RA0`。

## 门控

| Gate | 结果 |
|------|------|
| **L1** SEQ0 replication | **PASS**（\(H\le16\) 全 PASS；formal H16 median **8.17°**，\(\Delta=0.17°\) vs SEQ0 8.0°） |
| **L2** failure support | **PASS**（formal \(r_F=0.90\)；cal \(0.93\)） |
| **L3** observability | **PASS**（cal \(AUROC(U)=0.864\)，\(AUROC(t)=0.806\)，\(\Delta A=0.058\)） |
| Trigger \(\tau_U\) | **FAIL**（无 precision \(\ge 0.90\) 的候选） |

## Formal chain horizon

Formal aggregate \(H^*=32\)（\(H=64\) FAIL）。

| H | Chain med/P90 | pos median | pass |
|---|---------------|------------|------|
| 1 | **3.7°/6.8°** | 2.9 cm | PASS |
| 2 | **4.0°/7.5°** | 3.0 cm | PASS |
| 4 | **4.9°/8.2°** | 3.0 cm | PASS |
| 8 | **6.6°/10.4°** | 3.1 cm | PASS |
| 16 | **8.2°/14.3°** | 3.3 cm | PASS |
| 32 | **12.6°/24.3°** | 4.0 cm | PASS |
| 64 | **21.0°/47.2°** | 11.2 cm | FAIL |

自然 drift 在 **H=32 仍过 formal gate**，到 **H=64 越过**。这不是 replication 失败：H16 与 SEQ0 对齐。

## Observability（cal，\(n=6000\) 步，positive rate \(0.54\)）

| Score | AUROC vs \(y_t^{(4)}\) |
|-------|------------------------|
| \(U_t\)（primary） | **0.864** |
| \(B_{\mathrm{time}}=t\) | 0.806 |
| \(\Delta A\) | **0.058**（门 \(\ge 0.05\)） |
| AUPRC(\(U\)) | 0.900 |
| \(\bar U_t\) | 0.719（**劣于时间**） |
| \(R_t\) recent hazard | 0.747（**劣于时间**） |

信息在 **累积 surprise**，不在瞬时 hazard，也不在去时间平均。  
\(\Delta A\) 刚过门：\(U_t\) 有超越固定周期的 validity 信息，但余量很小。

## Trigger / oracle ceiling

- \(\tau_U^*=\) **null**。precision \(\ge 0.90\) 的 useful-pretrigger（lead \(\in(0,4]\)）集合为空。
- Adaptive GT reset **未执行**（无 trigger）。
- Per-sequence diagnostic \(H^*\) median：no-reset **16**；periodic GT reset \(K\in\{8,16,32\}\) 均为 **64**。

因此：**重新锚定这个动作本身有 ceiling**；**当前 \(U_t\to\) first-trigger 合同写不出可用阈值**。

## Pattern

```text
pattern = relative_belief_observable
tags = [trigger_calibration_failed]
unlocks_sparse_reanchor_prereg = false
unlocks_o0e1 = false
```

## 架构解读

三路剪枝里，本格落在 **C 的弱形式，不是 RA0 绿灯**：

1. Relative backbone 的自然 horizon **超过 SEQ0 的 16**，aggregate 到 **32** 仍过门，**64 才坏**。
2. \(q_{\mathrm{rel}}\) 累积 surprise **可观测** impending failure，且略优于纯时间。
3. 预注册的保守 trigger（precision \(\ge 0.90\)、lead \(\le 4\)）**校准失败**。  
   因此 **不能**把 BEL0 读成“已经有 adaptive re-anchor policy”。
4. Periodic oracle reset \(K=16\) 已把 diagnostic \(H^*\) 拉回 64。  
   若下一格要做 **真实 re-anchor**，优先比较 **periodic / keyframe**，而不是先造复杂 adaptive validity。

禁止：改 ICP、改 \(q_{\mathrm{rel}}\)、事后放宽 precision、用 absolute A1 fallback、把 tag 改成 PASS。

## 下一格（beam）

**不解锁 O0RA0 adaptive sparse re-anchor。**

候选（需新预注册，本格不跑）：

- **periodic re-anchor**（非 GT；\(K\in\{16,32\}\) 有 ceiling）
- 新 trigger 合同（lead 窗口 / sequence-level 定义），不得在本格扫阈值
