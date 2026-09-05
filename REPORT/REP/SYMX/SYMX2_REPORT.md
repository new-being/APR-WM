# SYM-X2 报告 — Symmetry Breaking & Gauge Reactivation

日期：2026-08-30  
状态：**正式冻结** `gauge_reactivation_supported`  
预注册：`REPORT/REG/SYMX/SYMX2_PREREG.md`  
产物：`runs/symx_x2/{summary.json,run.json}`  
依赖：X0=`causal_symmetry_supported`；X1=`quotient_utility_supported`；\(\hat G\) 冻结为 X0 C0 accept；\(\tau=0.04\) **未改**  
Host：`symx_rigid.v1`；**无 RGB**；seeds 42101/02/03  
**X3 / O0G6R / O1 仍 LOCKED。**

## 一句话

物理破缺（惯量 / 外形 / 各向异性摩擦）时，连续 3 个 batch 即可撤销 C0 的 yaw quotient；对称仍成立时 **0/20** 误撤；任务要 yaw 时 **不**撤 world \(G\)，只在 task 坐标上重新看见 \(\gamma\)。  
检测走的是 \(D_H\)，不是残差 MLP 的单步 \(E\)：错 quotient 并没有把 \(E\) 打爆。

\[
\boxed{\texttt{gauge\_reactivation\_supported}}
\]

## 仪器

切换前 3 个 C0 batch；切换后最多 8。每 batch 32 probe，\(H=40\)。  
Sentinel：\(I,\,R_y15,\,R_y90,\,R_x90\)。World 撤销看 \(R_y90\)：连续 \(m=3\) 个 batch \(\widehat D_H>\tau\)。  
Belief：\(b=\exp(-D_H/\tau)\)；档位 \(\{\mathrm{dormant},\mathrm{coarse},\mathrm{active}\}\)（\(D_H\le\tau / \le 4\tau\) / 其余）。Hard rule 冻结；belief 只报。

| ID | 切换 | 预期 |
|----|------|------|
| B1-iner | C0 \(\to\) C4 | 撤 world yaw |
| B1-geom | C0 \(\to\) C1 | 同上 |
| B2-fric | C0 \(\to\) C0fr（\(\mu\) 各向异性） | 同上 |
| B3-task | C5 动力学 + task 要 yaw | **不**撤 world \(G\) |

Recovery：C4 上 stale（继续用 C0 \(\hat G\) 的 \(d_G\)）vs 独立训的 full-state \(E^{B0}\) / 再激活 \(E_{\mathrm{post}}\)。

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-hold | X1 G1 仍成立；切换前 3×C0 \(D_H(R_y90)<\tau\) | **PASS**（\(D_H\sim10^{-9}\)） |
| G1-detect | B1-iner / B1-geom / B2-fric 均 \(T_{\mathrm{revoke}}\le 5\) | **PASS**（**3 / 3 / 3**） |
| G2-false-revoke | C0：\(P<0.05\) | **PASS**（**0/20**） |
| G3-recovery | B1-iner：\(E_{\mathrm{post}}\le 1.10\,E^{B0}\) | **PASS**（**1.003**） |
| G4-task-split | B3 不撤 world；task full yaw \(R^2\ge0.80\)，\(s/\hat G\le0.10\) | **PASS**（full **0.807**；Q **−0.839**） |

## 主表

| 条件 | \(D_H(R_y90)_0\) | \(T_{\mathrm{revoke}}\) | revoke | \(b_0\) | level |
|------|-----------------:|-----------------------:|:------:|--------:|:-----:|
| C0 pre | \(\sim 5\times10^{-9}\) | — | no | \(\approx 1\) | dormant |
| **B1-iner** | **6.44** | **3** | yes | \(\approx 0\) | active |
| **B1-geom** | 4.00 | **3** | yes | \(\approx 0\) | active |
| **B2-fric** | **3.14** | **3** | yes | \(\approx 0\) | active |
| **B3-task** | \(4\times10^{-9}\) | none | **no** | \(\approx 1\) | dormant |

\(D_H(I)\) 全程 \(\sim10^{-9}\)。B3 的 \(D_H(R_x90)\approx5.8\)（与 C0 相同：非 yaw 的 sentinel 仍不对称）。

Recovery（C4，\(d_I\)）：\(E_{\mathrm{stale}}=1.016\)，\(E_{\mathrm{post}}=1.017\)，\(E^{B0}=1.014\)，\(E_{\mathrm{mean}}=1.175\)。  
\(E_{\mathrm{post}}/E^{B0}=\mathbf{1.003}\)；\(E_{\mathrm{stale}}/E^{B0}=\mathbf{1.002}\)（诊断，非门）。

## 读数

1. **撤销是对的、也够快。** 三类物理破缺都在规则允许的最早时刻 \(T=m=3\) 触发。第一批 \(D_H\) 已经是 \(\tau\) 的几十倍，belief 立刻掉到 0、档位 `active`。不是「再观察几百步」。
2. **对偶风险没爆。** C0 上 20 窗 false revoke = 0。B3 动力学仍对称时 world \(G\) 保持 dormant。没有退化成「永远保留 full state」。
3. **摩擦破缺有效。** C0fr 外形/惯量同 C0，只有 \(\mu_x\neq\mu_z\)，\(D_H(R_y90)=3.14\)。方法不是在检测惯量张量。
4. **world \(\neq\) task。** B3 不撤销 world \(\hat G\)（\(D_H(R_y90)\sim10^{-9}\)）；task 用 full \(R\) 能读 yaw（\(R^2=0.807\)，贴门 0.80），用 \(s/\hat G\) 则不能（\(R^2=-0.839\)）。任务要 logo 朝向时，该激活的是 **task-\(\gamma\)**，不是把 world \(G\) 整段扔掉。
5. **G3 按门 PASS，但不要读成「错 quotient 让预测崩了」。** \(E_{\mathrm{stale}}\approx E_{\mathrm{post}}\)。这和 X1 一致：tiny residual MLP 上 yaw 本来就便宜，单步 \(E\) 看不出 C4 的 yaw 因果。**真正把 \(\gamma\) 找回来的是干预 \(D_H\)，不是预测误差上升。** 若工程上只盯 \(E\) 再决定是否 revoke，这一格会漏掉。

## Pattern

```text
pattern = gauge_reactivation_supported
G0_hold = PASS
G1_detect = PASS (T=3,3,3)
G2_false_revoke = PASS (0/20)
G3_recovery = PASS (1.003)
G4_task_split = PASS (world_revoke=false, yaw_full=0.807, yaw_Q=-0.839)
E_stale/E_B0 = 1.002   # diagnostic, not a gate
unlocks_symx3 = false
unlocks_o0g6r = false
unlocks_o1 = false
```

## 解锁

`gauge_reactivation_supported` → 表示合同 [`SYMX_MECHANISM.md`](../../REG/SYMX/SYMX_MECHANISM.md) 已冻结。  
**不**开 SYM-X3 / RGB；**不**开 O0G6R / O1。
