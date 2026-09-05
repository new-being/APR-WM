# SYM-X2 预注册 — Symmetry Breaking & Gauge Reactivation

日期：2026-08-30  
状态：**正式冻结** `gauge_reactivation_supported`  
依赖：SYM-X1 = `quotient_utility_supported`；X0 \(\tau=0.04\) **不改**  
**禁止**：RGB；改 X0/X1 门；开 X3；O0G6R / O1；R10。

## 科学问题

> 今天被判为无关的 \(\gamma\)（如 C0 yaw），明天突然有因果作用时，系统能否撤销错误 quotient 并重新激活 gauge——而不是永久删掉？

\[
s=(\bar s,\gamma),\qquad \bar s=s/\hat G,\quad \gamma\text{ dormant}\to\text{active when }b_t(g)\downarrow.
\]

第一版用冻结 hard rule（连续 batch 超阈）；同时报 belief \(b_t=\exp(-D_H/\tau)\) 与档位 \(\{\mathrm{dormant},\mathrm{coarse},\mathrm{active}\}\)。  
**任务破缺与物理破缺分开**：不把 task yaw 需求写成撤销 world \(G\)。

## 三类 breaking

| ID | 切换 | 预期 |
|----|------|------|
| **B1-iner** | C0 \(\to\) C4（\(I_x\neq I_z\)） | world \(D_H(R_y90)>\tau\)；撤销 yaw quotient |
| **B1-geom** | C0 \(\to\) C1（handle） | 同上 |
| **B2-fric** | C0 \(\to\) C0fr（\(\mu\) 各向异性，外形/惯量同 C0） | 同上（不只检测惯量） |
| **B3-task** | 动力学仍 C0/C5；任务要 logo/yaw 朝向 | **不**撤销 world \(G\)；只把 task-\(\gamma\) 置 active |

Host：`symx_rigid.v1`。无 RGB：B3 的 task 坐标是物体系 logo 的 yaw（与 C5 相同）。

## 协议

\(\hat G\) 冻结为 X0 的 C0 accept（含 \(R_y\)）。Sentinel：\(I,R_y(15^\circ),R_y(90^\circ),R_x(90^\circ)\)。  
每 batch **32** probe，\(H=40\)。切换前 3 个 C0 batch；切换后最多 8 个 batch。

**撤销（world）**：连续 \(m=3\) 个 batch \(\widehat D_H(R_y90)>\tau=0.04\)。  
\(T_{\mathrm{revoke}}\) = 达到该条件的 batch 数（最少 3）。门：\(\le 5\)。

False revoke：C0 上 20 个长度为 3 的窗，\(P<0.05\)。

## Recovery

物理破缺后，同数据训：

- **stale**：仍用 C0 \(\hat G\) 的 \(d_G\)（错误继续 quotient）
- **reactivated / B0**：\(d\)（\(\gamma\) active）

门：\(E_{\mathrm{post}}\le 1.10\,E^{B0}\)；并报 \(E_{\mathrm{stale}}/E^{B0}\)（错 quotient 是否更差）。

## Gates

| Gate | 条件 |
|------|------|
| G0-hold | X1 G1 仍成立；切换前 3 个 C0 batch \(D_H(R_y90)<\tau\) |
| G1-detect | B1-iner、B1-geom、B2-fric **均** \(T_{\mathrm{revoke}}\le 5\) |
| G2-false-revoke | C0：\(P_{\mathrm{false\ revoke}}<0.05\) |
| G3-recovery | 主声明 B1-iner：\(E_{\mathrm{post}}\le 1.10\,E^{B0}\) |
| G4-task-split | B3-task：**不**撤销 world \(G\)；task 用 full \(R\) 时 yaw \(R^2\ge0.80\)，用 \(s/\hat G\) 则 \(\le0.10\) |

## Patterns

| Pattern | 条件 |
|--------|------|
| `instrument_failure` | ¬G0 |
| `break_undetected` | G0 ∧ ¬G1 |
| `false_revoke` | G0 ∧ ¬G2 |
| `no_recovery` | G0∧G1 ∧ ¬G3 |
| `task_world_collapse` | G0 ∧ ¬G4 |
| `gauge_reactivation_supported` | G0–G4 |

## 解锁

`gauge_reactivation_supported` → 表示合同 [`SYMX_MECHANISM.md`](SYMX_MECHANISM.md) **已冻结**。  
报告：[`../../REP/SYMX/SYMX2_REPORT.md`](../../REP/SYMX/SYMX2_REPORT.md)。  
**X3 / O0G6R / O1 仍 LOCKED。**
