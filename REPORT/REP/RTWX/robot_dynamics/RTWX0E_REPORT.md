# RTWX-X0E 报告 — Direct Increment Representation

日期：2026-08-28  
状态：**已跑完**；**`rtwx_x0e_passed=true`**；pattern **`nonlinear_increment_required`**  
预注册：`REPORT/REG/RTWX/RTWX0E_PREREG.md`  
产物：`runs/rtwx_x0e/{run.json,header.json,metrics.json,logs.npz,run.log}`  
本格是 **新族**，不是 X0D 补丁。无容量声称。X0E1 **已解锁资格**（仅 PASS 条件；尚未开跑）。

## 一句话

冻结规模 48/24/24 × 120 native `qpos` 上，**直接拟合 \(\Delta x\)**（禁止 Euler \(\Delta q=\Delta t\dot q\)）可以通过 G0–G3。  
真正过 G1 的是 **M2（固定非线性基 + 线性头）**；M1 线性 increment 在 test 上 \(E_1=0.746\)，门槛为 \(0.75\times 0.992=0.744\)，**差一线未过**。  
因此 pattern 是 **`nonlinear_increment_required`**，不是“纯线性窗口转移已够”。

这支持：X0D 失败主要是 **错误的外部时钟/积分假设**，不是“native window 完全不能被低容量坐标压缩”。速度通道仍然偏弱；位置增量才是这次的增益来源。

## 门限（评分前冻结）

| 门 | 结果 | 细节 |
|----|------|------|
| G0 | **通过** | \(E_1^{M0}\approx 0.99/0.96/0.99\ge 0.02\)；\(\Delta q\) 非退化 |
| G1 | **通过** | test 上仅 M2：\(0.740\le 0.744\)；M1 \(0.746>0.744\) |
| G2 | **通过** | \(k^\star=\) M2；test \(E_{\mathrm{roll}10}=0.647<1.329\)；H50 \(0.723<10\times 1.223\) |
| G3 | **通过** | test \(E_1=0.740\le 1.10\times\) val \(0.720\) |

`unlocks_x0e1=true`；`capacity_claim=false`

## 测试集主指标

| 模型 | \(E_1\) | \(E_1^q\) | \(E_1^{\dot q}\) | \(E_{\mathrm{roll}10}\) | \(E_{\mathrm{roll}50}\) |
|------|---------|-----------|------------------|-------------------------|-------------------------|
| M0 identity | 0.992 | 0.493 | 1.209 | 1.329 | 1.223 |
| M1 linear \(\Delta x\) | 0.746 | 0.235 | 0.938 | 0.651 | 0.732 |
| M2 structured basis | **0.740** | **0.214** | 0.933 | **0.647** | **0.723** |

相对恒等：位置误差从 0.49 降到约 0.21；速度仍约 0.93（与 X0C/X0D 弱速度信息一致）。M2 相对 M1 的一步增益很小，但刚好跨过 0.75 相对门槛。

## 读数（不改门）

1. **Native window increment 是可学的。** 去掉 Euler 后，一步和 H10/H50 都稳定优于恒等，且不爆炸。这与 X0D 的 Euler-\(q\) 失败形成对照。
2. **线性已经很接近门，但预注册判定要 M2。** 不要把 M1 事后说成 PASS。
3. **Train 上 M2 的 \(E_{\mathrm{roll}10}\) 曾到 \(10^{37}\)**，val/test 正常。G2 按预注册看 **test**。这提示该基在个别轨迹上可能局部不稳定；不得据此改门槛，但开 X0E1 时应把 rollout 稳健性列入对照。
4. 仍是 oracle 仿真器状态 / native-qpos 诊断，不是官方观测基准，不是物理参数辨识。

## 账本

```text
RTWX-X0C = FAIL / servo_signal_insufficient
           hidden-force route STOP
RTWX-X0D = FAIL / low_order_structure_insufficient
           Euler closed-loop family STOP
RTWX-X0E = PASS / nonlinear_increment_required
           native-window macro-transition structure supported
           （M1 未过冻结 G1；不得改称“线性已够”）
X0E1     = ELIGIBLE / 本会话开跑
           structured vs latent capacity
X0D1 / TASK-XL / R10 = LOCKED
```
