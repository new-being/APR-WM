# RTWX-TASK-X1 报告 — Conditional Diffusion Action-Chunk Breadth Probe

日期：2026-09-01  
状态：**RAN（两段）**  
预注册：`REPORT/REG/RTWX/wam/RTWX0TASKX1_PREREG.md`

1. Stage A（原 \(\epsilon\) ckpt）：`diffusion_chunk_instrument_failure` — 下文。产物 `runs/rtwx_task_x1/`。  
2. Stage B（I1-B2 \(x_0\) ckpt）：`diffusion_chunk_breadth_insufficient` — [`RTWX0TASKX1_STAGEB_REPORT.md`](RTWX0TASKX1_STAGEB_REPORT.md)。产物 `runs/rtwx_task_x1_sb/`。

**未**写入 `runs/rtwx_mr0_p0/`。

## Stage A（\(\epsilon\)，历史）

\[
\boxed{\texttt{diffusion\_chunk\_instrument\_failure}}
\]

当时还不能写 `diffusion_chunk_breadth_insufficient`：没有尺度合法的可部署 chunk。  
B1 训满 100 epoch，val \(L_{\rm diff}\approx 0.155\)，\(n_\theta=247968\)。  
G0/G1/G3/G4 PASS；**G2 FAIL**。采样 L1 ≈ 160，\(D_{\rm pred}\approx 787\) vs \(D_{\rm demo}\approx 0.043\)。  
未 clip。闭合见 I0；参数化修复见 I1。

## Stage B（合法 \(x_0\) 仪器之后）

\[
\boxed{\texttt{diffusion\_chunk\_breadth\_insufficient}}
\]

B0 = AC0 MLP，B1 = I1-B2 sample/\(x_0\)；seeds 52601，各 48 幕：**双方 pooled = 0**。详见 Stage B 报告。
