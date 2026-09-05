# RTWX-TASK-X1 预注册 — Conditional Diffusion Action-Chunk Breadth Probe

日期：2026-09-01  
状态：**已冻结**  
依赖：AC0-B0 权重与 `split_manifest.json`  
**禁止**：改 split；process；history；RGB；CVAE；纠正数据；扫 \(K\)/DDIM steps；把权重写入 `runs/rtwx_mr0_p0/`；用 rollout 选 ckpt；修改旧 MR0-P0 合同

## 假说（弱）

\[
H_{X1}:\text{同样的显式状态与专家数据下，动作块生成式建模可能比点回归更适合当前闭环。}
\]

**不**声称“动作是多模态的所以 diffusion 会成功”。

## 唯一 \(\Delta\)

B0 = 冻结 AC0-B0 MLP；B1 = 扁平 MLP 去噪器 + cosine \(K_{\rm train}=100\) + DDIM \(N=10,\eta=0\)。  
输入 \((s_t,z_g)\)，输出 \(8\times 14\) `joint_position`。\(K_a=1\) 只执行 \(A_t[0]\)。  
一次 **planner call** = 10 次 denoiser forward ≠ 旧 P0 的 single-forward。

## 训练

复用 AC0 normalizer 与 episode split。ckpt **只**按 val \(L_{\rm diff}\)（\(\epsilon\)-MSE）。无辅助 loss。

## Stage A 仪器门

G0 shape；G1 finite；G2 反归一化落在 demo 范围 \(\pm 0.5\)；G3 同 seed \(\max|\Delta A|<10^{-6}\)；G4 非 repeat-fill。  
**无** \(L_1<0.015\) 门。

## Stage B（architecture selection）

seeds **52601**；\(N=16\)/task；B0/B1 各 48 幕 CRN。  
`scientific_result=false`，`purpose=model_family_selection`。

- \(S_{\rm pooled}<0.20\) 或未达 \(\Delta\ge 0.15\) / 至少两任务 success>0 / \(\Delta\)safety\(\le 0.02\)  
  → `diffusion_chunk_breadth_insufficient`，`winner=none`，剪掉 diffusion，不调 steps/Transformer/history。
- 否则 `diffusion_chunk_breadth_supported` → 另格 **TASK-X1-C0**（seeds 52701，\(N=32\)），**不是**改 MR0-P0。
