# RTWX-O0RAB0 报告 — Re-anchor Breadth Bakeoff (P0)

日期：2026-09-01  
状态：**RAN / 架构选择完成**；`scientific_result=false`；**winner=none**  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0RAB0_PREREG.md`  
产物：`runs/rtwx_o0rab0/`；BEL0 formal cache seed **37611**（已看过的数据，不作科学确认）

## 一句话

\[
\boxed{
\texttt{winner = none}
\quad
H^*_{B1,B2,B3}\le H^*_{B0}=32
}
\]

三种非 GT periodic re-anchor **都没有超过** no-reset backbone。  
**不跑 RA0。**

## P0 表（lexicographic 输入）

| Branch | K | \(H^*\) | H64 med | H64 P90 | med \(G_n\) | \(P(G_n>0)\) | \(P(G_n<-5^\circ)\) | apply |
|--------|--:|--------:|--------:|--------:|------------:|-------------:|--------------------:|------:|
| **B0** none | — | **32** | 22.3° | 45.7° | — | — | — | — |
| B1 abs | 16 | 8 | 77.4° | 122.0° | **−7.4°** | 0.21 | **0.54** | 1.00 |
| B1 abs | 32 | 16 | 76.8° | 122.0° | **−21.8°** | 0.16 | **0.66** | 1.00 |
| B2 initial | 16 | 16 | 33.1° | 76.2° | 0.0° | 0.29 | 0.17 | 0.64 |
| B2 initial | 32 | 16 | 29.6° | 71.5° | 0.0° | 0.25 | 0.18 | 0.54 |
| B3 rolling | 16 | 16 | 22.2° | 53.6° | 0.0° | 0.32 | 0.10 | 0.63 |
| B3 rolling | 32 | 16 | 28.3° | 58.5° | 0.0° | 0.28 | 0.16 | 0.55 |

Winner 规则要求 \(H^*>H^*_{B0}\)。六格全部 \(\le 32\) → `winner=none`。

## Reset 当场 vs 之后

B1（frozen A1）作为 **reset action 负效用**：

- \(e_n(a=0)\) median **52° / 60°**（K16 / K32）——reset 当场就是坏的
- \(P(G_n<-5^\circ)\) **0.54 / 0.66**——多数时候把更好的 chain 状态重置坏

B2/B3：

- apply rate 仅 **0.54–0.64**（其余 ICP invalid → no-op）
- 成功 reset 的 \(e_n(0)\) 约 **8–11°**，不是当场崩掉
- 但 \(H=32\) 相对 B0 **从 PASS 变成 FAIL**：长 baseline ICP 打断了仍有效的 adjacent chain

B0 本身：从 t=0 起 age 16 的 median 仍约 **7.7°**；H64 才到 22.3°/45.7°。

## Pattern

```text
pattern = reanchor_breadth_selection_complete
scientific_result = false
winner = none
next_frontier = [history_retrieval, persistent_surface]
RA0 = NOT RUN
```

## 架构剪枝（P0，非 fresh 科学结论）

1. **当前 single-frame A1 不适合**作为 relative backbone 的 correction source（比“absolute error 高”更强：reset gain 为负）。
2. **Initial-anchor / rolling keyframe 的固定周期 ICP** 也赢不了 no-reset：它们在 B0 仍过门的 H=32 上把状态打断。
3. BEL0 的 GT periodic reset 有 ceiling，**并不**意味着同一时刻的 non-GT estimator 有效。
4. 按冻结规则：**不开 RA0**；不把 P0 第二名（rolling K16）升格为 formal。

下一层 breadth（需新预注册）才比较：history retrieval vs persistent surface。禁止在本格加 retrieval / 改 K / 修 A1。
