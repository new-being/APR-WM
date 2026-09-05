# RTWX-O0G3B 预注册 — Correspondence Instrument Closure

日期：2026-08-30  
状态：**已冻结**  
依赖：O0G3R=`coverage_failure`（描述性 G1 instrument 亦 FAIL）  
**禁止**：fresh scientific generalization；改 \(N_{\min}\) 救 O0G3R；解锁 O0C2/O1；arch sweep。

## 科学问题（仪器，非泛化）

> dense canonical-coordinate predictor 能否至少把 **训练集** 记住？target / representation / optimization 是否闭合？

## 数据

复用 `runs/rtwx_o0g3r/cache_o0g3r_s*_train.npz`（seeds 29601/02/03）；GT mask + GT \(x^O/D_O\)。

## 诊断支路

| ID | 定义 | PASS（train \(E_{\mathrm{corr}}^{norm}<0.05\)） |
|----|------|--------------------------------------------------|
| B0 | index → embedding → \(\hat x^O\)（lookup control） | target pipeline 闭合 |
| B1 | \((u,v,d)\to\hat x^O\) 小 MLP | 像素坐标+depth 可表示 |
| B2 | oracle-mask crop RGBD → \(\hat x^O\)（per-pixel MLP on crop） | 去背景 sparsity 后可学 |
| Audit | \(\phi_i\approx\phi_j\) 但 \(\|x_i^O-x_j^O\|\) 大 | **描述**；\(\ge10\%\) → ambiguity tag |

## Patterns（互斥，优先靠前）

| Pattern | 条件 |
|---------|------|
| `target_pipeline_failure` | ¬B0 |
| `canonical_correspondence_ambiguous` | B0 ∧ audit 高歧义 |
| `pixel_coord_not_representable` | B0 ∧ ¬B1 |
| `crop_rgbd_not_representable` | B0 ∧ B1 ∧ ¬B2 |
| `correspondence_instrument_closed` | B0 ∧ B1 ∧ B2 ∧ ¬audit 高歧义 |

仅 `correspondence_instrument_closed` 允许预注册 **O0G3R2** confirmatory rerun。O0C2/O1 LOCKED。
