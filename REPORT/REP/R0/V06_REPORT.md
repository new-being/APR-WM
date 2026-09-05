# APR-WM V0.6: Compute-valid Adaptive Routing

## Executive conclusion

V0.6 provides a controlled, single-seed proof that learned residual routing can produce real CUDA speedup **after** an analytic interaction filter, but only in a measurable compute-valid region.

The strongest defensible conclusion is:

> Residual necessity remains accurately rankable when physical interaction density and residual necessity are separated. On this RTX 5070, learned hard routing becomes faster than the contact-conditioned baseline only when enough active edges are processed together to amortize router, top-k, packing, and kernel-launch overhead. Statistical sparsity alone is not sufficient.

For the representative regime with 80% active interactions and a 10% residual budget, the accurate `current` router reaches AUROC `0.9938`. With batch 512, measured speedup over the already-filtered `contact_packed` baseline rises from `1.484×` for the 1× expert to `3.890×` for the 8× expert. Batch 1 and 16 never break even, and the 8× expert at batch 128 remains borderline at `0.979×` in the longer confirmation run.

Thus H2 is **conditionally supported**, not universally established:

\[
(1-p)C_E > C_R + \frac{C_O}{N}
\]

The data exhibit all three terms: lower residual activation `p`, larger expert cost `C_E`, and more active work `N` favor routing; a small batch leaves sparse overhead dominant.

## 1. Controlled benchmark

The benchmark deliberately separates two events:

1. an interaction is physically active (Stage 0 analytic mask);
2. the active interaction enters a state-dependent nonlinear regime and needs a residual (Stage 1 learned router).

Interaction density and conditional residual necessity are sampled independently. No object ID or edge ID is exposed. Nonlinear necessity depends on the current extension, relative velocity, friction, and phase, so an interaction can switch between simple and residual regimes over time. The requested residual budget is supplied as a continuous regime threshold, allowing one router to operate across multiple budgets.

The residual target is zero in the simple regime and contains nonlinear stiffness/damping-like corrections in the complex regime. This is an edge-level compute benchmark rather than a full autoregressive world-model rollout.

## 2. Experimental protocol

The full matrix contains:

| Variable | Values |
|---|---|
| Active interaction density | 20%, 50%, 80%, 100% |
| Residual necessity among active edges | 5%, 10%, 20%, 40% |
| Residual expert | 1×, 2×, 4×, 8× actual FLOPs |
| Router | linear, tiny MLP, current MLP |
| World batch | 1, 16, 128, 512 |
| Execution | dense-all, contact-packed, masked-dense, hard-sparse |

The nominal expert labels track actual arithmetic within 3%:

| Expert | Parameters | FLOPs / active edge | Relative FLOPs |
|---|---:|---:|---:|
| 1× | 11,138 | 21,120 | 1.000× |
| 2× | 22,402 | 43,120 | 2.042× |
| 4× | 44,002 | 85,600 | 4.053× |
| 8× | 86,338 | 169,264 | 8.014× |

Router sizes are 13 parameters / 24 FLOPs for `linear`, 225 / 416 for `tiny`, and 513 / 832 for `current`.

The primary latency baseline is `contact_packed`: the residual expert runs on every Stage-0 active edge. `hard_sparse` additionally runs the router, exact-budget top-k, selected-row packing, and the expert only on routed rows. Stage-0 packing is common to both paths and is performed before the timed closure; this isolates whether learned routing adds value beyond the analytic filter. The full sweep uses 30 warm-ups and 12 timing samples, each averaging 10 forwards. The central regime is separately confirmed with 100 samples × 10 forwards.

## 3. Router accuracy is not interchangeable with router price

At 10% residual necessity, evaluated with the 8× expert and averaged over interaction densities:

| Router | AUROC | Necessity recall at exact 10% budget | Adaptive MSE − contact MSE |
|---|---:|---:|---:|
| Linear | 0.6231 | 0.175 | +0.008272 |
| Tiny | 0.8346 | 0.693 | +0.001592 |
| Current | 0.9942 | 0.863 | −0.000065 |

The linear and tiny routers sometimes report greater raw speedup, but they do not preserve accuracy and therefore are not valid substitutes. The 513-parameter `current` router is still cheap relative to every residual expert and is the only tested router that consistently supplies a high-quality routing signal.

For the current router with the 8× expert:

| Residual budget | AUROC | Recall | Contact MSE | Adaptive MSE | ΔMSE |
|---|---:|---:|---:|---:|---:|
| 5% | 0.9974 | 0.875 | 0.000533 | 0.000326 | −0.000207 |
| 10% | 0.9942 | 0.863 | 0.000721 | 0.000656 | −0.000065 |
| 20% | 0.9867 | 0.864 | 0.001139 | 0.001395 | +0.000257 |
| 40% | 0.9814 | 0.903 | 0.002562 | 0.003185 | +0.000623 |

Negative ΔMSE at 5–10% is possible because the dense expert emits small erroneous corrections on simple edges, while routing suppresses them. At 20–40%, absolute ΔMSE remains small but relative degradation is about 23–24%; these regimes should not be described as equal-accuracy without a task-specific tolerance.

## 4. Confirmed compute crossover

The longer confirmation run uses 80% active interactions, a 10% residual budget, and the accurate current router. Speedup is contact latency divided by hard-sparse latency:

| Batch | 1× expert | 2× expert | 4× expert | 8× expert |
|---:|---:|---:|---:|---:|
| 1 | 0.329× | 0.366× | 0.339× | 0.326× |
| 16 | 0.300× | 0.290× | 0.304× | 0.314× |
| 128 | 0.553× | 0.517× | 0.729× | 0.979× |
| 512 | **1.484×** | **2.079×** | **2.627×** | **3.890×** |

Batch-512 absolute medians are:

| Expert | Contact-packed | Hard-sparse | Speedup | Peak memory: contact → sparse |
|---|---:|---:|---:|---:|
| 1× | 1.108 ms | 0.746 ms | 1.484× | 56.8 → 30.7 MiB |
| 2× | 1.709 ms | 0.822 ms | 2.079× | 74.4 → 30.7 MiB |
| 4× | 2.354 ms | 0.896 ms | 2.627× | 99.1 → 30.7 MiB |
| 8× | 3.516 ms | 0.904 ms | 3.890× | 133.6 → 30.7 MiB |

At this exact-budget operating point, adaptive error is no worse than the contact-conditioned expert for all four expert sizes. The expert-dependent ΔMSE values are `−0.000218`, `−0.000194`, `−0.000119`, and `−0.000069` from 1× through 8×.

The full batch-512 sweep also shows the expected `C_O/N` boundary. At 10% necessity, full-sweep speedups for the current router are:

| Interaction density | 1× | 2× | 4× | 8× |
|---:|---:|---:|---:|---:|
| 20% | 0.36× | 0.53× | 0.96× | 1.46× |
| 50% | 0.91× | 1.00× | 2.21× | 2.81× |
| 80% | 1.47× | 2.01× | 2.87× | 4.50× |
| 100% | 1.91× | 2.69× | 3.20× | 4.22× |

The 80% row is independently confirmed above; other cells use the shorter full-matrix timing and should be read as regime indicators rather than precise benchmarks. Low interaction density leaves too few active rows to amortize sparse overhead unless the expert is large.

## 5. Analytical savings versus measured savings

With the current router and `p=0.1`, ignoring sparse overhead predicts speedups

\[
\frac{C_E}{C_R+pC_E}
\approx
7.17,\ 8.38,\ 9.11,\ 9.53
\]

for 1× through 8×. Measured batch-512 speedups are only `1.48–3.89×`, and smaller batches are slower than the baseline. The gap directly quantifies top-k, indexing, launch, and small-GEMM costs. Analytical FLOPs are useful for locating a possible region, but they cannot replace device timing.

Across the full 768 contact-versus-sparse comparisons, 191 are faster, but this count includes inaccurate linear/tiny routers and short timing samples. It is not an accuracy-controlled headline metric. The validated result is the high-AUROC current-router crossover shown by the longer central-regime run.

## 6. What V0.6 supports

Supported in this controlled benchmark:

1. dense physical interaction and sparse residual necessity can coexist;
2. necessity is predictable from current interaction state without object/edge identity;
3. learned routing can beat a contact-conditioned residual baseline in wall-clock CUDA latency;
4. the crossover moves systematically with active work, residual fraction, and expert cost;
5. cheap-but-inaccurate routers cannot be credited as compute-valid routing.

Not established yet:

1. multi-seed confidence intervals for the V0.6 crossover;
2. rollout-level accuracy and stability under dynamic routing;
3. routing learned from residual utility rather than the controlled regime label;
4. end-to-end speedup including state encoding and Stage-0 interaction discovery;
5. energy savings, custom fused sparse kernels, or transfer to realistic embodied dynamics.

The next defensible step before treating H2 as a paper-level result is to repeat the accurate-router crossover over multiple seeds and move its supervision from the synthetic necessity label to measured residual utility. V1 can then address unknown physics/system identification without conflating that problem with unresolved compute economics.

## Artifacts

- Full matrix: [`accuracy_matrix.csv`](runs/v06/core/accuracy_matrix.csv), [`latency_matrix.csv`](runs/v06/core/latency_matrix.csv), and [`break_even.csv`](runs/v06/core/break_even.csv)
- Long confirmation: [`break_even.csv`](runs/v06/confirm/break_even.csv) and [`latency_matrix.csv`](runs/v06/confirm/latency_matrix.csv)
- Figures: full-matrix [`expert_crossover.png`](runs/v06/core/expert_crossover.png), longer-run [`expert_crossover.png`](runs/v06/confirm/expert_crossover.png), and [`accuracy_regimes.png`](runs/v06/core/accuracy_regimes.png)
- Configuration summary: [`summary.json`](runs/v06/core/summary.json)
