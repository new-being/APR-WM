# RTWX RoboTwin 账本（native-qpos / macro-transition 线）

日期：2026-08-28  
本文件为 **只增不改** 的正式 ledger 摘要；各格详细报告见 `REPORT/REP/RTWX/`。

```text
RTWX-X0C   = FAIL / servo_signal_insufficient
RTWX-X0D   = FAIL / low_order_structure_insufficient
RTWX-X0E   = PASS / nonlinear_increment_required
RTWX-X0E1  = completed / structure_not_in_robust_set (open-loop contract)
RTWX-X0ES  = RAN / pathology_not_reproduced
RTWX-X0ES1 = RAN / rare_instability_intrinsic_map
RTWX-X0ES2 = RAN / instability_persists (50-step OL contraction STOP)
RTWX-X0EH  = RAN / short_horizon_sufficient
             H_plan=16, K_execute=4 deployment contract frozen
RTWX-X0EH1 = PASS / receding_structure_capacity_shift
             R_P^(K=4) = 0.988
             C_P^(K=4) ≈ 84.0×
             P_struct = 1,752
             P_NN^min = 147,224 (H=256, K=4 matched)
             deployment compute ≈ 76× lower (M2 vs H=256)
             source task: put_object_cabinet; seed 12601
RTWX-X0EH2 = PASS / cross_task_structure_supported
             holdouts: place_empty_cup (13601), stamp_seal (13602)
             R_P^(K=4) = 0.988 per holdout (C_P^(K=4) ≈ 84.0×)
             frozen φ; per-task W; K=4 contract unchanged
             cross_task_capacity_claim = true
RTWX-X0RGB = RAN / perception_failure
             sim RGB (front_camera 64×64, L=2, seed 14601)
             G_oracle=true; G_vis=false (E_q=1.018, E_qd=1.029)
             B0 E_exec^(K=4)=0.653; B1 E_exec=0.615; r_cat=0
             does not claim visual interface success; not real camera
RTWX-X0RGB1 = RAN / spatial_perception_failure
             sim RGB 128×128 CNN + L=4; seed 15601
             P2 E_q=0.968 E_qd=0.992; n_q_ok=0/12
             G_oracle=true; B0 E_exec^(K=4)=0.610
             128 CNN does not recover 12-DoF q; G_vis unchanged
             SUBLINE CLOSED: RGB → robot (q,qd)  [negative result]

RTWX-O0    = RAN / object_pose_failure
             place_empty_cup; seed 16601; 224×224 ResNet18; L=4
             G0=true; G1=false (E_p=0.536); G2=false (P90 e_R=118.7°)
             G3=false; G_info=false; B2 ≈ B0
             WORLD branch; does not unlock O1; not X0RGB2

RTWX-O0D   = RAN / perception_instrument_failure
             D1 E_p_mem=0.645 = B0 on 256 O0-train frames
             D0_vis=false (median cup mask=0); D0_track=true
             D2/D3 fail; D3 not interpretable (empty masks)
             no observability claim; O1 still LOCKED

RTWX-O0D1  = RAN / scalar_memorization_failure
             R1 nrmse(p_x)=0.146 (not mean collapse); random-label nrmse=0.87
             R2 skipped; R3 P_visible=0.333 (= in_FOV); occluded_in_fov=0
             cup id=66 matches camera when in FOV; front_camera FOV hole
             no O0R; O1 LOCKED

RTWX-O0D2  = RAN / image_memorization_failure
             A lookup PASS (~1e-7); B flatten true=0.110 rand=0.395 FAIL
             CoordConv p_x=0.011 PASS; GAP=0.631 FAIL
             FOV: head/observer/world=1.0; front=0.167; c*=head_camera
             O0R LOCKED (B not closed)

RTWX-O0D3  = RAN / instrument_qualified
             head_camera; CoordConv no-GAP; 64^2; seed 20601; 8x32 collect
             G0 FOV=vis=1.0; G1 px=0.0087; G2 rand=0.00097
             G3 E_p=0.0032; G4 med_eR=0.16deg P90=0.28deg
             unlocks O0R prereg; does NOT run O0R; O1 LOCKED

O0R       = RAN / object_pose_failure
             head_camera + CoordConv no-GAP 64^2; seed 21601
             G0 FOV=vis=0.962 PASS; G_ori PASS (med 9.3° P90 14.7°)
             G_pos FAIL Ep=0.343; G_info FAIL (~0.98×B0); G_vel FAIL
             unlocks_o1=false

RTWX-O0G  = RAN / geometry_insufficient
             head_camera; seed 22601; 8×32; stage g01
             G0 oracle (u,v,z_C)→p_B PASS (~1e-16)
             G1 mask+depth centroid FAIL (med 5.57cm > 5cm)
             audit: +5.1cm z bias (pose.p vs surface); xy~1.4cm
             G2 STOP; unlocks_o1=false

RTWX-O0G1b = RAN / geometry_reference_aligned
             head_camera; seed 23601; 8×32; no learning
             δ_O = 021_cup model_data0 center*scale (NOT +5.11cm fit)
             B0 surface FAIL med 5.42cm; B1 aligned PASS med 1.65cm (strong)
             unlocks_g2_prereg=true; unlocks_o1=false

RTWX-O0G2 = RAN / coverage_failure
             head_camera 64² U-Net; seed 24601; 48/24/24×120
             G0 FAIL FOV=vis=0.902 (<0.95)
             G1–G4 PASS descriptive: B2≈B1 Ep=0.049 med=1.62cm; IoU≈1; 0.05×B0
             unlocks_o0c=false; unlocks_o1=false
             conditional RGBD→position supported; availability NOT

RTWX-O0V  = RAN / dual_view_coverage_supported
             head∨observer; seeds 25601/02/03; 24×120; no train
             B0 head agg vis=0.889 unstable (all seeds <0.95)
             B1 any agg=0.996; per-seed 1.000/1.000/0.987 PASS
             unlocks_o0g2r_prereg=true; unlocks_o1=false

RTWX-O0G2R = RAN / dual_view_geometry_position_supported
             head+observer; seed 26601; 48/24/24×120; shared U-Net; median fusion
             G0 any=1.000; B1/B2 Ep=0.150 med=1.44cm strong; η≈1; 0.43×B0
             localization_near_oracle=true; unlocks_o0c_prereg=true; unlocks_o1=false
             R_BO=GT nuisance only

RTWX-O0C  = RAN / orientation_failure
             head+observer; seeds 27601/02/03; 24/12/12×120; R̂=CoordConv R_CO→GeoMedian
             G0 PASS; G1 FAIL (med eR=12.8° but P90=90.7°>30°); unlocks_o1=false
             B3 Ep=0.272 med=1.91cm; ΔE_R≈5e-4; ori_beats_mean=false
             position near-oracle remains; orientation heavy-tail / mode-confusion

RTWX-O0C1 = RAN / dual_view_both_wrong_mode (diagnostic)
             D1: ~9.6% near 90° (not 180°); fusion med=14.9° P90=90.0°
             D2: cata 86.4% = h✗o✗; fusion✗=0; e_best P90=82° (not fusion-select)
             D4: C4-Y resolves only 9% cata → not symmetry_mismatch
             D3: d_view correlates with failure (uncertainty signal; no τ rescue)
             unlocks_o1=false; does_not_rewrite_o0c=true

RTWX-O0G3 = RAN / oracle_orientation_geometry_supported
             G0 only; seeds 28601/02/03; 24×120; oracle x^O + Kabsch; no train
             med=P90=0; P([75,105])=0; residual~1e-16; unlocks_o0g3r_prereg=true
             unlocks_o1=false

RTWX-O0G3R = RAN / coverage_failure
             seeds 29601/02/03; 24/12/12×120; RGBD corr U-Net + Kabsch; B0/B1/B2
             G0 FAIL: P_enough=0.875, joint_det=0.941; G1 instrument FAIL E_corr=0.517
             unlocks_o0c2_prereg=false; unlocks_o1=false

RTWX-O0G3B = RAN / pixel_coord_not_representable
             train memorization only; B0 lookup PASS (0.0027); B1 uvd FAIL (0.30); B2 crop FAIL (0.48)
             target pipeline OK; dense x^O path not closed; unlocks_o0g3r2=false

RTWX-O0G3A = RAN / correspondence_availability_characterized
             deficit 12.5% frames; median n_union=8 when deficit; mask area <~87px → P_enough=0

RTWX-O0G4 = RAN / sparse_keypoints_unobservable
             G0 only; 4 frozen CAD kps; ori PASS cov FAIL P_enough=0.694
             unlocks_o0g4r_prereg=false

RTWX-O0G5A = RAN / local_descriptor_ambiguous
             128²; FPFH no-train; seeds 31601/02/03; 12×120
             G0 PASS P_support=0.973; G2 FAIL med e_norm=8.00 top5≈0
             unlocks_o0g5r_prereg=false; unlocks_o1=false

RTWX-O0G5B = RAN / global_canonical_identity_supported
             instrument only; K=512 FPS; B0 acc=1; B1 med=0.045 top5=0.83
             B2 med=0.049 top5=0.80; B1≈B2 (no global>>local on train)
             unlocks_o0g5c_prereg=true; unlocks_o0g5r=false

RTWX-O0G5C = RAN / canonical_identity_generalization_failure
             fresh 33601/02/03; G0 PASS 0.915; B1 med=0.119 top5=0.609
             B2 med=0.119 top5=0.596; B1≈B2; uv-shuffle top5=0.446
             unlocks_o0g5r_prereg=false; unlocks_o1=false

RTWX-O0G6A = RAN / global_geometry_ambiguous
             same 128² 33601/02/03; no ICP; GT fit RMS=0.021 D_O
             P_top1=0.478 FAIL; Top5=1.0; margin≈0
             Ry90/180 ratio≈1.00; Rx/Rz90 ratio≈462
             unlocks_o0g6r_prereg=false; identity branch CLOSED

SYM-X0     = RAN / causal_symmetry_supported
             oracle rigid; no RGB in D_H; seeds 40101/02/03; τ=0.04
             G0–G4 PASS; false quotient 0/289; recall=1.000
             C4 rejects nontriv yaw (D_H Ry90=8.20); C5=C0 accept set
             unlocks_symx1_prereg=true; X2/X3/O0G6R/O1 LOCKED

SYM-X1     = RAN / quotient_utility_supported
             Ĝ frozen from X0; TF one-step E; W*=32
             G0=0.759; G1 B2/B0=1.006; G2=1.000; G3 C4=0.998
             G4 C0 yaw R^2=-0.033 axis=0.829; B3 yaw=0.390 (no spontaneous quotient)
             R_P(C0)=-1.09 (not a gate); unlocks_symx2_prereg=true
             X3 / O0G6R / O1 LOCKED

SYM-X2     = RAN / gauge_reactivation_supported
             seeds 42101/02/03; τ=0.04; m=3; no RGB
             G1 T_revoke=3/3/3 (iner/geom/fric); G2 false revoke 0/20
             G3 E_post/E_B0=1.003; E_stale/E_B0=1.002 (diagnostic)
             G4 B3 world not revoked; task yaw R^2 full=0.807 Q=-0.839
             unlocks_symx3=false; O0G6R / O1 LOCKED

RTWX-O0Q0  = RAN / counterfactual_instrument_failure
             seeds 45101/02/03; no RGB; τ=0.05
             G0 FAIL D_H(I)_max=236.6 (P2/P3 EE-teleport)
             P1 table D_H~0.005 accept-all (B2 not excited)
             P0 air B1 med D_H=1.13 (diagnostic only)
             unlocks_o0q1=false; O0 target unchanged

RTWX-O0Q0R0 = RAN / counterfactual_restore_failure
             official seeds 0/1/2; no teleport; no yaw science
             G0 FAIL D_H(I)_max=10.27; P2=0 P3=2
             P1 identity=0, friction probe D_H~0.16
             I_x≈I_z (rel 0.78%); unlocks_o0q0r1=false

RTWX-O0Q0R0B = RAN / precontact_branch_failure
             G0 PASS (P0 H=6 D_H(I)~1e-4; A* identity closed)
             G1 FAIL P2=P3=0 (IK failed; min_d~0.27)
             P1 friction probe 0.16; P0 probe 0.036<τ
             unlocks_o0q0r1=false

RTWX-O0Q0R0C = RAN / native_demo_action_replay_failure
             G0 PASS; G3 PASS (I_x/I_z=4 D_H~0.48–0.68); G4 PASS (P1 0.16)
             G1 FAIL P2=P3=0; hdf5 qa via take_action min_d~0.21–0.27
             H_P0=6 frozen; qa[t]=q[t+1] (shifted obs, not independent cmd)
             unlocks_o0q0r1=false

RTWX-O0Q0R0D = RAN / capture_incomplete
             G0 PASS sapien_joint_drive_target; take_action=0 on expert
             G1 FAIL play_once 0/3 (cuRobo missing); prefix 300 gripper steps only
             hdf5 joint_action = save_freq=15 drive sample, not dense A*
             unlocks_o0q0r1=false

RTWX-O0Q0R0E = RAN / dense_trace_unavailable
             path A FAIL (no curobo); path B FAIL (no _traj_data pkl)
             no mplib substitute; P0/P1 not reopened
             unlocks_o0q0r1=false

O0Q*       = PAUSED / infrastructure-blocked
             no causal quotient conclusion; O0 target stays full (T,R)
             do not burn mainline cells on dense-trace acquisition

RTWX-O0A0  = RAN / appearance_yaw_generalization_failure
             G0 PASS; B1 RGB fresh med=90° P90=165° top1=0.042
             train CE = ln(24); B0/B1/B2 all chance; RGB 1-NN top1=0.109
             formal = no-GAP run only; O0A1 LOCKED

O0T0       = PREREGISTERED / NOT RUN / DEPRIORITIZED
             canonical yaw budget stopped; not FAIL

RTWX-O0E0  = RAN / observation_support_failure
             NATURAL distribution insufficient to TEST O0E0
             NOT a claim that RGBD cannot recover (p,n)
             B2 effective axis = UNTESTED
             P(V_any)=0.966 < 0.98; STOP before UNet/B2
             G0b diagnostic: r_excite=0.126 (constant-up / low-entropy axis)

O0E0P0     = RAN / controlled_pose_observation_qualified
             G0a P_any=1.000; G0b r_excite=0.600; G0c 8/8 octants
             static set_pose; same head+observer; B2 NOT RUN
             unlocks O0E0R0 science on P0 cache only
             not a natural-task axis claim

O0E0R0     = RAN / effective_axis_failure
             B2 first test FAIL; B0 excluded; oracle ceil ep=9.9cm

O0E0R1     = RAN / quotient_viable_under_oracle_cloud
             ceil GTmask+GTp=0.17°; dominant=learned seg corruption

O0E0R2     = RAN / segmentation_cloud_failure
             S0 instrument PASS; S1 FAIL (observer IoU=0.46); H_seg rejected

O0E0R3     = RAN / reference_coupled_failure
             fresh seed 37603; L2 G_I1=0.15° PASS; L3 formal axis=22.4° FAIL
             position formal PASS (3.55cm); reference/centering = formal blocker
             S0 cloud stable; B2 objective still frozen/viable

O0E0R4     = RAN / multiview_reference_insufficient
             mv seed 37604; K∈{1,2,4,8} naive union; B2 frozen
             center-λ diag: e_center=3.50cm → axis=28.1°; λ=0.5 → 14.6° PASS
             K=1 sanity 20.5°; K=8 med 19.7° still FAIL (P90=84.9°)
             position PASS all K; naive multiview does NOT close reference
             unlocks_o0e1=false; O1 LOCKED

O0E0R5     = RAN / visibility_reference_joint_failure
             seed 37605; only swap fixed delta_y -> CAD delta_vis(n,T_BC)
             G_I1 oracle-axis PASS: ep=1.83cm axis=14.6deg
             L3 formal vis med=16.6deg FAIL (old=21.1deg); position PASS 2.12cm
             pattern B: reference model viable; joint center-axis coupling blocks
             unlocks_o0e1=false; O1 LOCKED

O0E0R6     = RAN / joint_inference_insufficient
             reuse R5 seed 37605; inference-only; A0 sanity delta=0.21deg
             A0 med=16.85 P90=129; A1 n1 med=17.1 P90=33; A1 n2 med=20.0 (degradation)
             A2 med=18.9 P90=31.3; D_ho AUROC=0.37 (no score weight)
             unlocks_o0e1=false

O0E0R7     = PREREG / PAUSED (beam frontier; not auto-NEXT)
             persistent surface probe retained on frontier

O0REL0     = RAN / ego_motion_compensation_failure
             seed 37606; C1 4.5deg/0.66cm C2 4.7deg/0.74cm PASS; C0 L2 FAIL rho=0.04
             vs A1 ~20deg; formal pattern unchanged

O0REL0A    = RAN / overlap_support_hypothesis_supported
             zero-train audit; overlap-gated validity mechanism confirmed

O0HYB0     = RAN / hybrid_state_update_insufficient
             relative-only 4/4 PASS; absolute ~22deg; router closed

O0SEQ0     = RAN / relative_sequence_supported (stable_to_16)
             seed 37609; N=100 T=16; H*=16 chain all horizons PASS
             H16 chain 8.0/16.1deg pos=3.3cm; absolute ~31deg FAIL all H
             unlocks_relative_sequence_branch=true

O0BEL0     = RAN / relative_belief_observable (trigger_calibration_failed)
             cal 37610 / formal 37611; N=100+100 T=64
             L1 H<=16 PASS; H16 formal 8.17deg; H*=32; H64 FAIL 21.0/47.2deg
             AUROC(U)=0.864 vs time 0.806, dA=0.058; tau_U=null
             unlocks_sparse_reanchor_prereg=false

O0RAB0     = RAN / P0 winner=none (scientific_result=false)
             BEL0 cache 37611; B0 H*=32; B1 abs RESET HURTS (med Gn -7/-22)
             B2/B3 H*=16 < B0; RA0 NOT RUN

O0MEMB0    = RAN / P0 winner=none (scientific_result=false)
             B0/B1 H*=32; B2 surface H*=16; retrieval ages ~P(A=1)=0.51
             stop_world_perception_local_dfs

RTWX-MR0   = RAN / action_chunk_contract_failure
             Ha=0 (no frozen chunk policy); 7-cell task sweep NOT RUN
             G0-S stride ICP valid Ks=2/4/8: 0.772/0.753/0.725 (all <0.80)
             hold-last-qpos forbidden; Ks*/Ka* undefined
             infrastructure block, not science no-go on fs>fa

RTWX-MR0-P0 = RAN / action_chunk_contract_failure (instrument)
             unlocks_mr0_a=false; G1/G2/G3 skipped; no training this cell

RTWX-MR0-A = RAN-BLOCKED / mr0_a_blocked_unqualified_policy
             4-cell Ka sweep not executed; HA untested

RTWX-AC1-D0 = RAN / state_support_metric_invalid
             G1 FAIL (cabinet expert test r_OOD=0.206); G2 AUROC=0.994; G3 p_early=1.0
             covariate shift plausible_unlocalized; unlocks_ac1_r0=false

RTWX-AC3 = RAN / native_expert_replay_failure
             _traj_data missing 24/24 held-out; R0 not run
             D0: Q3 velocity absent; Q4 next_index (e_next=0); no raw hdf5

RTWX-MA0 = RAN / official_act_stack_unqualified
             P0: next_state_as_action, da=14, freq=15
             P1 official ACT 6000 + eval.sh: cup 3/16=0.188 < G0=0.20
             nonzero_closed_loop=true (seeds 100007/100010/100015)
             C0/MA1 locked

RTWX-TASK-X1 = RAN / diffusion_chunk_instrument_failure then breadth_insufficient
             Stage A G2 FAIL on epsilon (L1~160); I1 repaired x0 instrument
             Stage B seeds 52601: B0=0/48 B1(x0-diff)=0/48 safety=0
             winner=none; stop_diffusion_local_dfs; no C0; did not write MR0-P0

RTWX-TASK-X1-I0 = RAN / diffusion_denoiser_insufficient
             C0/C1 oracle-eps PASS; C2 t=99 L1_norm=558 (t=10/50 PASS)
             sampler contract OK; same ckpt; no clip; no Stage B

RTWX-TASK-X1-I1 = RAN / epsilon_parameterization_pathology_supported
             B0 eps t=99 L1_norm=558 FAIL; B1 v and B2 x0 rec+Stage A PASS
             winner=B2 (L1_99=0.026 vs v 0.126); not complexity-tie to v
             Stage B subsequently RAN on this ckpt (see TASK-X1 Stage B)

## 机制链收束（至 X0EH1）

```text
force reconstruction failed (X0C)
→ Euler macro model failed (X0D)
→ direct increment succeeded (X0E/M2)
→ rare open-loop instability (X0ES1: ρ_J>1)
→ pointwise contraction insufficient for 50-step OL (X0ES2)
→ K=4 receding reset sufficient (X0EH)
→ deployment-relevant capacity shift (X0EH1)
→ cross-task external validity (X0EH2)
→ sim RGB perception failure (X0RGB)
→ spatial CNN still fails q recovery (X0RGB1); RGB→s^R CLOSED
→ RGB→s^O ResNet18 collapse to train-mean (O0); O1 LOCKED
```

## 论文主结果资格

**X0EH1** 是当前 RoboTwin 线上第一个可进入论文主结果表的 **deployment-contract capacity** 结果（source task）。  
**X0EH2** 将同一结论扩展到两个 holdout manipulation 任务，支持 **cross-task structural compression** 假说。  
禁止事后：扩 width、微调 M2 basis、改 matched threshold、追更大 compression ratio。

## 下一格

**RGB→\((q,\dot q)\) 子线关闭。** O0 = `object_pose_failure`，**O1 LOCKED**。  
WORLD 线若继续，需新预注册（仍问 \(RGB\to s^O\) 仪器，不是滤波/dynamics）。不得改 M2。R10 / TASK-XL LOCKED。

---

## 目录拆分（2026-08-29，只增）

**RTWX** 仍是项目内命名（RoboTwin X-series），不是论文术语。科学问题切断如下，**不**再开 `X0RGB2`：

```text
REPORT/{REG,REP}/RTWX/
  robot_dynamics/                 # X0C–X0EH2, X0S/F/R, … 文件名未改
    rgb_robot_probe/              # X0RGB, X0RGB1 CLOSED; 文件名未改
  world_perception/               # O0, O1, …  NEW WORLD branch
  object_dynamics/                # D0, I0（尚未开格）
  attention_compute/             # A0–U0（尚未开格）
  ledger/                         # 本账本
```

```text
RTWX-X0RGB* = CLOSED   robot-state-from-RGB probe
RTWX-O0+    = NEW      RGB → s^O  (world_perception)
```

旧扁平路径保留指针文件。

O0 正式 RAN：`object_pose_failure`。报告：`REPORT/REP/RTWX/world_perception/RTWX0O0_REPORT.md`。
