from __future__ import annotations

import argparse
import json

import torch

from .config import ExperimentConfig, load_config
from .evaluate import evaluate_model
from .models import MODEL_NAMES, build_model, parameter_count
from .simulator import SyntheticWorld
from .train import load_checkpoint, resolve_device, run_suite, summarize_existing, train_one
from .v05 import (
    ablation_experiment,
    aggregate_seeds,
    continuum_experiment,
    horizon_experiment,
    latency_experiment,
    pareto_experiment,
    seed_experiment,
)
from .v06 import V06Config, run_v06
from .v07 import V07Config, run_v07
from .v1 import V1Config, run_v1
from .v2 import V2Config, run_v2
from .v3 import V3Config, run_v3
from .v4 import V4Config, run_v4
from .v5 import V5Config, run_v5
from .v6 import V6Config, run_v6
from .r0 import V6R0Config, inspect_robotwin, run_v6r0_smoke
from .r01 import V6R01Config, run_v6r01_closure
from .r02 import V6R02Config, run_v6r02
from .r03 import V6R03Config, run_v6r03
from .r04a import V6R04AConfig, run_v6r04a
from .r04b import V6R04BConfig, run_v6r04b
from .r05 import V6R05Config, run_v6r05
from .r05p import V6R05PConfig, run_v6r05p
from .r06 import V6R06Config, run_v6r06


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "device", None) is not None:
        config.device = args.device
    if getattr(args, "output", None) is not None:
        config.output_dir = args.output
    if getattr(args, "steps", None) is not None:
        config.train.steps = args.steps
    if getattr(args, "batch_size", None) is not None:
        config.train.batch_size = args.batch_size
    config.validate()
    return config


def doctor(config: ExperimentConfig) -> dict[str, object]:
    device = resolve_device(config.device)
    nodes = config.train.n_objects
    edges = nodes * (nodes - 1)
    effective_batch = config.train.batch_size * config.train.trajectory_steps
    edge_values = effective_batch * edges * (
        2 * config.model.hidden_dim + 2 * 33
    )
    rough_activation_mb = edge_values * 4 * 8 / (1024**2)
    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "effective_transition_batch": effective_batch,
        "directed_edges_per_world": edges,
        "rough_training_activation_mb": round(rough_activation_mb, 1),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        architectures = torch.cuda.get_arch_list()
        capability_name = f"sm_{capability[0]}{capability[1]}"
        report.update(
            {
                "gpu": properties.name,
                "gpu_memory_mb": round(properties.total_memory / (1024**2)),
                "compute_capability": capability_name,
                "wheel_architectures": architectures,
                "native_arch_supported": capability_name in architectures,
            }
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aprwm-v0", description="APR-WM V0 physics/residual competition experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check CUDA and the 8GB memory profile")
    doctor_parser.add_argument("--config", default="configs/rtx5070_8gb.json")
    doctor_parser.add_argument("--device", default=None)

    train_parser = subparsers.add_parser("train", help="Train one baseline")
    train_parser.add_argument("--config", default="configs/rtx5070_8gb.json")
    train_parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--output", default=None)
    train_parser.add_argument("--steps", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)

    suite_parser = subparsers.add_parser("suite", help="Run all four baselines sequentially")
    suite_parser.add_argument("--config", default="configs/rtx5070_8gb.json")
    suite_parser.add_argument("--device", default=None)
    suite_parser.add_argument("--output", default=None)
    suite_parser.add_argument("--steps", type=int, default=None)
    suite_parser.add_argument("--batch-size", type=int, default=None)

    summary_parser = subparsers.add_parser(
        "summarize", help="Rebuild suite summaries from existing model metrics"
    )
    summary_parser.add_argument("--output", default="runs/v0_rtx5070")

    pareto_parser = subparsers.add_parser("v05-pareto", help="Run routing Pareto curves")
    pareto_parser.add_argument("--checkpoint", required=True)
    pareto_parser.add_argument("--output", default="runs/v05/core")
    pareto_parser.add_argument("--device", default="auto")
    pareto_parser.add_argument("--batches", type=int, default=4)
    pareto_parser.add_argument("--batch-size", type=int, default=128)
    pareto_parser.add_argument("--horizon", type=int, default=20)

    horizon_parser = subparsers.add_parser("v05-horizon", help="Run error-vs-horizon curves")
    horizon_parser.add_argument("--checkpoints", nargs="+", required=True)
    horizon_parser.add_argument("--output", default="runs/v05/core")
    horizon_parser.add_argument("--device", default="auto")

    continuum_parser = subparsers.add_parser(
        "v05-continuum", help="Sweep interaction complexity with fixed material identity"
    )
    continuum_parser.add_argument("--checkpoint", required=True)
    continuum_parser.add_argument("--output", default="runs/v05/core")
    continuum_parser.add_argument("--device", default="auto")
    continuum_parser.add_argument("--grid-size", type=int, default=25)

    latency_parser = subparsers.add_parser(
        "v05-latency", help="Measure real CUDA conditional-execution latency"
    )
    latency_parser.add_argument("--checkpoints", nargs="+", required=True)
    latency_parser.add_argument("--output", default="runs/v05/core")
    latency_parser.add_argument("--device", default="auto")
    latency_parser.add_argument("--iterations", type=int, default=200)

    seeds_parser = subparsers.add_parser("v05-seeds", help="Run and aggregate multiple seeds")
    seeds_parser.add_argument("--config", default="configs/rtx5070_8gb.json")
    seeds_parser.add_argument("--output", default="runs/v05/seeds")
    seeds_parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])

    aggregate_parser = subparsers.add_parser(
        "v05-aggregate", help="Aggregate already-completed seed suites"
    )
    aggregate_parser.add_argument("--output", default="runs/v05/seeds")

    ablation_parser = subparsers.add_parser(
        "v05-ablation", help="Run router warm-up/distillation/routing ablations"
    )
    ablation_parser.add_argument("--config", default="configs/rtx5070_8gb.json")
    ablation_parser.add_argument("--output", default="runs/v05/ablations")
    ablation_parser.add_argument("--seed", type=int, default=7)

    v06_parser = subparsers.add_parser(
        "v06", help="Run compute-valid dense-interaction routing benchmark"
    )
    v06_parser.add_argument("--output", default="runs/v06/core")
    v06_parser.add_argument("--device", default="auto")
    v06_parser.add_argument("--full-matrix", action="store_true")
    v06_parser.add_argument("--reuse-models", action="store_true")
    v06_parser.add_argument("--train-steps", type=int, default=1500)
    v06_parser.add_argument("--latency-iterations", type=int, default=30)

    v07_parser = subparsers.add_parser(
        "v07", help="Run five-seed residual-utility routing validation"
    )
    v07_parser.add_argument("--output", default="runs/v07/core")
    v07_parser.add_argument("--device", default="auto")
    v07_parser.add_argument("--seeds", nargs="+", type=int, default=[13, 23, 33, 43, 53])
    v07_parser.add_argument("--expert-train-steps", type=int, default=1200)
    v07_parser.add_argument("--router-train-steps", type=int, default=1000)
    v07_parser.add_argument("--latency-iterations", type=int, default=100)
    v07_parser.add_argument("--reuse-models", action="store_true")

    v1_parser = subparsers.add_parser(
        "v1", help="Run unknown-physics posterior and residual-misuse validation"
    )
    v1_parser.add_argument("--output", default="runs/v1/core")
    v1_parser.add_argument("--device", default="auto")
    v1_parser.add_argument("--seeds", nargs="+", type=int, default=[13, 23, 33, 43, 53])
    v1_parser.add_argument("--expert-train-steps", type=int, default=1200)
    v1_parser.add_argument("--router-train-steps", type=int, default=1000)

    v2_parser = subparsers.add_parser(
        "v2", help="Run misspecification-aware active probing experiment"
    )
    v2_parser.add_argument("--output", default="runs/v2/core")
    v2_parser.add_argument("--device", default="auto")
    v2_parser.add_argument("--seeds", nargs="+", type=int, default=[13, 23, 33, 43, 53])
    v2_parser.add_argument("--episodes", type=int, default=4096)
    v2_parser.add_argument("--probes", type=int, default=2)

    v3_parser = subparsers.add_parser(
        "v3", help="Run tangent-triggered open-set model-revision experiment"
    )
    v3_parser.add_argument("--output", default="runs/v3/core")
    v3_parser.add_argument("--device", default="auto")
    v3_parser.add_argument("--seeds", nargs="+", type=int, default=[101, 111, 121, 131, 141])
    v3_parser.add_argument("--episodes", type=int, default=4096)

    v4_parser = subparsers.add_parser(
        "v4", help="Run residual-guided active operator-discovery experiment"
    )
    v4_parser.add_argument("--output", default="runs/v4/core")
    v4_parser.add_argument("--device", default="auto")
    v4_parser.add_argument("--seeds", nargs="+", type=int, default=[201, 211, 221, 231, 241])
    v4_parser.add_argument("--episodes", type=int, default=4096)

    v5_parser = subparsers.add_parser(
        "v5", help="Run noisy sequential operator-discrimination experiment"
    )
    v5_parser.add_argument("--output", default="runs/v5/core")
    v5_parser.add_argument("--device", default="auto")
    v5_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[301, 311, 321, 331, 341]
    )
    v5_parser.add_argument("--episodes", type=int, default=2048)
    v5_parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.025, 0.05, 0.10, 0.15],
    )

    v6_parser = subparsers.add_parser(
        "v6", help="Run adaptive model-revision validation experiment"
    )
    v6_parser.add_argument("--output", default="runs/v6/core")
    v6_parser.add_argument("--device", default="auto")
    v6_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[401, 411, 421, 431, 441]
    )
    v6_parser.add_argument("--episodes", type=int, default=2048)
    v6_parser.add_argument("--validation-max", type=int, default=32)
    v6_parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.025, 0.05, 0.10, 0.15],
    )

    r0_doctor_parser = subparsers.add_parser(
        "v6r0-doctor", help="Check RoboTwin/SAPIEN readiness and task limitations"
    )
    r0_doctor_parser.add_argument(
        "--robotwin-repo", default="/home/dong/Projects/RoboTwin"
    )
    r0_doctor_parser.add_argument(
        "--robotwin-python",
        default="/home/dong/miniconda3/envs/RoboTwin/bin/python",
    )

    r0_parser = subparsers.add_parser(
        "v6r0-smoke", help="Run the three-seed controlled SAPIEN hinge bridge"
    )
    r0_parser.add_argument("--output", default="runs/v6r0/smoke")
    r0_parser.add_argument("--device", default="cpu")
    r0_parser.add_argument("--seeds", nargs="+", type=int, default=[13, 23, 33])
    r0_parser.add_argument("--episodes-per-regime", type=int, default=24)
    r0_parser.add_argument(
        "--robotwin-repo", default="/home/dong/Projects/RoboTwin"
    )

    r01_parser = subparsers.add_parser(
        "v6r01-closure",
        help="Validate the SAPIEN hinge interface in generalized-force coordinates",
    )
    r01_parser.add_argument("--output", default="runs/v6r01/closure")
    r01_parser.add_argument("--device", default="cpu")
    r01_parser.add_argument("--seeds", nargs="+", type=int, default=[13, 23, 33])
    r01_parser.add_argument("--episodes", type=int, default=24)
    r01_parser.add_argument("--samples-per-episode", type=int, default=48)

    r02_parser = subparsers.add_parser(
        "v6r02",
        help="Reintegrate frozen V6 in generalized-force coordinates",
    )
    r02_parser.add_argument("--output", default="runs/v6r02/smoke")
    r02_parser.add_argument("--device", default="cpu")
    r02_parser.add_argument("--seeds", nargs="+", type=int, default=[13, 23, 33])
    r02_parser.add_argument("--episodes-per-regime", type=int, default=24)

    r03_parser = subparsers.add_parser(
        "v6r03",
        help="Run held-out C0/C1/C2/native SAPIEN bridge evaluation",
    )
    r03_parser.add_argument("--output", default="runs/v6r03/formal")
    r03_parser.add_argument("--device", default="cpu")
    r03_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1001, 1011, 1021, 1031, 1041]
    )
    r03_parser.add_argument("--episodes-per-regime", type=int, default=24)

    r04a_parser = subparsers.add_parser(
        "v6r04a",
        help="Run rollout-aware native revision acceptance experiment",
    )
    r04a_parser.add_argument("--output", default="runs/v6r04a/confirmatory")
    r04a_parser.add_argument("--device", default="cpu")
    r04a_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[4001, 4011, 4021, 4031, 4041]
    )
    r04a_parser.add_argument("--episodes", type=int, default=24)

    r04b_parser = subparsers.add_parser(
        "v6r04b",
        help="Run history-aware C2 residual fallback experiment",
    )
    r04b_parser.add_argument("--output", default="runs/v6r04b/formal")
    r04b_parser.add_argument("--device", default="cpu")
    r04b_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[3001, 3011, 3021, 3031, 3041]
    )
    r04b_parser.add_argument("--episodes", type=int, default=24)

    r05_parser = subparsers.add_parser(
        "v6r05",
        help="Rank dynamical safety diagnostics on R0.4 revision episodes",
    )
    r05_parser.add_argument("--output", default="runs/v6r05/diagnostics")
    r05_parser.add_argument("--device", default="cpu")
    r05_parser.add_argument(
        "--selection-seeds", nargs="+", type=int,
        default=[2001, 2011, 2021, 2031, 2041],
    )
    r05_parser.add_argument(
        "--confirmation-seeds", nargs="+", type=int,
        default=[4001, 4011, 4021, 4031, 4041],
    )
    r05_parser.add_argument("--episodes", type=int, default=24)

    r05p_parser = subparsers.add_parser(
        "v6r05p",
        help="Run operator-controlled passivity falsification benchmark",
    )
    r05p_parser.add_argument("--output", default="runs/v6r05p/formal")
    r05p_parser.add_argument("--device", default="cpu")
    r05p_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[5001, 5011, 5021, 5031, 5041]
    )
    r05p_parser.add_argument("--episodes", type=int, default=24)

    r06_parser = subparsers.add_parser(
        "v6r06",
        help="Run passivity-feasible then rollout-utility revision",
    )
    r06_parser.add_argument("--output", default="runs/v6r06/confirmatory")
    r06_parser.add_argument("--device", default="cpu")
    r06_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[7001, 7011, 7021, 7031, 7041]
    )
    r06_parser.add_argument("--episodes", type=int, default=24)

    r1_mj0_parser = subparsers.add_parser(
        "r1-mj0",
        help="Run R1-MJ0 native MuJoCo single-hinge force-space closure",
    )
    r1_mj0_parser.add_argument("--output", default="runs/r1_mj0/closure")
    r1_mj0_parser.add_argument("--duration", type=float, default=10.0)

    r1_rs0_parser = subparsers.add_parser(
        "r1-rs0",
        help="Run R1-RS0 robosuite Door Mode-A C0 (requires passing MJ0)",
    )
    r1_rs0_parser.add_argument("--output", default="runs/r1_rs0/c0")
    r1_rs0_parser.add_argument(
        "--mj0-summary",
        default="runs/r1_mj0/closure/summary.json",
        help="Path to MJ0 summary.json; RS0 stays locked unless mj0_passed",
    )
    r1_rs0_parser.add_argument("--duration", type=float, default=10.0)

    r1_rs1_parser = subparsers.add_parser(
        "r1-rs1",
        help="Run R1-RS1 frozen R0.6 Door Mode-A scientific transfer",
    )
    r1_rs1_parser.add_argument("--output", default="runs/r1_rs1/formal")
    r1_rs1_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
        help="Path to RS0 summary.json; RS1 stays locked unless rs0_passed",
    )
    r1_rs1_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 8901 x 5 regimes x P1); no scientific gates",
    )

    r1_rs1a_parser = subparsers.add_parser(
        "r1-rs1a",
        help="Run R1-RS1A multi-evidence inadequacy detection (trigger-only)",
    )
    r1_rs1a_parser.add_argument("--output", default="runs/r1_rs1a/formal")
    r1_rs1a_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
        help="Path to RS0 summary.json",
    )
    r1_rs1a_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 8951 x 5 regimes x P1); no ROC gates",
    )

    r1_rs1a1_parser = subparsers.add_parser(
        "r1-rs1a1",
        help="Run R1-RS1A.1 compositional inadequacy evidence (D3'/Dg)",
    )
    r1_rs1a1_parser.add_argument("--output", default="runs/r1_rs1a1/formal")
    r1_rs1a1_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1a1_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 8961 x 5 x P1)",
    )

    r1_rs1a2_parser = subparsers.add_parser(
        "r1-rs1a2",
        help="Run R1-RS1A.2 weak structural signal detection",
    )
    r1_rs1a2_parser.add_argument("--output", default="runs/r1_rs1a2/formal")
    r1_rs1a2_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1a2_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 8971 x 5 x P1)",
    )

    r1_rs1a3_parser = subparsers.add_parser(
        "r1-rs1a3",
        help="Run R1-RS1A.3 excitation-limited identifiability",
    )
    r1_rs1a3_parser.add_argument("--output", default="runs/r1_rs1a3/formal")
    r1_rs1a3_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1a3_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 8981 x C0/C1-L x A0,f=0.4)",
    )

    r1_rs1a4_parser = subparsers.add_parser(
        "r1-rs1a4",
        help="Run R1-RS1A.4 detectability vs consequence",
    )
    r1_rs1a4_parser.add_argument("--output", default="runs/r1_rs1a4/formal")
    r1_rs1a4_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1a4_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 8991 x alpha in {0,-0.12} x A0)",
    )

    r1_rs1a5_parser = subparsers.add_parser(
        "r1-rs1a5",
        help="Run R1-RS1A.5 value-of-epistemic-excitation (probe population)",
    )
    r1_rs1a5_parser.add_argument("--output", default="runs/r1_rs1a5/formal")
    r1_rs1a5_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1a5_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 9001)",
    )

    r1_rs1b_parser = subparsers.add_parser(
        "r1-rs1b",
        help="Run long-horizon admissibility on revise-worthy Door revisions",
    )
    r1_rs1b_parser.add_argument("--output", default="runs/r1_rs1b/formal")
    r1_rs1b_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1b_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
        help="Passing RS1A.5 summary containing the frozen intake policy",
    )
    r1_rs1b_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 9011 x alpha=-0.24 x 1.5A0)",
    )

    r1_rs1b1_parser = subparsers.add_parser(
        "r1-rs1b1",
        help="Calibrate modeled-support departure against H32 prediction risk",
    )
    r1_rs1b1_parser.add_argument("--output", default="runs/r1_rs1b1/formal")
    r1_rs1b1_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1b1_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs1b1_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 9021 x alpha=-0.24 x 1.5A0)",
    )

    r1_rs1b2_parser = subparsers.add_parser(
        "r1-rs1b2",
        help="Identify whether intermediate support-departure bands can be populated",
    )
    r1_rs1b2_parser.add_argument("--output", default="runs/r1_rs1b2/formal")
    r1_rs1b2_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1b2_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs1b2_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 9031 x alpha=-0.24 x 1.5A0)",
    )

    r1_rs1c_parser = subparsers.add_parser(
        "r1-rs1c",
        help="Run R1-RS1C layered epistemic-physical policy (new hypothesis)",
    )
    r1_rs1c_parser.add_argument("--output", default="runs/r1_rs1c/formal")
    r1_rs1c_parser.add_argument(
        "--rs0-summary",
        default="runs/r1_rs0/c0/summary.json",
    )
    r1_rs1c_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs1c_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 9041 x {0,-0.24} x {0.5,1.5}A0)",
    )

    r1_rs2_c0_parser = subparsers.add_parser(
        "r1-rs2-c0",
        help="Run oracle J^T f robot-contact closure before RS2 formal",
    )
    r1_rs2_c0_parser.add_argument("--output", default="runs/r1_rs2/c0")
    r1_rs2_c0_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
        help="Passing RS1A.5 artifact with frozen exposure-cell thresholds",
    )
    r1_rs2_c0_parser.add_argument(
        "--smoke",
        action="store_true",
        help="One-cell contact plumbing smoke; does not evaluate C0 gates",
    )

    r1_rs2_parser = subparsers.add_parser(
        "r1-rs2",
        help="Run frozen RS1C policy under robot-contact excitation (RS2 Formal)",
    )
    r1_rs2_parser.add_argument("--output", default="runs/r1_rs2/formal")
    r1_rs2_parser.add_argument(
        "--rs2-c0-summary",
        default="runs/r1_rs2/c0_v2/summary.json",
        help="Passing RS2-C0 summary; Formal stays locked unless rs2_c0_pass",
    )
    r1_rs2_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs2_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 11101 x C0/C1-H x slow_pull); no GO gates",
    )

    r1_rs2a_parser = subparsers.add_parser(
        "r1-rs2a",
        help="Diagnose contact vs Mode-A tangent-visible exposure (no policy retune)",
    )
    r1_rs2a_parser.add_argument("--output", default="runs/r1_rs2a/formal")
    r1_rs2a_parser.add_argument(
        "--rs2-formal-summary",
        default="runs/r1_rs2/formal/summary.json",
        help="Completed RS2 Formal summary; RS2A stays locked until scientific_result",
    )
    r1_rs2a_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs2a_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (Mode-A 1.5A0 + contact fast_pull); no H1–H3",
    )

    r1_rs2b_parser = subparsers.add_parser(
        "r1-rs2b",
        help="Diagnose Mode-A vs contact consequence transport (no policy retune)",
    )
    r1_rs2b_parser.add_argument("--output", default="runs/r1_rs2b/formal")
    r1_rs2b_parser.add_argument(
        "--rs2a-summary",
        default="runs/r1_rs2a/formal/summary.json",
        help="Completed RS2A summary; RS2B stays locked until scientific_result",
    )
    r1_rs2b_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs2b_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (seed 11101 x C0/C2); no H0–H2",
    )

    r1_rs3a_parser = subparsers.add_parser(
        "r1-rs3a",
        help="Learner-visible transport-calibrated structural evidence (no policy)",
    )
    r1_rs3a_parser.add_argument("--output", default="runs/r1_rs3a/formal")
    r1_rs3a_parser.add_argument(
        "--rs2b-summary",
        default="runs/r1_rs2b/formal/summary.json",
        help="Completed RS2B summary; RS3A stays locked until scientific_result",
    )
    r1_rs3a_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs3a_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (Mode-A 1.5A0 + contact fast_pull); no H1–H2",
    )

    r1_rs3a1_parser = subparsers.add_parser(
        "r1-rs3a1",
        help="Cross-fitted structural excess-risk evidence (no policy, no S_perp repair)",
    )
    r1_rs3a1_parser.add_argument("--output", default="runs/r1_rs3a1/formal")
    r1_rs3a1_parser.add_argument(
        "--rs3a-summary",
        default="runs/r1_rs3a/formal/summary.json",
        help="Completed RS3A summary; RS3A.1 stays locked until scientific_result",
    )
    r1_rs3a1_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs3a1_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (Mode-A 1.5A0 + contact fast_pull); no H1–H2",
    )

    r1_rs4a_parser = subparsers.add_parser(
        "r1-rs4a",
        help="Geometry-conditioned structural evidence (LOIO; no domain ID, no scalar repair)",
    )
    r1_rs4a_parser.add_argument("--output", default="runs/r1_rs4a/formal")
    r1_rs4a_parser.add_argument(
        "--rs3a1-summary",
        default="runs/r1_rs3a1/formal/summary.json",
        help="Completed RS3A.1 summary; RS4A stays locked until scientific_result",
    )
    r1_rs4a_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs4a_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only (Mode-A + fast_pull + pull_push); no LOIO GO",
    )

    r1_rs5a_parser = subparsers.add_parser(
        "r1-rs5a",
        help="Intervention-indexed calibration with abstention (no revision / no RS4A.1)",
    )
    r1_rs5a_parser.add_argument("--output", default="runs/r1_rs5a/formal")
    r1_rs5a_parser.add_argument(
        "--rs4a-summary",
        default="runs/r1_rs4a/formal/summary.json",
        help="Completed RS4A summary; RS5A stays locked until scientific_result",
    )
    r1_rs5a_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r1_rs5a_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no H1–H4",
    )

    r3_v7a_parser = subparsers.add_parser(
        "r3-v7a",
        help="Mixture-trained contextual epistemic belief (not RS5B; not old V7 routing)",
    )
    r3_v7a_parser.add_argument("--output", default="runs/r3_v7a/formal")
    r3_v7a_parser.add_argument(
        "--rs5a-summary",
        default="runs/r1_rs5a/formal/summary.json",
        help="Completed RS5A summary; R3-V7A stays locked until scientific_result",
    )
    r3_v7a_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7a_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no mixture GO",
    )

    r3_v7b_parser = subparsers.add_parser(
        "r3-v7b",
        help="Persistent contextual epistemic belief vs V7A static summary",
    )
    r3_v7b_parser.add_argument("--output", default="runs/r3_v7b/formal")
    r3_v7b_parser.add_argument(
        "--v7a-summary",
        default="runs/r3_v7a/formal/summary.json",
        help="Completed V7A summary; R3-V7B stays locked until v7a_go",
    )
    r3_v7b_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7b_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no H1–H3",
    )

    r3_v7b1_parser = subparsers.add_parser(
        "r3-v7b1",
        help="Same GRU as V7B; IBS objective vs time-weighted BCE (no V7C)",
    )
    r3_v7b1_parser.add_argument("--output", default="runs/r3_v7b1/formal")
    r3_v7b1_parser.add_argument(
        "--v7b-summary",
        default="runs/r3_v7b/formal/summary.json",
        help="Completed V7B scientific summary; GO may be false",
    )
    r3_v7b1_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7b1_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no IBS GO",
    )

    r3_v7b2_parser = subparsers.add_parser(
        "r3-v7b2",
        help="Fast/slow epistemic state vs single GRU and capacity-matched GRU",
    )
    r3_v7b2_parser.add_argument("--output", default="runs/r3_v7b2/formal")
    r3_v7b2_parser.add_argument(
        "--v7b1-summary",
        default="runs/r3_v7b1/formal/summary.json",
        help="Completed V7B.1 scientific summary; GO may be false",
    )
    r3_v7b2_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7b2_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no fast/slow GO",
    )

    r3_v7b3_parser = subparsers.add_parser(
        "r3-v7b3",
        help="Matched C0/C1 evidence-warranted belief vs B2 BCE (no V7C)",
    )
    r3_v7b3_parser.add_argument("--output", default="runs/r3_v7b3/formal")
    r3_v7b3_parser.add_argument(
        "--v7b2-summary",
        default="runs/r3_v7b2/formal/summary.json",
    )
    r3_v7b3_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7b3_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no warranted GO",
    )

    r3_v7c_parser = subparsers.add_parser(
        "r3-v7c",
        help="B5 GRU: oracle vs sensorized vs no contact (no V7D)",
    )
    r3_v7c_parser.add_argument("--output", default="runs/r3_v7c/formal")
    r3_v7c_parser.add_argument(
        "--v7b3-summary",
        default="runs/r3_v7b3/formal/summary.json",
    )
    r3_v7c_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7c_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no V7C GO",
    )

    r3_v7c1_parser = subparsers.add_parser(
        "r3-v7c1",
        help="Learned tactile z vs B5-S warranted belief (no V7D)",
    )
    r3_v7c1_parser.add_argument("--output", default="runs/r3_v7c1/formal")
    r3_v7c1_parser.add_argument(
        "--v7c-summary",
        default="runs/r3_v7c/formal/summary.json",
    )
    r3_v7c1_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7c1_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no V7C.1 GO",
    )

    r3_v7c2_parser = subparsers.add_parser(
        "r3-v7c2",
        help="Tactile field/encoder/fusion information locus (diagnostic; no V7D)",
    )
    r3_v7c2_parser.add_argument("--output", default="runs/r3_v7c2/formal")
    r3_v7c2_parser.add_argument(
        "--v7c1-summary",
        default="runs/r3_v7c1/formal/summary.json",
    )
    r3_v7c2_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7c2_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no locus classification",
    )

    r3_v7c3_parser = subparsers.add_parser(
        "r3-v7c3",
        help="Tactile observation sufficiency ablation (no encoder / no V7D)",
    )
    r3_v7c3_parser.add_argument("--output", default="runs/r3_v7c3/formal")
    r3_v7c3_parser.add_argument(
        "--v7c2-summary",
        default="runs/r3_v7c2/formal/summary.json",
    )
    r3_v7c3_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7c3_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no observation deltas",
    )

    r3_v7c4_parser = subparsers.add_parser(
        "r3-v7c4",
        help="NSG representation sufficiency (no B5 / no V7D)",
    )
    r3_v7c4_parser.add_argument("--output", default="runs/r3_v7c4/formal")
    r3_v7c4_parser.add_argument(
        "--v7c3-summary",
        default="runs/r3_v7c3/formal/summary.json",
    )
    r3_v7c4_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7c4_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no representation trichotomy",
    )

    r3_v7c5_parser = subparsers.add_parser(
        "r3-v7c5",
        help="B5-S + Z_NSG belief integration (no V7D)",
    )
    r3_v7c5_parser.add_argument("--output", default="runs/r3_v7c5/formal")
    r3_v7c5_parser.add_argument(
        "--v7c4-summary",
        default="runs/r3_v7c4/formal/summary.json",
    )
    r3_v7c5_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7c5_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no belief-integration GO",
    )

    r3_v7d_parser = subparsers.add_parser(
        "r3-v7d",
        help="B5-S + Z_RGB conditional modality value (frozen encoder)",
    )
    r3_v7d_parser.add_argument("--output", default="runs/r3_v7d/formal")
    r3_v7d_parser.add_argument(
        "--v7c5-summary",
        default="runs/r3_v7c5/formal/summary.json",
    )
    r3_v7d_parser.add_argument(
        "--rs1a5-summary",
        default="runs/r1_rs1a5/formal/summary.json",
    )
    r3_v7d_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no conditional-usefulness GO",
    )

    r4_i0_parser = subparsers.add_parser(
        "r4-i0",
        help="R4 contact-regime closure (infrastructure; not C0 GO)",
    )
    r4_i0_parser.add_argument("--output", default="runs/r4_i0/v2")
    r4_i1_parser = subparsers.add_parser(
        "r4-i1",
        help="R4 no-leak audit of h^S (infrastructure; not C0 GO)",
    )
    r4_i1_parser.add_argument("--output", default="runs/r4_i1/formal")
    r4_i1_parser.add_argument(
        "--i0-summary",
        default="runs/r4_i0/v2/summary.json",
    )

    r4_c0_parser = subparsers.add_parser(
        "r4-c0",
        help="R4 observation necessity of tactile given h^S",
    )
    r4_c0_parser.add_argument("--output", default="runs/r4_c0/formal")
    r4_c0_parser.add_argument(
        "--i0-summary",
        default="runs/r4_i0/v2/summary.json",
    )
    r4_c0_parser.add_argument(
        "--i1-summary",
        default="runs/r4_i1/formal/summary.json",
    )
    r4_c0_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Plumbing smoke only; no conditional-usefulness GO",
    )

    r4_c1_parser = subparsers.add_parser(
        "r4-c1",
        help="R4 consequence-warranted local epistemic value",
    )
    r4_c1_parser.add_argument("--output", default="runs/r4_c1/formal")
    r4_c1_parser.add_argument(
        "--i0-summary",
        default="runs/r4_i0/v2/summary.json",
    )
    r4_c1_parser.add_argument(
        "--i1-summary",
        default="runs/r4_i1/formal/summary.json",
    )
    r4_c1_parser.add_argument(
        "--c0-summary",
        default="runs/r4_c0/formal/summary.json",
    )

    r4_c2_parser = subparsers.add_parser(
        "r4-c2",
        help="R4 continuous future-consequence epistemic value",
    )
    r4_c2_parser.add_argument("--output", default="runs/r4_c2/formal")
    r4_c2_parser.add_argument(
        "--i0-summary",
        default="runs/r4_i0/v2/summary.json",
    )
    r4_c2_parser.add_argument(
        "--i1-summary",
        default="runs/r4_i1/formal/summary.json",
    )
    r4_c2_parser.add_argument(
        "--c1-summary",
        default="runs/r4_c1/formal/summary.json",
    )

    r5_i0_parser = subparsers.add_parser(
        "r5-i0",
        help="R5 physical feasibility of consequence-varying lambda (no neural probe)",
    )
    r5_i0_parser.add_argument("--output", default="runs/r5_i0/kt")
    r5_i0_parser.add_argument(
        "--mechanism",
        choices=("kt", "prestress", "margin"),
        default="kt",
        help="kt=I0 v3 integral-constrained solref; prestress=v2; margin=v1",
    )

    r5_ss_parser = subparsers.add_parser(
        "r5-i0-selfstress",
        help="R5 self-stress geometric-stiffness I0 (no neural probe)",
    )
    r5_ss_parser.add_argument("--output", default="runs/r5_i0_selfstress/formal")

    r5_i1_ss_parser = subparsers.add_parser(
        "r5-i1-selfstress",
        help="R5 self-stress I1: X predicts Y_future given h^S (ridge, no Adam)",
    )
    r5_i1_ss_parser.add_argument("--output", default="runs/r5_i1_selfstress/formal")
    r5_i1_ss_parser.add_argument(
        "--i0-summary",
        default="runs/r5_i0_selfstress/formal/summary.json",
    )

    r5_d0_pf_parser = subparsers.add_parser(
        "r5-d0-preflight",
        help="R5 self-stress D0 oracle action-ranking preflight (no planner)",
    )
    r5_d0_pf_parser.add_argument("--output", default="runs/r5_d0_preflight/formal")
    r5_d0_pf_parser.add_argument(
        "--i1-summary",
        default="runs/r5_i1_selfstress/formal/summary.json",
    )

    r5_d0_parser = subparsers.add_parser(
        "r5-d0",
        help="R5-D0 executed regret: h^S planner vs (h^S, X) world-model planner",
    )
    r5_d0_parser.add_argument("--output", default="runs/r5_d0/formal")
    r5_d0_parser.add_argument(
        "--preflight-summary",
        default="runs/r5_d0_preflight/formal/summary.json",
    )

    r6_a0_parser = subparsers.add_parser(
        "r6-a0",
        help="R6-A0 frozen D0 action-margin / cost-error audit (no training)",
    )
    r6_a0_parser.add_argument("--output", default="runs/r6_a0/formal")
    r6_a0_parser.add_argument(
        "--d0-summary",
        default="runs/r5_d0/formal/summary.json",
    )

    r6_a1_parser = subparsers.add_parser(
        "r6-a1",
        help="R6-A1 nested-LOO pairwise J-margin uncertainty audit (no planner)",
    )
    r6_a1_parser.add_argument("--output", default="runs/r6_a1/formal")
    r6_a1_parser.add_argument(
        "--a0-summary",
        default="runs/r6_a0/formal/summary.json",
    )
    r6_a1_parser.add_argument(
        "--d0-summary",
        default="runs/r5_d0/formal/summary.json",
    )

    r6_b0_parser = subparsers.add_parser(
        "r6-b0",
        help="R6-B0 frozen PX-vs-pi0 abstention (algebraic; no new rollouts)",
    )
    r6_b0_parser.add_argument("--output", default="runs/r6_b0/formal")
    r6_b0_parser.add_argument(
        "--a1-summary",
        default="runs/r6_a1/formal/summary.json",
    )
    r6_b0_parser.add_argument(
        "--d0-summary",
        default="runs/r5_d0/formal/summary.json",
    )

    r7_p0_parser = subparsers.add_parser(
        "r7-p0",
        help="R7-P0 fresh actuator-effectiveness switchability preflight (no certificate)",
    )
    r7_p0_parser.add_argument("--output", default="runs/r7_p0/formal")
    r7_p0_parser.add_argument(
        "--b0-summary",
        default="runs/r6_b0/formal/summary.json",
    )

    r7_a0_parser = subparsers.add_parser(
        "r7-a0",
        help="R7-A0 one-sided split-conformal switch certificate (fresh alpha)",
    )
    r7_a0_parser.add_argument("--output", default="runs/r7_a0/formal")
    r7_a0_parser.add_argument(
        "--p0-summary",
        default="runs/r7_p0/formal/summary.json",
    )

    r7_p1_parser = subparsers.add_parser(
        "r7-p1",
        help="R7-P1 saturated-actuator misspecification preflight (no certificate)",
    )
    r7_p1_parser.add_argument("--output", default="runs/r7_p1/formal")
    r7_p1_parser.add_argument(
        "--a0-summary",
        default="runs/r7_a0/formal/summary.json",
    )

    r7_a1_parser = subparsers.add_parser(
        "r7-a1",
        help="R7-A1 frozen conformal certificate on saturated actuator family",
    )
    r7_a1_parser.add_argument("--output", default="runs/r7_a1/formal")
    r7_a1_parser.add_argument(
        "--p1-summary",
        default="runs/r7_p1/formal/summary.json",
    )

    r7_a2_parser = subparsers.add_parser(
        "r7-a2",
        help="R7-A2 recall recovery with frozen certificate and one spline",
    )
    r7_a2_parser.add_argument("--output", default="runs/r7_a2/formal")
    r7_a2_parser.add_argument(
        "--a1-summary",
        default="runs/r7_a1/formal/summary.json",
    )

    r7_b0_parser = subparsers.add_parser(
        "r7-b0",
        help="R7-B0 world-model-mediated switch certificate",
    )
    r7_b0_parser.add_argument("--output", default="runs/r7_b0/formal")
    r7_b0_parser.add_argument(
        "--a2-summary",
        default="runs/r7_a2/formal/summary.json",
    )

    r7_b1_parser = subparsers.add_parser(
        "r7-b1",
        help="R7-B1 frozen B0 certificate under F_max dynamics shift",
    )
    r7_b1_parser.add_argument("--output", default="runs/r7_b1/formal")
    r7_b1_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r8_p0_parser = subparsers.add_parser(
        "r8-p0",
        help="R8-P0 active certificate-validity feasibility (no detector)",
    )
    r8_p0_parser.add_argument("--output", default="runs/r8_p0/formal")
    r8_p0_parser.add_argument(
        "--b1-summary",
        default="runs/r7_b1/formal/summary.json",
    )
    r8_p0_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r8_p1_parser = subparsers.add_parser(
        "r8-p1",
        help="R8-P1 state-neutral four-phase validity probe",
    )
    r8_p1_parser.add_argument("--output", default="runs/r8_p1/formal")
    r8_p1_parser.add_argument(
        "--p0-summary",
        default="runs/r8_p0/formal/summary.json",
    )
    r8_p1_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r8_p2_parser = subparsers.add_parser(
        "r8-p2",
        help="R8-P2 damping-aware three-phase validity probe",
    )
    r8_p2_parser.add_argument("--output", default="runs/r8_p2/formal")
    r8_p2_parser.add_argument(
        "--p1-summary",
        default="runs/r8_p1/formal/summary.json",
    )
    r8_p2_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r8_r0_parser = subparsers.add_parser(
        "r8-r0",
        help="R8-R0 billed P0 probe then task-independent reset",
    )
    r8_r0_parser.add_argument("--output", default="runs/r8_r0/formal")
    r8_r0_parser.add_argument(
        "--p2-summary",
        default="runs/r8_p2/formal/summary.json",
    )
    r8_r0_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r9_p0_parser = subparsers.add_parser(
        "r9-p0",
        help="R9-P0 amortized validity-acquisition feasibility",
    )
    r9_p0_parser.add_argument("--output", default="runs/r9_p0/formal")
    r9_p0_parser.add_argument(
        "--r0-summary",
        default="runs/r8_r0/formal/summary.json",
    )
    r9_p0_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r9_a0_parser = subparsers.add_parser(
        "r9-a0",
        help="R9-A0 persistent certificate-validity belief",
    )
    r9_a0_parser.add_argument("--output", default="runs/r9_a0/formal")
    r9_a0_parser.add_argument(
        "--p0-summary",
        default="runs/r9_p0/formal/summary.json",
    )
    r9_a0_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r9_b0_parser = subparsers.add_parser(
        "r9-b0",
        help="R9-B0 passive unidirectional validity-license revocation",
    )
    r9_b0_parser.add_argument("--output", default="runs/r9_b0/formal")
    r9_b0_parser.add_argument(
        "--a0-summary",
        default="runs/r9_a0/formal/summary.json",
    )
    r9_b0_parser.add_argument(
        "--p0-summary",
        default="runs/r9_p0/formal/summary.json",
    )
    r9_b0_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r9_b1_p0_parser = subparsers.add_parser(
        "r9-b1-p0",
        help="R9-B1-P0 periodic active revalidation feasibility",
    )
    r9_b1_p0_parser.add_argument("--output", default="runs/r9_b1_p0/formal")
    r9_b1_p0_parser.add_argument(
        "--b0-r9-summary",
        default="runs/r9_b0/formal/summary.json",
    )
    r9_b1_p0_parser.add_argument(
        "--a0-summary",
        default="runs/r9_a0/formal/summary.json",
    )
    r9_b1_p0_parser.add_argument(
        "--p0-summary",
        default="runs/r9_p0/formal/summary.json",
    )
    r9_b1_p0_parser.add_argument(
        "--b0-summary",
        default="runs/r7_b0/formal/summary.json",
    )

    r10_c0_parser = subparsers.add_parser(
        "r10-c0",
        help="R10-C0 real sensor-chain C0 (locked without hardware log)",
    )
    r10_c0_parser.add_argument("--output", default="runs/r10_c0/formal")
    r10_c0_parser.add_argument(
        "--residual-log",
        default="runs/r10_c0/real/residual.h5",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--batches", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        config = load_config(args.config)
        config = apply_overrides(config, args)
        print(json.dumps(doctor(config), ensure_ascii=False, indent=2))
        return
    if args.command in ("train", "suite"):
        config = apply_overrides(load_config(args.config), args)
        if args.command == "train":
            train_one(args.model, config)
        else:
            run_suite(config)
        return
    if args.command == "summarize":
        results = summarize_existing(args.output)
        print(json.dumps({"models": [item["model"] for item in results]}, indent=2))
        return
    if args.command == "v05-pareto":
        rows = pareto_experiment(
            args.checkpoint,
            args.output,
            batches=args.batches,
            batch_size=args.batch_size,
            horizon=args.horizon,
            device_name=args.device,
        )
        from .plots import plot_pareto

        plot_pareto(args.output)
        print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))
        return
    if args.command == "v05-horizon":
        rows = horizon_experiment(
            args.checkpoints, args.output, device_name=args.device
        )
        from .plots import plot_horizon

        plot_horizon(args.output)
        print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))
        return
    if args.command == "v05-continuum":
        summary = continuum_experiment(
            args.checkpoint,
            args.output,
            grid_size=args.grid_size,
            device_name=args.device,
        )
        from .plots import plot_continuum

        plot_continuum(args.output)
        print(json.dumps(summary, indent=2))
        return
    if args.command == "v05-latency":
        rows = latency_experiment(
            args.checkpoints,
            args.output,
            iterations=args.iterations,
            device_name=args.device,
        )
        from .plots import plot_latency

        plot_latency(args.output)
        print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))
        return
    if args.command == "v05-seeds":
        result = seed_experiment(args.config, args.output, args.seeds)
        print(json.dumps(result["paired"], indent=2))
        return
    if args.command == "v05-aggregate":
        result = aggregate_seeds(args.output)
        print(json.dumps(result["paired"], indent=2))
        return
    if args.command == "v05-ablation":
        rows = ablation_experiment(
            args.config, args.output, seed=args.seed
        )
        print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))
        return
    if args.command == "v06":
        config = V06Config(
            train_steps=args.train_steps,
            latency_iterations=args.latency_iterations,
        )
        result = run_v06(
            args.output,
            device_name=args.device,
            full_matrix=args.full_matrix,
            reuse_models=args.reuse_models,
            config=config,
        )
        from .plots import plot_v06

        plot_v06(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v07":
        config = V07Config(
            seeds=tuple(args.seeds),
            expert_train_steps=args.expert_train_steps,
            router_train_steps=args.router_train_steps,
            latency_iterations=args.latency_iterations,
        )
        result = run_v07(
            args.output,
            device_name=args.device,
            config=config,
            reuse_models=args.reuse_models,
        )
        from .plots import plot_v07

        plot_v07(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v1":
        config = V1Config(
            seeds=tuple(args.seeds),
            expert_train_steps=args.expert_train_steps,
            router_train_steps=args.router_train_steps,
        )
        result = run_v1(args.output, device_name=args.device, config=config)
        from .plots import plot_v1

        plot_v1(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v2":
        config = V2Config(
            seeds=tuple(args.seeds), episodes=args.episodes, probes=args.probes
        )
        result = run_v2(args.output, device_name=args.device, config=config)
        from .plots import plot_v2

        plot_v2(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v3":
        config = V3Config(seeds=tuple(args.seeds), episodes=args.episodes)
        result = run_v3(args.output, device_name=args.device, config=config)
        from .plots import plot_v3

        plot_v3(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v4":
        config = V4Config(seeds=tuple(args.seeds), episodes=args.episodes)
        result = run_v4(args.output, device_name=args.device, config=config)
        from .plots import plot_v4

        plot_v4(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v5":
        config = V5Config(
            seeds=tuple(args.seeds),
            episodes=args.episodes,
            noise_levels=tuple(args.noise_levels),
        )
        result = run_v5(args.output, device_name=args.device, config=config)
        from .plots import plot_v5

        plot_v5(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6":
        config = V6Config(
            seeds=tuple(args.seeds),
            episodes=args.episodes,
            validation_max=args.validation_max,
            noise_levels=tuple(args.noise_levels),
        )
        result = run_v6(args.output, device_name=args.device, config=config)
        from .plots import plot_v6

        plot_v6(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r0-doctor":
        result = inspect_robotwin(args.robotwin_repo, args.robotwin_python)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "v6r0-smoke":
        config = V6R0Config(
            seeds=tuple(args.seeds),
            episodes_per_regime=args.episodes_per_regime,
        )
        result = run_v6r0_smoke(
            args.output,
            device_name=args.device,
            config=config,
            robotwin_repo=args.robotwin_repo,
        )
        from .plots import plot_v6r0

        plot_v6r0(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r01-closure":
        config = V6R01Config(
            seeds=tuple(args.seeds),
            episodes=args.episodes,
            samples_per_episode=args.samples_per_episode,
        )
        result = run_v6r01_closure(
            args.output, device_name=args.device, config=config
        )
        from .plots import plot_v6r01

        plot_v6r01(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r02":
        config = V6R02Config(
            seeds=tuple(args.seeds),
            episodes_per_regime=args.episodes_per_regime,
        )
        result = run_v6r02(args.output, device_name=args.device, config=config)
        from .plots import plot_v6r02

        plot_v6r02(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r03":
        config = V6R03Config(
            seeds=tuple(args.seeds),
            episodes_per_regime=args.episodes_per_regime,
        )
        result = run_v6r03(args.output, device_name=args.device, config=config)
        from .plots import plot_v6r03

        plot_v6r03(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r04a":
        config = V6R04AConfig(seeds=tuple(args.seeds), episodes_per_regime=args.episodes)
        result = run_v6r04a(args.output, device_name=args.device, config=config)
        from .plots import plot_v6r04a

        plot_v6r04a(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r04b":
        config = V6R04BConfig(seeds=tuple(args.seeds), episodes_per_regime=args.episodes)
        result = run_v6r04b(args.output, device_name=args.device, config=config)
        from .plots import plot_v6r04b

        plot_v6r04b(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r05":
        config = V6R05Config(
            selection_seeds=tuple(args.selection_seeds),
            confirmation_seeds=tuple(args.confirmation_seeds),
            episodes_per_seed=args.episodes,
        )
        result = run_v6r05(args.output, device_name=args.device, config=config)
        from .plots import plot_v6r05

        plot_v6r05(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r05p":
        config = V6R05PConfig(
            seeds=tuple(args.seeds), episodes_per_regime=args.episodes
        )
        result = run_v6r05p(args.output, device_name=args.device, config=config)
        from .plots import plot_v6r05p

        plot_v6r05p(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "v6r06":
        config = V6R06Config(
            seeds=tuple(args.seeds), episodes_per_regime=args.episodes
        )
        result = run_v6r06(args.output, device_name=args.device, config=config)
        from .plots import plot_v6r06

        plot_v6r06(args.output)
        print(json.dumps(result, indent=2))
        return
    if args.command == "r1-mj0":
        from .r1_mj import R1MJ0Config, run_r1_mj0

        result = run_r1_mj0(
            args.output,
            config=R1MJ0Config(duration_s=args.duration),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "r1-rs1":
        from .r1_rs1 import R1RS1Config, run_r1_rs1

        result = run_r1_rs1(
            args.output,
            rs0_summary=args.rs0_summary,
            config=R1RS1Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1a":
        from .r1_rs1a import R1RS1AConfig, run_r1_rs1a

        result = run_r1_rs1a(
            args.output,
            rs0_summary=args.rs0_summary,
            config=R1RS1AConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1a1":
        from .r1_rs1a1 import R1RS1A1Config, run_r1_rs1a1

        result = run_r1_rs1a1(
            args.output,
            rs0_summary=args.rs0_summary,
            config=R1RS1A1Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1a2":
        from .r1_rs1a2 import R1RS1A2Config, run_r1_rs1a2

        result = run_r1_rs1a2(
            args.output,
            rs0_summary=args.rs0_summary,
            config=R1RS1A2Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1a3":
        from .r1_rs1a3 import R1RS1A3Config, run_r1_rs1a3

        result = run_r1_rs1a3(
            args.output,
            rs0_summary=args.rs0_summary,
            config=R1RS1A3Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1b":
        from .r1_rs1b import R1RS1BConfig, run_r1_rs1b

        result = run_r1_rs1b(
            args.output,
            rs0_summary=args.rs0_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS1BConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1b1":
        from .r1_rs1b1 import R1RS1B1Config, run_r1_rs1b1

        result = run_r1_rs1b1(
            args.output,
            rs0_summary=args.rs0_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS1B1Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1b2":
        from .r1_rs1b2 import R1RS1B2Config, run_r1_rs1b2

        result = run_r1_rs1b2(
            args.output,
            rs0_summary=args.rs0_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS1B2Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1c":
        from .r1_rs1c import R1RS1CConfig, run_r1_rs1c

        result = run_r1_rs1c(
            args.output,
            rs0_summary=args.rs0_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS1CConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs2-c0":
        from .r1_rs2 import R1RS2C0Config, run_r1_rs2_c0

        result = run_r1_rs2_c0(
            args.output,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS2C0Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs2":
        from .r1_rs2_formal import R1RS2FormalConfig, run_r1_rs2_formal

        result = run_r1_rs2_formal(
            args.output,
            rs2_c0_summary=args.rs2_c0_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS2FormalConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs2a":
        from .r1_rs2a import R1RS2AConfig, run_r1_rs2a

        result = run_r1_rs2a(
            args.output,
            rs2_formal_summary=args.rs2_formal_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS2AConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs2b":
        from .r1_rs2b import R1RS2BConfig, run_r1_rs2b

        result = run_r1_rs2b(
            args.output,
            rs2a_summary=args.rs2a_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS2BConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs3a":
        from .r1_rs3a import R1RS3AConfig, run_r1_rs3a

        result = run_r1_rs3a(
            args.output,
            rs2b_summary=args.rs2b_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS3AConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs3a1":
        from .r1_rs3a1 import R1RS3A1Config, run_r1_rs3a1

        result = run_r1_rs3a1(
            args.output,
            rs3a_summary=args.rs3a_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS3A1Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs4a":
        from .r1_rs4a import R1RS4AConfig, run_r1_rs4a

        result = run_r1_rs4a(
            args.output,
            rs3a1_summary=args.rs3a1_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS4AConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs5a":
        from .r1_rs5a import R1RS5AConfig, run_r1_rs5a

        result = run_r1_rs5a(
            args.output,
            rs4a_summary=args.rs4a_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R1RS5AConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7a":
        from .r3_v7a import R3V7AConfig, run_r3_v7a

        result = run_r3_v7a(
            args.output,
            rs5a_summary=args.rs5a_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7AConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7b":
        from .r3_v7b import R3V7BConfig, run_r3_v7b

        result = run_r3_v7b(
            args.output,
            v7a_summary=args.v7a_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7BConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7b1":
        from .r3_v7b1 import R3V7B1Config, run_r3_v7b1

        result = run_r3_v7b1(
            args.output,
            v7b_summary=args.v7b_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7B1Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7b2":
        from .r3_v7b2 import R3V7B2Config, run_r3_v7b2

        result = run_r3_v7b2(
            args.output,
            v7b1_summary=args.v7b1_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7B2Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7b3":
        from .r3_v7b3 import R3V7B3Config, run_r3_v7b3

        result = run_r3_v7b3(
            args.output,
            v7b2_summary=args.v7b2_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7B3Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7c":
        from .r3_v7c import R3V7CConfig, run_r3_v7c

        result = run_r3_v7c(
            args.output,
            v7b3_summary=args.v7b3_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7CConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7c1":
        from .r3_v7c1 import R3V7C1Config, run_r3_v7c1

        result = run_r3_v7c1(
            args.output,
            v7c_summary=args.v7c_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7C1Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7c2":
        from .r3_v7c2 import R3V7C2Config, run_r3_v7c2

        result = run_r3_v7c2(
            args.output,
            v7c1_summary=args.v7c1_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7C2Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7c3":
        from .r3_v7c3 import R3V7C3Config, run_r3_v7c3

        result = run_r3_v7c3(
            args.output,
            v7c2_summary=args.v7c2_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7C3Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7c4":
        from .r3_v7c4 import R3V7C4Config, run_r3_v7c4

        result = run_r3_v7c4(
            args.output,
            v7c3_summary=args.v7c3_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7C4Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7c5":
        from .r3_v7c5 import R3V7C5Config, run_r3_v7c5

        result = run_r3_v7c5(
            args.output,
            v7c4_summary=args.v7c4_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7C5Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r3-v7d":
        from .r3_v7d import R3V7DConfig, run_r3_v7d

        result = run_r3_v7d(
            args.output,
            v7c5_summary=args.v7c5_summary,
            rs1a5_summary=args.rs1a5_summary,
            config=R3V7DConfig(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r4-i0":
        from .r4_i0 import R4I0Config, run_r4_i0

        result = run_r4_i0(args.output, config=R4I0Config())
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r4-i1":
        from .r4_i1 import run_r4_i1

        result = run_r4_i1(args.output, i0_summary=args.i0_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r4-c0":
        from .r4_c0 import R4C0Config, run_r4_c0

        result = run_r4_c0(
            args.output,
            i0_summary=args.i0_summary,
            i1_summary=args.i1_summary,
            config=R4C0Config(),
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r4-c1":
        from .r4_c1 import run_r4_c1

        result = run_r4_c1(
            args.output,
            i0_summary=args.i0_summary,
            i1_summary=args.i1_summary,
            c0_summary=args.c0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r4-c2":
        from .r4_c2 import run_r4_c2

        result = run_r4_c2(
            args.output,
            i0_summary=args.i0_summary,
            i1_summary=args.i1_summary,
            c1_summary=args.c1_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r5-i0":
        from .r5_i0 import run_r5_i0

        result = run_r5_i0(args.output, mechanism=args.mechanism)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r5-i0-selfstress":
        from .r5_i0_selfstress import run_r5_i0_selfstress

        result = run_r5_i0_selfstress(args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r5-i1-selfstress":
        from .r5_i1_selfstress import run_r5_i1_selfstress

        result = run_r5_i1_selfstress(args.output, i0_summary=args.i0_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r5-d0-preflight":
        from .r5_d0_preflight import run_r5_d0_preflight

        result = run_r5_d0_preflight(args.output, i1_summary=args.i1_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r5-d0":
        from .r5_d0 import run_r5_d0

        result = run_r5_d0(args.output, preflight_summary=args.preflight_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r6-a0":
        from .r6_a0 import run_r6_a0

        result = run_r6_a0(args.output, d0_summary=args.d0_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r6-a1":
        from .r6_a1 import run_r6_a1

        result = run_r6_a1(
            args.output,
            a0_summary=args.a0_summary,
            d0_summary=args.d0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r6-b0":
        from .r6_b0 import run_r6_b0

        result = run_r6_b0(
            args.output,
            a1_summary=args.a1_summary,
            d0_summary=args.d0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r7-p0":
        from .r7_p0 import run_r7_p0

        result = run_r7_p0(args.output, b0_summary=args.b0_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r7-a0":
        from .r7_a0 import run_r7_a0

        result = run_r7_a0(args.output, p0_summary=args.p0_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r7-p1":
        from .r7_p1 import run_r7_p1

        result = run_r7_p1(args.output, a0_summary=args.a0_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r7-a1":
        from .r7_a1 import run_r7_a1

        result = run_r7_a1(args.output, p1_summary=args.p1_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r7-a2":
        from .r7_a2 import run_r7_a2

        result = run_r7_a2(args.output, a1_summary=args.a1_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r7-b0":
        from .r7_b0 import run_r7_b0

        result = run_r7_b0(args.output, a2_summary=args.a2_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r7-b1":
        from .r7_b1 import run_r7_b1

        result = run_r7_b1(args.output, b0_summary=args.b0_summary)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r8-p0":
        from .r8_p0 import run_r8_p0

        result = run_r8_p0(
            args.output,
            b1_summary=args.b1_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r8-p1":
        from .r8_p1 import run_r8_p1

        result = run_r8_p1(
            args.output,
            p0_summary=args.p0_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r8-p2":
        from .r8_p2 import run_r8_p2

        result = run_r8_p2(
            args.output,
            p1_summary=args.p1_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r8-r0":
        from .r8_r0 import run_r8_r0

        result = run_r8_r0(
            args.output,
            p2_summary=args.p2_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r9-p0":
        from .r9_p0 import run_r9_p0

        result = run_r9_p0(
            args.output,
            r0_summary=args.r0_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r9-a0":
        from .r9_a0 import run_r9_a0

        result = run_r9_a0(
            args.output,
            p0_summary=args.p0_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r9-b0":
        from .r9_b0 import run_r9_b0

        result = run_r9_b0(
            args.output,
            a0_summary=args.a0_summary,
            p0_summary=args.p0_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r9-b1-p0":
        from .r9_b1_p0 import run_r9_b1_p0

        result = run_r9_b1_p0(
            args.output,
            b0_r9_summary=args.b0_r9_summary,
            a0_summary=args.a0_summary,
            p0_summary=args.p0_summary,
            b0_summary=args.b0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r10-c0":
        from .r10_c0 import run_r10_c0

        result = run_r10_c0(args.output, residual_log=args.residual_log)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1a4":
        from .r1_rs1a4 import run_r1_rs1a4

        result = run_r1_rs1a4(
            args.output,
            rs0_summary=args.rs0_summary,
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs1a5":
        from .r1_rs1a5 import run_r1_rs1a5

        result = run_r1_rs1a5(
            args.output,
            rs0_summary=args.rs0_summary,
            smoke=bool(args.smoke),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "r1-rs0":
        from .r1_rs import R1RS0Config, run_r1_rs0

        result = run_r1_rs0(
            args.output,
            mj0_summary=args.mj0_summary,
            config=R1RS0Config(duration_s=args.duration),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        device = resolve_device(args.device)
        model, config = load_checkpoint(args.checkpoint, device)
        config.device = str(device)
        if args.batches is not None:
            config.eval.batches = args.batches
        metrics = evaluate_model(model, SyntheticWorld(config.world), config, device)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    main()
