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

    sim_x0_parser = subparsers.add_parser(
        "sim-x0",
        help="SIM-X0 nominal MuJoCo force accounting (not R10-C0)",
    )
    sim_x0_parser.add_argument("--output", default="runs/sim_x0/formal")
    sim_x0_parser.add_argument("--duration", type=float, default=10.0)

    sim_x1_parser = subparsers.add_parser(
        "sim-x1",
        help="SIM-X1 oracle damping mismatch on simx_hinge.v1 (not R10-C0)",
    )
    sim_x1_parser.add_argument("--output", default="runs/sim_x1/formal")
    sim_x1_parser.add_argument("--duration", type=float, default=10.0)
    sim_x1_parser.add_argument(
        "--x0-summary",
        default="runs/sim_x0/formal/summary.json",
    )

    sim_x2_parser = subparsers.add_parser(
        "sim-x2",
        help="SIM-X2 action-conditioned validity information channel (not R10)",
    )
    sim_x2_parser.add_argument("--output", default="runs/sim_x2/formal")
    sim_x2_parser.add_argument("--duration", type=float, default=10.0)
    sim_x2_parser.add_argument(
        "--x1-summary",
        default="runs/sim_x1/formal/summary.json",
    )

    sim_x3_parser = subparsers.add_parser(
        "sim-x3",
        help="SIM-X3 persistent validity lifecycle (not R10)",
    )
    sim_x3_parser.add_argument("--output", default="runs/sim_x3/formal")
    sim_x3_parser.add_argument(
        "--x2-summary",
        default="runs/sim_x2/formal/summary.json",
    )
    sim_x3_parser.add_argument("--noise-seeds", type=int, default=100)

    vis_x0_parser = subparsers.add_parser(
        "vis-x0",
        help="VIS-X0 GT seg+depth state interface smoke (not R10)",
    )
    vis_x0_parser.add_argument("--output", default="runs/vis_x0/formal")
    vis_x0_parser.add_argument("--duration", type=float, default=4.0)

    vis_x1_parser = subparsers.add_parser(
        "vis-x1",
        help="VIS-X1 RGB-D perception pseudo-residual (not R10)",
    )
    vis_x1_parser.add_argument("--output", default="runs/vis_x1/formal")
    vis_x1_parser.add_argument("--duration", type=float, default=4.0)
    vis_x1_parser.add_argument(
        "--x0-summary",
        default="runs/vis_x0/formal/summary.json",
    )

    vis_x2_parser = subparsers.add_parser(
        "vis-x2",
        help="VIS-X2 perception→false physics diagnosis (not R10)",
    )
    vis_x2_parser.add_argument("--output", default="runs/vis_x2/formal")
    vis_x2_parser.add_argument("--duration", type=float, default=2.0)
    vis_x2_parser.add_argument(
        "--x1-summary",
        default="runs/vis_x1/formal/summary.json",
    )

    vis_x3_parser = subparsers.add_parser(
        "vis-x3",
        help="VIS-X3 perception uncertainty→physics-attribution veto (not R10)",
    )
    vis_x3_parser.add_argument("--output", default="runs/vis_x3/formal")
    vis_x3_parser.add_argument("--duration", type=float, default=2.0)
    vis_x3_parser.add_argument(
        "--x2-summary",
        default="runs/vis_x2/formal/summary.json",
    )

    vis_ext0_parser = subparsers.add_parser(
        "vis-ext0",
        help="VIS-EXT0 robosuite Door visual external-validity bridge (not R10)",
    )
    vis_ext0_parser.add_argument("--output", default="runs/vis_ext0/formal")
    vis_ext0_parser.add_argument("--duration", type=float, default=1.0)
    vis_ext0_parser.add_argument(
        "--x3-summary",
        default="runs/vis_x3/formal/summary.json",
    )

    cap_x0_parser = subparsers.add_parser(
        "cap-x0",
        help="CAP-X0 arm3 benchmark/accounting (not capacity claim, not R10)",
    )
    cap_x0_parser.add_argument("--output", default="runs/cap_x0/formal")
    cap_x0_parser.add_argument("--duration", type=float, default=2.0)
    cap_x0_parser.add_argument("--train-scenes", type=int, default=128)
    cap_x0_parser.add_argument("--val-scenes", type=int, default=32)
    cap_x0_parser.add_argument("--test-scenes", type=int, default=64)
    cap_x0_parser.add_argument("--traj-per-scene", type=int, default=8)

    cap_x1_parser = subparsers.add_parser(
        "cap-x1",
        help="CAP-X1 matched-family capacity R_P(0) (rho=0, mu=0; not R10)",
    )
    cap_x1_parser.add_argument("--output", default="runs/cap_x1/formal")
    cap_x1_parser.add_argument("--x0-data", default="runs/cap_x0/formal")
    cap_x1_parser.add_argument("--plan-tasks", type=int, default=8)
    cap_x1_parser.add_argument("--rollout-episodes", type=int, default=32)

    cap_x2_p0_parser = subparsers.add_parser(
        "cap-x2-p0",
        help="CAP-X2-P0 oracle planning harness lock (before rho capacity sweep)",
    )
    cap_x2_p0_parser.add_argument("--output", default="runs/cap_x2/p0")
    cap_x2_p0_parser.add_argument("--n-tasks", type=int, default=24)

    cap_x2_parser = subparsers.add_parser(
        "cap-x2",
        help="CAP-X2 R_P(rho) capacity decay (requires P0 freeze; not R10)",
    )
    cap_x2_parser.add_argument("--output", default="runs/cap_x2/formal")
    cap_x2_parser.add_argument("--p0-dir", default="runs/cap_x2/p0")

    cap_x3_p0_parser = subparsers.add_parser(
        "cap-x3-p0",
        help="CAP-X3-P0 identifiability preflight (no K90; not R10)",
    )
    cap_x3_p0_parser.add_argument("--output", default="runs/cap_x3/p0")

    cap_x3_parser = subparsers.add_parser(
        "cap-x3",
        help="CAP-X3 few-shot θ vs same-dim latent (requires P0; not R10)",
    )
    cap_x3_parser.add_argument("--output", default="runs/cap_x3/formal")
    cap_x3_parser.add_argument("--p0-dir", default="runs/cap_x3/p0")

    rtwx_x0_p0_parser = subparsers.add_parser(
        "rtwx-x0-p0",
        help="RoboTwin-X0-P0 oracle-state drawer instrument (no RGB; not official task)",
    )
    rtwx_x0_p0_parser.add_argument("--output", default="runs/rtwx_x0/p0")

    rtwx_x0_smoke_parser = subparsers.add_parser(
        "rtwx-x0-smoke",
        help="RoboTwin-X0 official put_object_cabinet smoke (oracle state; no RGB in s)",
    )
    rtwx_x0_smoke_parser.add_argument("--output", default="runs/rtwx_x0/smoke")
    rtwx_x0_smoke_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0_smoke_parser.add_argument("--seed", type=int, default=0)

    rtwx_x0_parser = subparsers.add_parser(
        "rtwx-x0",
        help="RoboTwin-X0 formal oracle-state capacity (requires smoke PASS; not R10)",
    )
    rtwx_x0_parser.add_argument("--output", default="runs/rtwx_x0/formal")
    rtwx_x0_parser.add_argument("--smoke-summary", default="runs/rtwx_x0/smoke/summary.json")
    rtwx_x0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")

    rtwx_x0r_parser = subparsers.add_parser(
        "rtwx-x0r",
        help="RoboTwin-X0R cabinet excitation & dynamics interface (not X0 re-score; not R10)",
    )
    rtwx_x0r_parser.add_argument("--output", default="runs/rtwx_x0r")
    rtwx_x0r_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")

    rtwx_x0s_parser = subparsers.add_parser(
        "rtwx-x0s",
        help="RTWX-X0S cabinet dynamics structure audit (no neural residual; not capacity; not R10)",
    )
    rtwx_x0s_parser.add_argument("--output", default="runs/rtwx_x0s")
    rtwx_x0s_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0s_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0s_parser.add_argument("--apply-mode", default="command_only", choices=("command_only", "cmd_plus_passive"))

    rtwx_x0f_parser = subparsers.add_parser(
        "rtwx-x0f",
        help="RTWX-X0F force-channel / clock audit (no train; no phi; not capacity; not R10)",
    )
    rtwx_x0f_parser.add_argument("--output", default="runs/rtwx_x0f")
    rtwx_x0f_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0f_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))

    rtwx_x0c_parser = subparsers.add_parser(
        "rtwx-x0c",
        help="RTWX-X0C native-qpos closed-loop servo input (no nets; not capacity; not R10)",
    )
    rtwx_x0c_parser.add_argument("--output", default="runs/rtwx_x0c")
    rtwx_x0c_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0c_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0c_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short env smoke (not frozen formal n); write to a non-formal output dir",
    )

    rtwx_x0d_parser = subparsers.add_parser(
        "rtwx-x0d",
        help="RTWX-X0D native closed-loop structure (no force, no nets, not capacity)",
    )
    rtwx_x0d_parser.add_argument("--output", default="runs/rtwx_x0d")
    rtwx_x0d_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0d_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0d_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short env smoke (not frozen formal n); write to a non-formal output dir",
    )

    rtwx_x0e_parser = subparsers.add_parser(
        "rtwx-x0e",
        help="RTWX-X0E direct native-window increment (not X0D patch; no nets; not capacity)",
    )
    rtwx_x0e_parser.add_argument("--output", default="runs/rtwx_x0e")
    rtwx_x0e_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0e_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0e_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short env smoke (not frozen formal n); write to a non-formal output dir",
    )

    rtwx_x0e1_parser = subparsers.add_parser(
        "rtwx-x0e1",
        help="RTWX-X0E1 structured vs latent capacity on native-window increment (not R10)",
    )
    rtwx_x0e1_parser.add_argument("--output", default="runs/rtwx_x0e1")
    rtwx_x0e1_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0e1_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0e1_parser.add_argument("--x0e-summary", default="runs/rtwx_x0e/run.json")
    rtwx_x0e1_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Numpy/small-grid smoke (not frozen formal widths)",
    )

    rtwx_x0es_parser = subparsers.add_parser(
        "rtwx-x0es",
        help="RTWX-X0ES frozen M2 macro-transition rollout stability audit (no retrain; not capacity)",
    )
    rtwx_x0es_parser.add_argument("--output", default="runs/rtwx_x0es")
    rtwx_x0es_parser.add_argument("--cache", default="runs/rtwx_x0e1/cache_splits.npz")
    rtwx_x0es_parser.add_argument("--x0e1-summary", default="runs/rtwx_x0e1/run.json")

    rtwx_x0es1_parser = subparsers.add_parser(
        "rtwx-x0es1",
        help="RTWX-X0ES1 rare local instability confirmation on fresh data (frozen M2; not capacity)",
    )
    rtwx_x0es1_parser.add_argument("--output", default="runs/rtwx_x0es1")
    rtwx_x0es1_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0es1_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0es1_parser.add_argument("--old-cache", default="runs/rtwx_x0e1/cache_splits.npz")
    rtwx_x0es1_parser.add_argument("--smoke", action="store_true")

    rtwx_x0es2_parser = subparsers.add_parser(
        "rtwx-x0es2",
        help="RTWX-X0ES2 stability-constrained X0E-M2 (same basis; not capacity)",
    )
    rtwx_x0es2_parser.add_argument("--output", default="runs/rtwx_x0es2")
    rtwx_x0es2_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0es2_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0es2_parser.add_argument("--old-cache", default="runs/rtwx_x0e1/cache_splits.npz")
    rtwx_x0es2_parser.add_argument("--smoke", action="store_true")
    rtwx_x0es2_parser.add_argument("--select-only", action="store_true")

    rtwx_x0eh_parser = subparsers.add_parser(
        "rtwx-x0eh",
        help="RTWX-X0EH receding-horizon sufficiency audit (frozen M2; diagnostic only)",
    )
    rtwx_x0eh_parser.add_argument("--output", default="runs/rtwx_x0eh")
    rtwx_x0eh_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0eh_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0eh_parser.add_argument("--old-cache", default="runs/rtwx_x0e1/cache_splits.npz")
    rtwx_x0eh_parser.add_argument("--smoke", action="store_true")

    rtwx_x0eh1_parser = subparsers.add_parser(
        "rtwx-x0eh1",
        help="RTWX-X0EH1 K=4 receding structured vs neural capacity (fully fresh splits)",
    )
    rtwx_x0eh1_parser.add_argument("--output", default="runs/rtwx_x0eh1")
    rtwx_x0eh1_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0eh1_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0eh1_parser.add_argument("--smoke", action="store_true")

    rtwx_x0eh2_parser = subparsers.add_parser(
        "rtwx-x0eh2",
        help="RTWX-X0EH2 cross-task K=4 receding capacity external validity",
    )
    rtwx_x0eh2_parser.add_argument("--output", default="runs/rtwx_x0eh2")
    rtwx_x0eh2_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0eh2_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0eh2_parser.add_argument("--smoke", action="store_true")
    rtwx_x0eh2_parser.add_argument("--include-exploratory", action="store_true")
    rtwx_x0eh2_parser.add_argument(
        "--tasks-only",
        nargs="+",
        choices=["place_empty_cup", "stamp_seal", "adjust_bottle"],
        default=None,
    )

    rtwx_x0rgb_parser = subparsers.add_parser(
        "rtwx-x0rgb",
        help="RTWX-X0RGB sim RGB → state → frozen M2 (not real camera; not R10)",
    )
    rtwx_x0rgb_parser.add_argument("--output", default="runs/rtwx_x0rgb")
    rtwx_x0rgb_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0rgb_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0rgb_parser.add_argument("--smoke", action="store_true")

    rtwx_x0rgb1_parser = subparsers.add_parser(
        "rtwx-x0rgb1",
        help="RTWX-X0RGB1 spatial-temporal RGB→(q,qd); frozen M2; not R10",
    )
    rtwx_x0rgb1_parser.add_argument("--output", default="runs/rtwx_x0rgb1")
    rtwx_x0rgb1_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_x0rgb1_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_x0rgb1_parser.add_argument("--smoke", action="store_true")

    rtwx_o0_parser = subparsers.add_parser(
        "rtwx-o0",
        help="RTWX-O0 RGB→cup object state (not robot q; not M2; not R10)",
    )
    rtwx_o0_parser.add_argument("--output", default="runs/rtwx_o0")
    rtwx_o0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0d_parser = subparsers.add_parser(
        "rtwx-o0d",
        help="RTWX-O0D perception instrument audit (no O1; no observability claim)",
    )
    rtwx_o0d_parser.add_argument("--output", default="runs/rtwx_o0d")
    rtwx_o0d_parser.add_argument("--o0-cache", default="runs/rtwx_o0")
    rtwx_o0d_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0d_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0d_parser.add_argument("--smoke", action="store_true")

    rtwx_o0d1_parser = subparsers.add_parser(
        "rtwx-o0d1",
        help="RTWX-O0D1 perception instrument repair (no O0R/O1; no claim)",
    )
    rtwx_o0d1_parser.add_argument("--output", default="runs/rtwx_o0d1")
    rtwx_o0d1_parser.add_argument("--o0-cache", default="runs/rtwx_o0")
    rtwx_o0d1_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0d1_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0d1_parser.add_argument("--smoke", action="store_true")

    rtwx_o0d2_parser = subparsers.add_parser(
        "rtwx-o0d2",
        help="RTWX-O0D2 perception instrument closure (lookup/flatten/FOV; no O0R)",
    )
    rtwx_o0d2_parser.add_argument("--output", default="runs/rtwx_o0d2")
    rtwx_o0d2_parser.add_argument("--o0-cache", default="runs/rtwx_o0")
    rtwx_o0d2_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0d2_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0d2_parser.add_argument("--smoke", action="store_true")

    rtwx_o0d3_parser = subparsers.add_parser(
        "rtwx-o0d3",
        help="RTWX-O0D3 head_camera + CoordConv qualification (no O0R unless qualified)",
    )
    rtwx_o0d3_parser.add_argument("--output", default="runs/rtwx_o0d3")
    rtwx_o0d3_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0d3_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0d3_parser.add_argument("--smoke", action="store_true")

    rtwx_o0r_parser = subparsers.add_parser(
        "rtwx-o0r",
        help="RTWX-O0R fresh head_camera+CoordConv RGB→s^O (frozen O0D3; not R10)",
    )
    rtwx_o0r_parser.add_argument("--output", default="runs/rtwx_o0r")
    rtwx_o0r_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0r_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0r_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g_parser = subparsers.add_parser(
        "rtwx-o0g",
        help="RTWX-O0G geometry-mediated position (G0/G1/G2; pause RGB→p_B; not O1)",
    )
    rtwx_o0g_parser.add_argument("--output", default="runs/rtwx_o0g")
    rtwx_o0g_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g_parser.add_argument(
        "--stage",
        default="g0",
        choices=("g0", "g1", "g01", "g2", "all"),
        help="Diagnostic layer: g0 oracle geom; g01=G0+G1; g2 deferred until G0∧G1",
    )
    rtwx_o0g_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g1b_parser = subparsers.add_parser(
        "rtwx-o0g1b",
        help="RTWX-O0G1b reference-frame alignment (known δ_O; not O0G retune; not O1)",
    )
    rtwx_o0g1b_parser.add_argument("--output", default="runs/rtwx_o0g1b")
    rtwx_o0g1b_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g1b_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g1b_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g2_parser = subparsers.add_parser(
        "rtwx-o0g2",
        help="RTWX-O0G2 learned mask + geometry-mediated position (no RGB→p; not O1)",
    )
    rtwx_o0g2_parser.add_argument("--output", default="runs/rtwx_o0g2")
    rtwx_o0g2_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g2_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g2_parser.add_argument("--smoke", action="store_true")

    rtwx_o0v_parser = subparsers.add_parser(
        "rtwx-o0v",
        help="RTWX-O0V robust visual coverage (multi-seed; head∨observer; no train; not O1)",
    )
    rtwx_o0v_parser.add_argument("--output", default="runs/rtwx_o0v")
    rtwx_o0v_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0v_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0v_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g2r_parser = subparsers.add_parser(
        "rtwx-o0g2r",
        help="RTWX-O0G2R dual-view geometry-mediated position confirmation (not O1)",
    )
    rtwx_o0g2r_parser.add_argument("--output", default="runs/rtwx_o0g2r")
    rtwx_o0g2r_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g2r_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g2r_parser.add_argument("--smoke", action="store_true")

    rtwx_o0c_parser = subparsers.add_parser(
        "rtwx-o0c",
        help="RTWX-O0C composite non-oracle object pose T_BO (unlocks O1 on PASS)",
    )
    rtwx_o0c_parser.add_argument("--output", default="runs/rtwx_o0c")
    rtwx_o0c_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0c_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0c_parser.add_argument("--smoke", action="store_true")

    rtwx_o0c1_parser = subparsers.add_parser(
        "rtwx-o0c1",
        help="RTWX-O0C1 orientation tail mechanism audit (diagnostic; not O1)",
    )
    rtwx_o0c1_parser.add_argument("--output", default="runs/rtwx_o0c1")
    rtwx_o0c1_parser.add_argument("--o0c-cache", default="runs/rtwx_o0c")
    rtwx_o0c1_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g3_parser = subparsers.add_parser(
        "rtwx-o0g3",
        help="RTWX-O0G3 G0 oracle correspondence→Kabsch orientation geometry (not O1)",
    )
    rtwx_o0g3_parser.add_argument("--output", default="runs/rtwx_o0g3")
    rtwx_o0g3_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g3_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g3_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g3r_parser = subparsers.add_parser(
        "rtwx-o0g3r",
        help="RTWX-O0G3R learned x_O + Kabsch orientation (not O1)",
    )
    rtwx_o0g3r_parser.add_argument("--output", default="runs/rtwx_o0g3r")
    rtwx_o0g3r_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g3r_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g3r_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g3b_parser = subparsers.add_parser(
        "rtwx-o0g3b",
        help="RTWX-O0G3B correspondence instrument closure (no O1)",
    )
    rtwx_o0g3b_parser.add_argument("--output", default="runs/rtwx_o0g3b")
    rtwx_o0g3b_parser.add_argument("--g3r-cache", default="runs/rtwx_o0g3r")
    rtwx_o0g3b_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g3b_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g3a_parser = subparsers.add_parser(
        "rtwx-o0g3a",
        help="RTWX-O0G3A correspondence availability audit (descriptive)",
    )
    rtwx_o0g3a_parser.add_argument("--output", default="runs/rtwx_o0g3a")
    rtwx_o0g3a_parser.add_argument("--g3r-cache", default="runs/rtwx_o0g3r")
    rtwx_o0g3a_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g4_parser = subparsers.add_parser(
        "rtwx-o0g4",
        help="RTWX-O0G4 G0 oracle sparse CAD keypoints→Kabsch (not O1)",
    )
    rtwx_o0g4_parser.add_argument("--output", default="runs/rtwx_o0g4")
    rtwx_o0g4_parser.add_argument("--g3r-cache", default="runs/rtwx_o0g3r")
    rtwx_o0g4_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g4_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g4_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g5a_parser = subparsers.add_parser(
        "rtwx-o0g5a",
        help="RTWX-O0G5A CAD descriptor observability (frozen FPFH, no train)",
    )
    rtwx_o0g5a_parser.add_argument("--output", default="runs/rtwx_o0g5a")
    rtwx_o0g5a_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g5a_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0g5a_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g5b_parser = subparsers.add_parser(
        "rtwx-o0g5b",
        help="RTWX-O0G5B global-context CAD descriptor instrument (no O1)",
    )
    rtwx_o0g5b_parser.add_argument("--output", default="runs/rtwx_o0g5b")
    rtwx_o0g5b_parser.add_argument("--g5a-cache", default="runs/rtwx_o0g5a")
    rtwx_o0g5b_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g5b_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g5c_parser = subparsers.add_parser(
        "rtwx-o0g5c",
        help="RTWX-O0G5C fresh CAD-anchor identity generalization (no RANSAC)",
    )
    rtwx_o0g5c_parser.add_argument("--output", default="runs/rtwx_o0g5c")
    rtwx_o0g5c_parser.add_argument("--g5a-cache", default="runs/rtwx_o0g5a")
    rtwx_o0g5c_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g5c_parser.add_argument("--smoke", action="store_true")

    rtwx_o0g6a_parser = subparsers.add_parser(
        "rtwx-o0g6a",
        help="RTWX-O0G6A whole-object geometry orientation landscape (no ICP)",
    )
    rtwx_o0g6a_parser.add_argument("--output", default="runs/rtwx_o0g6a")
    rtwx_o0g6a_parser.add_argument("--g5c-cache", default="runs/rtwx_o0g5c")
    rtwx_o0g6a_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0g6a_parser.add_argument("--smoke", action="store_true")

    rtwx_o0a0_parser = subparsers.add_parser(
        "rtwx-o0a0",
        help="RTWX-O0A0 appearance-conditioned yaw observability (oracle mask; no full-R)",
    )
    rtwx_o0a0_parser.add_argument("--output", default="runs/rtwx_o0a0")
    rtwx_o0a0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0a0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0a0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0_parser = subparsers.add_parser(
        "rtwx-o0e0",
        help="RTWX-O0E0 symmetry-aware effective pose (p, axis); no yaw primary",
    )
    rtwx_o0e0_parser.add_argument("--output", default="runs/rtwx_o0e0")
    rtwx_o0e0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0e0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0p0_parser = subparsers.add_parser(
        "rtwx-o0e0p0",
        help="RTWX-O0E0P0 controlled-pose observation qualification (no B2)",
    )
    rtwx_o0e0p0_parser.add_argument("--output", default="runs/rtwx_o0e0p0")
    rtwx_o0e0p0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0p0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0e0p0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0r0_parser = subparsers.add_parser(
        "rtwx-o0e0r0",
        help="RTWX-O0E0R0 effective-pose science on O0E0P0 controlled cache (frozen B0/B1/B2)",
    )
    rtwx_o0e0r0_parser.add_argument("--output", default="runs/rtwx_o0e0r0")
    rtwx_o0e0r0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0e0r0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0r0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0r1_parser = subparsers.add_parser(
        "rtwx-o0e0r1",
        help="RTWX-O0E0R1 reference/centering oracle audit on P0 cache (no train)",
    )
    rtwx_o0e0r1_parser.add_argument("--output", default="runs/rtwx_o0e0r1")
    rtwx_o0e0r1_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0e0r1_parser.add_argument("--r0-cache", default="runs/rtwx_o0e0r0")
    rtwx_o0e0r1_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0r1_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0r2_parser = subparsers.add_parser(
        "rtwx-o0e0r2",
        help="RTWX-O0E0R2 controlled-pose segmentation repair + contingent science",
    )
    rtwx_o0e0r2_parser.add_argument("--output", default="runs/rtwx_o0e0r2")
    rtwx_o0e0r2_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0e0r2_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0e0r2_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0r2_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0r3_parser = subparsers.add_parser(
        "rtwx-o0e0r3",
        help="RTWX-O0E0R3 S0 formal effective-pose on fresh controlled test",
    )
    rtwx_o0e0r3_parser.add_argument("--output", default="runs/rtwx_o0e0r3")
    rtwx_o0e0r3_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0e0r3_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0e0r3_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0r3_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0r4_parser = subparsers.add_parser(
        "rtwx-o0e0r4",
        help="RTWX-O0E0R4 multi-observation reference test + center-lambda diagnostic",
    )
    rtwx_o0e0r4_parser.add_argument("--output", default="runs/rtwx_o0e0r4")
    rtwx_o0e0r4_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0e0r4_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0e0r4_parser.add_argument("--r3-cache", default="runs/rtwx_o0e0r3")
    rtwx_o0e0r4_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0r4_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0r5_parser = subparsers.add_parser(
        "rtwx-o0e0r5",
        help="RTWX-O0E0R5 visibility-aware object reference (frozen B2)",
    )
    rtwx_o0e0r5_parser.add_argument("--output", default="runs/rtwx_o0e0r5")
    rtwx_o0e0r5_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0e0r5_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0e0r5_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0r5_parser.add_argument("--smoke", action="store_true")

    rtwx_o0e0r6_parser = subparsers.add_parser(
        "rtwx-o0e0r6",
        help="RTWX-O0E0R6 finite-step alternating center-axis inference (inference-only)",
    )
    rtwx_o0e0r6_parser.add_argument("--output", default="runs/rtwx_o0e0r6")
    rtwx_o0e0r6_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0e0r6_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0e0r6_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0e0r6_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0e0r6_parser.add_argument("--smoke", action="store_true")

    rtwx_o0rel0_parser = subparsers.add_parser(
        "rtwx-o0rel0",
        help="RTWX-O0REL0 relative effective-pose tracking probe (C0/C1/C2 factorial)",
    )
    rtwx_o0rel0_parser.add_argument("--output", default="runs/rtwx_o0rel0")
    rtwx_o0rel0_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0rel0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0rel0_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0rel0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0rel0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0rel0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0rel0a_parser = subparsers.add_parser(
        "rtwx-o0rel0a",
        help="RTWX-O0REL0A ego-transform vs visibility-overlap audit (zero train)",
    )
    rtwx_o0rel0a_parser.add_argument("--output", default="runs/rtwx_o0rel0a")
    rtwx_o0rel0a_parser.add_argument("--rel0-run", default="runs/rtwx_o0rel0")
    rtwx_o0rel0a_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0rel0a_parser.add_argument("--smoke", action="store_true")

    rtwx_o0hyb0_parser = subparsers.add_parser(
        "rtwx-o0hyb0",
        help="RTWX-O0HYB0 prior-anchored hybrid one-step state update",
    )
    rtwx_o0hyb0_parser.add_argument("--output", default="runs/rtwx_o0hyb0")
    rtwx_o0hyb0_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0hyb0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0hyb0_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0hyb0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0hyb0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0hyb0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0seq0_parser = subparsers.add_parser(
        "rtwx-o0seq0",
        help="RTWX-O0SEQ0 relative propagation horizon audit",
    )
    rtwx_o0seq0_parser.add_argument("--output", default="runs/rtwx_o0seq0")
    rtwx_o0seq0_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0seq0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0seq0_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0seq0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0seq0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0seq0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0bel0_parser = subparsers.add_parser(
        "rtwx-o0bel0",
        help="RTWX-O0BEL0 relative belief / re-anchor observability audit",
    )
    rtwx_o0bel0_parser.add_argument("--output", default="runs/rtwx_o0bel0")
    rtwx_o0bel0_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0bel0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0bel0_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0bel0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0bel0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0bel0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0rab0_parser = subparsers.add_parser(
        "rtwx-o0rab0",
        help="RTWX-O0RAB0 re-anchor breadth bakeoff (P0, not scientific)",
    )
    rtwx_o0rab0_parser.add_argument("--output", default="runs/rtwx_o0rab0")
    rtwx_o0rab0_parser.add_argument("--bel0-cache", default="runs/rtwx_o0bel0")
    rtwx_o0rab0_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0rab0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0rab0_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0rab0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0rab0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0ra0_parser = subparsers.add_parser(
        "rtwx-o0ra0",
        help="RTWX-O0RA0 fresh confirmation of frozen RAB0 winner",
    )
    rtwx_o0ra0_parser.add_argument("--output", default="runs/rtwx_o0ra0")
    rtwx_o0ra0_parser.add_argument("--p0-run", default="runs/rtwx_o0rab0")
    rtwx_o0ra0_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0ra0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0ra0_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0ra0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0ra0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0ra0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0memb0_parser = subparsers.add_parser(
        "rtwx-o0memb0",
        help="RTWX-O0MEMB0 reference memory breadth probe (P0, not scientific)",
    )
    rtwx_o0memb0_parser.add_argument("--output", default="runs/rtwx_o0memb0")
    rtwx_o0memb0_parser.add_argument("--bel0-cache", default="runs/rtwx_o0bel0")
    rtwx_o0memb0_parser.add_argument("--natural-cache", default="runs/rtwx_o0e0")
    rtwx_o0memb0_parser.add_argument("--p0-cache", default="runs/rtwx_o0e0p0")
    rtwx_o0memb0_parser.add_argument("--r5-run", default="runs/rtwx_o0e0r5")
    rtwx_o0memb0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0memb0_parser.add_argument("--smoke", action="store_true")

    rtwx_mr0_parser = subparsers.add_parser(
        "rtwx-mr0",
        help="RTWX-MR0 multi-rate state–action resampling probe",
    )
    rtwx_mr0_parser.add_argument("--output", default="runs/rtwx_mr0")
    rtwx_mr0_parser.add_argument("--bel0-cache", default="runs/rtwx_o0bel0")
    rtwx_mr0_parser.add_argument("--device", default=None)
    rtwx_mr0_parser.add_argument("--workers", type=int, default=None)
    rtwx_mr0_parser.add_argument("--resume", action="store_true")
    rtwx_mr0_parser.add_argument("--smoke", action="store_true")

    rtwx_mr0_p0_parser = subparsers.add_parser(
        "rtwx-mr0-p0",
        help="RTWX-MR0-P0 action-chunk capability qualification (instrument)",
    )
    rtwx_mr0_p0_parser.add_argument("--output", default="runs/rtwx_mr0_p0")
    rtwx_mr0_p0_parser.add_argument("--device", default=None)
    rtwx_mr0_p0_parser.add_argument("--workers", type=int, default=None)
    rtwx_mr0_p0_parser.add_argument("--resume", action="store_true")
    rtwx_mr0_p0_parser.add_argument("--smoke", action="store_true")

    rtwx_mr0_a_parser = subparsers.add_parser(
        "rtwx-mr0-a",
        help="RTWX-MR0-A action resampling sweep (blocked until P0 qualifies)",
    )
    rtwx_mr0_a_parser.add_argument("--output", default="runs/rtwx_mr0_a")
    rtwx_mr0_a_parser.add_argument("--p0-run", default="runs/rtwx_mr0_p0")
    rtwx_mr0_a_parser.add_argument("--device", default=None)
    rtwx_mr0_a_parser.add_argument("--workers", type=int, default=None)
    rtwx_mr0_a_parser.add_argument("--resume", action="store_true")
    rtwx_mr0_a_parser.add_argument("--smoke", action="store_true")

    rtwx_ac0_parser = subparsers.add_parser(
        "rtwx-ac0",
        help="RTWX-AC0 explicit-state action-chunk provisioning (B0 vs B1 MLP)",
    )
    rtwx_ac0_parser.add_argument("--output", default="runs/rtwx_ac0")
    rtwx_ac0_parser.add_argument("--device", default=None)
    rtwx_ac0_parser.add_argument("--workers", type=int, default=None)
    rtwx_ac0_parser.add_argument("--resume", action="store_true")
    rtwx_ac0_parser.add_argument("--smoke", action="store_true")

    rtwx_ac1_d0_parser = subparsers.add_parser(
        "rtwx-ac1-d0",
        help="RTWX-AC1-D0 closed-loop state-coverage diagnosis (no training)",
    )
    rtwx_ac1_d0_parser.add_argument("--output", default="runs/rtwx_ac1_d0")
    rtwx_ac1_d0_parser.add_argument("--ac0-run", default="runs/rtwx_ac0")
    rtwx_ac1_d0_parser.add_argument("--device", default=None)
    rtwx_ac1_d0_parser.add_argument("--workers", type=int, default=None)
    rtwx_ac1_d0_parser.add_argument("--smoke", action="store_true")

    rtwx_ac1_d1_parser = subparsers.add_parser(
        "rtwx-ac1-d1",
        help="RTWX-AC1-D1 expert-only state-support metric audit (no policy, no AUROC selection)",
    )
    rtwx_ac1_d1_parser.add_argument("--output", default="runs/rtwx_ac1_d1")
    rtwx_ac1_d1_parser.add_argument("--ac0-run", default="runs/rtwx_ac0")
    rtwx_ac1_d1_parser.add_argument("--smoke", action="store_true")

    rtwx_ac2_parser = subparsers.add_parser(
        "rtwx-ac2",
        help="RTWX-AC2 policy-state sufficiency (P0 expert replay then Ha=1 B0/B1/B2)",
    )
    rtwx_ac2_parser.add_argument("--output", default="runs/rtwx_ac2")
    rtwx_ac2_parser.add_argument("--ac0-run", default="runs/rtwx_ac0")
    rtwx_ac2_parser.add_argument("--workers", type=int, default=None)
    rtwx_ac2_parser.add_argument("--smoke", action="store_true")

    rtwx_ac3_parser = subparsers.add_parser(
        "rtwx-ac3",
        help="RTWX-AC3 demonstration execution contract audit (native replay; no policy)",
    )
    rtwx_ac3_parser.add_argument("--output", default="runs/rtwx_ac3")
    rtwx_ac3_parser.add_argument("--ac0-run", default="runs/rtwx_ac0")
    rtwx_ac3_parser.add_argument("--ac2-run", default="runs/rtwx_ac2")
    rtwx_ac3_parser.add_argument("--workers", type=int, default=None)
    rtwx_ac3_parser.add_argument("--smoke", action="store_true")

    rtwx_ma0_parser = subparsers.add_parser(
        "rtwx-ma0",
        help="RTWX-MA0 official ACT/XPolicyLab positive control (P0 audit then official train/eval)",
    )
    rtwx_ma0_parser.add_argument("--output", default="runs/rtwx_ma0")
    rtwx_ma0_parser.add_argument("--p0-only", action="store_true")
    rtwx_ma0_parser.add_argument("--skip-train", action="store_true")
    rtwx_ma0_parser.add_argument("--conda-env", default="Robotwin")
    rtwx_ma0_parser.add_argument("--gpu-id", default="0")
    rtwx_ma0_parser.add_argument("--smoke", action="store_true")

    rtwx_task_x1_parser = subparsers.add_parser(
        "rtwx-task-x1",
        help="RTWX-TASK-X1 conditional diffusion chunk vs AC0-B0 (does not write MR0-P0)",
    )
    rtwx_task_x1_parser.add_argument("--output", default="runs/rtwx_task_x1")
    rtwx_task_x1_parser.add_argument("--ac0-run", default="runs/rtwx_ac0")
    rtwx_task_x1_parser.add_argument("--workers", type=int, default=None)
    rtwx_task_x1_parser.add_argument("--smoke", action="store_true")
    rtwx_task_x1_parser.add_argument("--stage-b-only", action="store_true")
    rtwx_task_x1_parser.add_argument("--b1-ckpt", default="runs/rtwx_task_x1_i1/B2/best.pt")

    rtwx_task_x1_i0_parser = subparsers.add_parser(
        "rtwx-task-x1-i0",
        help="RTWX-TASK-X1-I0 diffusion sampler closure (oracle epsilon; no clip; no Stage B)",
    )
    rtwx_task_x1_i0_parser.add_argument("--output", default="runs/rtwx_task_x1_i0")
    rtwx_task_x1_i0_parser.add_argument("--x1-run", default="runs/rtwx_task_x1")
    rtwx_task_x1_i0_parser.add_argument("--ac0-run", default="runs/rtwx_ac0")

    rtwx_task_x1_i1_parser = subparsers.add_parser(
        "rtwx-task-x1-i1",
        help="RTWX-TASK-X1-I1 diffusion parameterization probe (epsilon vs v vs x0; no Stage B)",
    )
    rtwx_task_x1_i1_parser.add_argument("--output", default="runs/rtwx_task_x1_i1")
    rtwx_task_x1_i1_parser.add_argument("--x1-run", default="runs/rtwx_task_x1")
    rtwx_task_x1_i1_parser.add_argument("--ac0-run", default="runs/rtwx_ac0")
    rtwx_task_x1_i1_parser.add_argument("--smoke", action="store_true")

    rtwx_o0q0_parser = subparsers.add_parser(
        "rtwx-o0q0",
        help="RTWX-O0Q0 task-object causal yaw audit (no RGB; does not change O0 target)",
    )
    rtwx_o0q0_parser.add_argument("--output", default="runs/rtwx_o0q0")
    rtwx_o0q0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0q0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0q0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0q0r0_parser = subparsers.add_parser(
        "rtwx-o0q0r0",
        help="RTWX-O0Q0R0 native counterfactual instrument (no yaw science; no teleport)",
    )
    rtwx_o0q0r0_parser.add_argument("--output", default="runs/rtwx_o0q0r0")
    rtwx_o0q0r0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0q0r0_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0q0r0_parser.add_argument("--smoke", action="store_true")

    rtwx_o0q0r0b_parser = subparsers.add_parser(
        "rtwx-o0q0r0b",
        help="RTWX-O0Q0R0B pre-contact branch instrument (no contact restore; no yaw science)",
    )
    rtwx_o0q0r0b_parser.add_argument("--output", default="runs/rtwx_o0q0r0b")
    rtwx_o0q0r0b_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0q0r0b_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0q0r0b_parser.add_argument("--smoke", action="store_true")

    rtwx_o0q0r0c_parser = subparsers.add_parser(
        "rtwx-o0q0r0c",
        help="RTWX-O0Q0R0C native demo action-trace + excitation qualification (no IK; no yaw science)",
    )
    rtwx_o0q0r0c_parser.add_argument("--output", default="runs/rtwx_o0q0r0c")
    rtwx_o0q0r0c_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0q0r0c_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0q0r0c_parser.add_argument("--smoke", action="store_true")

    rtwx_o0q0r0d_parser = subparsers.add_parser(
        "rtwx-o0q0r0d",
        help="RTWX-O0Q0R0D native control-channel audit (no IK; no hdf5-as-action; no yaw science)",
    )
    rtwx_o0q0r0d_parser.add_argument("--output", default="runs/rtwx_o0q0r0d")
    rtwx_o0q0r0d_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0q0r0d_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0q0r0d_parser.add_argument("--smoke", action="store_true")

    rtwx_o0q0r0e_parser = subparsers.add_parser(
        "rtwx-o0q0r0e",
        help="RTWX-O0Q0R0E native dense-trace acquisition (no new planner; no yaw science)",
    )
    rtwx_o0q0r0e_parser.add_argument("--output", default="runs/rtwx_o0q0r0e")
    rtwx_o0q0r0e_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    rtwx_o0q0r0e_parser.add_argument("--backend", default="robotwin", choices=("robotwin", "numpy"))
    rtwx_o0q0r0e_parser.add_argument("--smoke", action="store_true")

    sym_x0_parser = subparsers.add_parser(
        "sym-x0",
        help="SYM-X0 causal symmetry discovery (oracle state; no RGB)",
    )
    sym_x0_parser.add_argument("--output", default="runs/symx_x0")
    sym_x0_parser.add_argument("--smoke", action="store_true")

    sym_x1_parser = subparsers.add_parser(
        "sym-x1",
        help="SYM-X1 quotient representation utility (oracle state; no RGB)",
    )
    sym_x1_parser.add_argument("--output", default="runs/symx_x1")
    sym_x1_parser.add_argument("--x0-summary", default="runs/symx_x0")
    sym_x1_parser.add_argument("--smoke", action="store_true")

    sym_x2_parser = subparsers.add_parser(
        "sym-x2",
        help="SYM-X2 symmetry breaking and gauge reactivation (oracle state; no RGB)",
    )
    sym_x2_parser.add_argument("--output", default="runs/symx_x2")
    sym_x2_parser.add_argument("--x0-summary", default="runs/symx_x0")
    sym_x2_parser.add_argument("--x1-summary", default="runs/symx_x1")
    sym_x2_parser.add_argument("--smoke", action="store_true")

    task_x0_parser = subparsers.add_parser(
        "task-x0",
        help="TASK-X0 oracle progress instrument (no diffusion; not X1; not R10)",
    )
    task_x0_parser.add_argument(
        "--task",
        required=True,
        choices=["place_empty_cup", "put_object_cabinet", "stamp_seal"],
    )
    task_x0_parser.add_argument("--output", default=None)
    task_x0_parser.add_argument("--robotwin-repo", default="/root/RoboTwin")
    task_x0_parser.add_argument("--n-demo", type=int, default=None)
    task_x0_parser.add_argument("--max-seed-attempts", type=int, default=None)
    task_x0_parser.add_argument("--seed", type=int, default=9001)

    plan_x0_parser = subparsers.add_parser(
        "plan-x0",
        help="PLAN-X0 teacher/sensitivity instrument (no proposal claim; not R10)",
    )
    plan_x0_parser.add_argument("--output", default="runs/plan_x0/formal")
    plan_x0_parser.add_argument("--train-scenes", type=int, default=128)
    plan_x0_parser.add_argument("--val-scenes", type=int, default=32)
    plan_x0_parser.add_argument("--test-scenes", type=int, default=64)
    plan_x0_parser.add_argument("--targets-per-scene", type=int, default=32)

    plan_x1_parser = subparsers.add_parser(
        "plan-x1",
        help="PLAN-X1 sensitivity-aware Gaussian proposal (H1/H2; not R10)",
    )
    plan_x1_parser.add_argument("--output", default="runs/plan_x1/formal")
    plan_x1_parser.add_argument("--x0-data", default="runs/plan_x0/formal")

    plan_x15_parser = subparsers.add_parser(
        "plan-x15",
        help="PLAN-X1.5 learned action density vs mean+iso (not Hessian; not R10)",
    )
    plan_x15_parser.add_argument("--output", default="runs/plan_x15/formal")
    plan_x15_parser.add_argument("--x0-data", default="runs/plan_x0/formal")
    plan_x15_parser.add_argument("--skip-cem", action="store_true")
    plan_x15_parser.add_argument("--max-eval-conditions", type=int, default=None)

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
    if args.command == "sim-x0":
        from .sim_x0 import SIMX0Config, run_sim_x0

        result = run_sim_x0(
            args.output,
            config=SIMX0Config(duration_s=args.duration),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "sim-x1":
        from .sim_x1 import SIMX1Config, run_sim_x1

        result = run_sim_x1(
            args.output,
            config=SIMX1Config(duration_s=args.duration),
            x0_summary=args.x0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "sim-x2":
        from .sim_x2 import SIMX2Config, run_sim_x2

        result = run_sim_x2(
            args.output,
            config=SIMX2Config(duration_s=args.duration),
            x1_summary=args.x1_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "sim-x3":
        from .sim_x3 import SIMX3Config, run_sim_x3

        result = run_sim_x3(
            args.output,
            config=SIMX3Config(n_noise_seeds=args.noise_seeds),
            x2_summary=args.x2_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "vis-x0":
        from .vis_x0 import VISX0Config, run_vis_x0

        result = run_vis_x0(
            args.output,
            config=VISX0Config(duration_s=args.duration),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "vis-x1":
        from .vis_x1 import VISX1Config, run_vis_x1

        result = run_vis_x1(
            args.output,
            config=VISX1Config(duration_s=args.duration),
            x0_summary=args.x0_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "vis-x2":
        from .vis_x2 import VISX2Config, run_vis_x2

        result = run_vis_x2(
            args.output,
            config=VISX2Config(duration_s=args.duration),
            x1_summary=args.x1_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "vis-x3":
        from .vis_x3 import VISX3Config, run_vis_x3

        result = run_vis_x3(
            args.output,
            config=VISX3Config(duration_s=args.duration),
            x2_summary=args.x2_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "vis-ext0":
        from .vis_ext0 import VISEXT0Config, run_vis_ext0

        result = run_vis_ext0(
            args.output,
            config=VISEXT0Config(duration_s=args.duration),
            x3_summary=args.x3_summary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "cap-x0":
        from .cap_x0 import CAPX0Config, run_cap_x0

        result = run_cap_x0(
            args.output,
            config=CAPX0Config(
                duration_s=args.duration,
                n_train_scenes=args.train_scenes,
                n_val_scenes=args.val_scenes,
                n_test_scenes=args.test_scenes,
                traj_per_scene=args.traj_per_scene,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "cap-x1":
        from .cap_x1 import CAPX1Config, run_cap_x1

        result = run_cap_x1(
            args.output,
            config=CAPX1Config(
                x0_data=args.x0_data,
                plan_tasks=args.plan_tasks,
                rollout_episodes=args.rollout_episodes,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "cap-x2-p0":
        from .cap_x2_p0 import CAPX2P0Config, run_cap_x2_p0

        result = run_cap_x2_p0(
            args.output,
            config=CAPX2P0Config(n_tasks=args.n_tasks),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "cap-x2":
        from .cap_x2 import CAPX2Config, run_cap_x2

        result = run_cap_x2(
            args.output,
            config=CAPX2Config(p0_dir=args.p0_dir),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "cap-x3-p0":
        from .cap_x3_p0 import CAPX3P0Config, run_cap_x3_p0

        result = run_cap_x3_p0(args.output, config=CAPX3P0Config())
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "cap-x3":
        from .cap_x3 import CAPX3Config, run_cap_x3

        result = run_cap_x3(args.output, config=CAPX3Config(p0_dir=args.p0_dir))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0-p0":
        from .rtwx_x0_p0 import RTWX0P0Config, run_rtwx_x0_p0

        result = run_rtwx_x0_p0(args.output, config=RTWX0P0Config())
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0-smoke":
        from .rtwx_x0_smoke import RTWX0SmokeConfig, run_rtwx_x0_smoke

        result = run_rtwx_x0_smoke(
            args.output,
            config=RTWX0SmokeConfig(robotwin_repo=args.robotwin_repo, seed=args.seed),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0":
        from .rtwx_x0 import RTWX0Config, run_rtwx_x0

        result = run_rtwx_x0(
            args.output,
            config=RTWX0Config(
                smoke_summary=args.smoke_summary,
                robotwin_repo=args.robotwin_repo,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0r":
        from .rtwx_x0r import RTWX0RConfig, run_rtwx_x0r

        result = run_rtwx_x0r(
            args.output,
            config=RTWX0RConfig(output=args.output, robotwin_repo=args.robotwin_repo),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0s":
        from .rtwx_x0s import RTWX0SConfig, run_rtwx_x0s

        result = run_rtwx_x0s(
            args.output,
            config=RTWX0SConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                apply_mode=args.apply_mode,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0f":
        from .rtwx_x0f import RTWX0FConfig, run_rtwx_x0f

        result = run_rtwx_x0f(
            args.output,
            config=RTWX0FConfig(output=args.output, robotwin_repo=args.robotwin_repo, backend=args.backend),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0c":
        from .rtwx_x0c import RTWX0CConfig, run_rtwx_x0c

        if args.smoke:
            cfg = RTWX0CConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=True,
                n_train_ep=1,
                n_val_ep=1,
                n_test_ep=1,
                n_steps=8,
            )
        else:
            cfg = RTWX0CConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_x0c(args.output, config=cfg)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0d":
        from .rtwx_x0d import RTWX0DConfig, run_rtwx_x0d

        if args.smoke:
            cfg_d = RTWX0DConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=True,
                n_train_ep=1,
                n_val_ep=1,
                n_test_ep=1,
                n_steps=8,
            )
        else:
            cfg_d = RTWX0DConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_x0d(args.output, config=cfg_d)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0e":
        from .rtwx_x0e import RTWX0EConfig, run_rtwx_x0e

        if args.smoke:
            cfg_e = RTWX0EConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=True,
                n_train_ep=1,
                n_val_ep=1,
                n_test_ep=1,
                n_steps=8,
            )
        else:
            cfg_e = RTWX0EConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_x0e(args.output, config=cfg_e)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0e1":
        from .rtwx_x0e1 import RTWX0E1Config, run_rtwx_x0e1

        if args.smoke:
            cfg_e1 = RTWX0E1Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend="numpy",
                smoke=True,
                n_train_ep=4,
                n_val_ep=3,
                n_test_ep=3,
                n_steps=80,
                pure_widths=(8, 16),
                res_widths=(4, 8),
                train_seeds=(201,),
                epochs=12,
            )
        else:
            cfg_e1 = RTWX0E1Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                x0e_summary=args.x0e_summary,
            )
        result = run_rtwx_x0e1(args.output, config=cfg_e1)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0es":
        from .rtwx_x0es import RTWX0ESConfig, run_rtwx_x0es

        result = run_rtwx_x0es(
            args.output,
            config=RTWX0ESConfig(output=args.output, cache=args.cache, x0e1_summary=args.x0e1_summary),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0es1":
        from .rtwx_x0es1 import RTWX0ES1Config, run_rtwx_x0es1

        if args.smoke:
            cfg_es1 = RTWX0ES1Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_ep=6,
                n_train_ep=6,
                n_steps=80,
                seed=11,
                old_cache=args.old_cache,
            )
        else:
            cfg_es1 = RTWX0ES1Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                old_cache=args.old_cache,
            )
        result = run_rtwx_x0es1(args.output, config=cfg_es1)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0es2":
        from .rtwx_x0es2 import RTWX0ES2Config, run_rtwx_x0es2

        if args.smoke:
            cfg_es2 = RTWX0ES2Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_ep=6,
                n_train_ep=6,
                n_steps=80,
                seed=11,
                epochs=2,
                select_only=bool(args.select_only),
                old_cache=args.old_cache,
            )
        else:
            cfg_es2 = RTWX0ES2Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                old_cache=args.old_cache,
                select_only=bool(args.select_only),
            )
        result = run_rtwx_x0es2(args.output, config=cfg_es2)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0eh":
        from .rtwx_x0eh import RTWX0EHConfig, run_rtwx_x0eh

        if args.smoke:
            cfg_eh = RTWX0EHConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_ep=6,
                n_train_ep=6,
                n_steps=80,
                seed=11,
                old_cache=args.old_cache,
            )
        else:
            cfg_eh = RTWX0EHConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                old_cache=args.old_cache,
            )
        result = run_rtwx_x0eh(args.output, config=cfg_eh)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0eh1":
        from .rtwx_x0eh1 import RTWX0EH1Config, run_rtwx_x0eh1

        if args.smoke:
            cfg_eh1 = RTWX0EH1Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=6,
                n_val_ep=3,
                n_test_ep=3,
                n_steps=80,
                seed=11,
                pure_widths=(8, 16),
                train_seeds=(201,),
                epochs=2,
            )
        else:
            cfg_eh1 = RTWX0EH1Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_x0eh1(args.output, config=cfg_eh1)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0eh2":
        from .rtwx_x0eh2 import RTWX0EH2Config, run_rtwx_x0eh2

        if args.smoke:
            cfg_eh2 = RTWX0EH2Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=6,
                n_val_ep=3,
                n_test_ep=3,
                n_steps=80,
                pure_widths=(8, 16),
                train_seeds=(201,),
                epochs=2,
                tasks_only=("place_empty_cup", "stamp_seal"),
            )
        else:
            cfg_eh2 = RTWX0EH2Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                include_exploratory=args.include_exploratory,
                tasks_only=tuple(args.tasks_only) if args.tasks_only else None,
            )
        result = run_rtwx_x0eh2(args.output, config=cfg_eh2)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0rgb":
        from .rtwx_x0rgb import RTWX0RGBConfig, run_rtwx_x0rgb

        if args.smoke:
            cfg_rgb = RTWX0RGBConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=6,
                n_val_ep=3,
                n_test_ep=3,
                n_steps=80,
                seed=11,
                vis_epochs=5,
                b2_hidden=64,
            )
        else:
            cfg_rgb = RTWX0RGBConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_x0rgb(args.output, config=cfg_rgb)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-x0rgb1":
        from .rtwx_x0rgb1 import RTWX0RGB1Config, run_rtwx_x0rgb1

        if args.smoke:
            cfg_rgb1 = RTWX0RGB1Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=4,
                n_val_ep=2,
                n_test_ep=2,
                n_steps=40,
                seed=11,
                vis_epochs=3,
                rgb_size=32,
                cnn_ch=(8, 16, 32, 32),
            )
        else:
            cfg_rgb1 = RTWX0RGB1Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_x0rgb1(args.output, config=cfg_rgb1)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0":
        from .rtwx_o0 import RTWXO0Config, run_rtwx_o0

        if args.smoke:
            cfg_o0 = RTWXO0Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=3,
                n_val_ep=2,
                n_test_ep=2,
                n_steps=24,
                seed=11,
                vis_epochs=3,
                rgb_size=32,
            )
        else:
            cfg_o0 = RTWXO0Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0(args.output, config=cfg_o0)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0d":
        from .rtwx_o0d import RTWXO0DConfig, run_rtwx_o0d

        if args.smoke:
            cfg_o0d = RTWXO0DConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=3,
                n_val_ep=2,
                n_test_ep=2,
                n_steps=16,
                seed=13,
                vis_epochs=3,
                mem_n=32,
                mem_epochs=12,
                rgb_size=32,
            )
        else:
            cfg_o0d = RTWXO0DConfig(
                output=args.output,
                o0_cache=args.o0_cache,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0d(args.output, config=cfg_o0d)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0d1":
        from .rtwx_o0d1 import RTWXO0D1Config, run_rtwx_o0d1

        if args.smoke:
            cfg_d1 = RTWXO0D1Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=2,
                n_steps=8,
                seed=17,
                rgb_size=32,
                n_scalar=16,
                n_pose=24,
                epoch_scalar=80,
                epoch_pose=40,
            )
        else:
            cfg_d1 = RTWXO0D1Config(
                output=args.output,
                o0_cache=args.o0_cache,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0d1(args.output, config=cfg_d1)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0d2":
        from .rtwx_o0d2 import RTWXO0D2Config, run_rtwx_o0d2

        if args.smoke:
            cfg_d2 = RTWXO0D2Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=2,
                n_steps=8,
                seed=19,
                n_mem=16,
                epoch_lut=400,
                epoch_mlp=400,
                epoch_cnn=40,
            )
        else:
            cfg_d2 = RTWXO0D2Config(
                output=args.output,
                o0_cache=args.o0_cache,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0d2(args.output, config=cfg_d2)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0d3":
        from .rtwx_o0d3 import RTWXO0D3Config, run_rtwx_o0d3

        if args.smoke:
            cfg_d3 = RTWXO0D3Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=2,
                n_steps=16,
                seed=21,
                rgb_size=64,
                n_scalar=16,
                epoch_s=80,
                epoch_p=40,
                epoch_r=40,
            )
        else:
            cfg_d3 = RTWXO0D3Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0d3(args.output, config=cfg_d3)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0r":
        from .rtwx_o0r import RTWXO0RConfig, run_rtwx_o0r

        if args.smoke:
            cfg_o0r = RTWXO0RConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=3,
                n_val_ep=2,
                n_test_ep=2,
                n_steps=24,
                seed=23,
                rgb_size=64,
                vis_epochs=6,
            )
        else:
            cfg_o0r = RTWXO0RConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0r(args.output, config=cfg_o0r)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g":
        from .rtwx_o0g import RTWXO0GConfig, run_rtwx_o0g

        if args.smoke:
            cfg_o0g = RTWXO0GConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                stage=args.stage,
                n_ep=2,
                n_steps=16,
                seed=29,
            )
        else:
            cfg_o0g = RTWXO0GConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                stage=args.stage,
            )
        result = run_rtwx_o0g(args.output, config=cfg_o0g)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g1b":
        from .rtwx_o0g1b import RTWXO0G1BConfig, run_rtwx_o0g1b

        if args.smoke:
            cfg_g1b = RTWXO0G1BConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_ep=2,
                n_steps=16,
                seed=31,
            )
        else:
            cfg_g1b = RTWXO0G1BConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0g1b(args.output, config=cfg_g1b)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g2":
        from .rtwx_o0g2 import RTWXO0G2Config, run_rtwx_o0g2

        if args.smoke:
            cfg_g2 = RTWXO0G2Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=3,
                n_val_ep=2,
                n_test_ep=2,
                n_steps=24,
                seed=37,
                rgb_size=64,
                epochs=8,
            )
        else:
            cfg_g2 = RTWXO0G2Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0g2(args.output, config=cfg_g2)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0v":
        from .rtwx_o0v import RTWXO0VConfig, run_rtwx_o0v

        if args.smoke:
            cfg_v = RTWXO0VConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                seeds=(11, 12, 13),
                n_ep=2,
                n_steps=32,
            )
        else:
            cfg_v = RTWXO0VConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0v(args.output, config=cfg_v)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g2r":
        from .rtwx_o0g2r import RTWXO0G2RConfig, run_rtwx_o0g2r

        if args.smoke:
            cfg_g2r = RTWXO0G2RConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=3,
                n_val_ep=2,
                n_test_ep=2,
                n_steps=16,
                seed=41,
                rgb_size=64,
                epochs=8,
            )
        else:
            cfg_g2r = RTWXO0G2RConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0g2r(args.output, config=cfg_g2r)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0c":
        from .rtwx_o0c import RTWXO0CConfig, run_rtwx_o0c

        if args.smoke:
            cfg_o0c = RTWXO0CConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                n_train_ep=2,
                n_val_ep=1,
                n_test_ep=1,
                n_steps=12,
                seeds=(41, 42),
                rgb_size=64,
                epochs_unet=6,
                epochs_ori=8,
            )
        else:
            cfg_o0c = RTWXO0CConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0c(args.output, config=cfg_o0c)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0c1":
        from .rtwx_o0c1 import RTWXO0C1Config, run_rtwx_o0c1

        if args.smoke:
            cfg_c1 = RTWXO0C1Config(
                output=args.output,
                o0c_cache=args.o0c_cache,
                smoke=True,
                seeds=(41, 42),
                epochs_ori=6,
            )
        else:
            cfg_c1 = RTWXO0C1Config(output=args.output, o0c_cache=args.o0c_cache)
        result = run_rtwx_o0c1(args.output, config=cfg_c1)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g3":
        from .rtwx_o0g3 import RTWXO0G3Config, run_rtwx_o0g3

        if args.smoke:
            cfg_g3 = RTWXO0G3Config(
                output=args.output,
                backend="numpy",
                smoke=True,
                seeds=(41, 42),
                n_ep=2,
                n_steps=16,
            )
        else:
            cfg_g3 = RTWXO0G3Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0g3(args.output, config=cfg_g3)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g3r":
        from .rtwx_o0g3r import RTWXO0G3RConfig, run_rtwx_o0g3r

        if args.smoke:
            cfg_g3r = RTWXO0G3RConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                seeds=(41,),
                n_train_ep=3,
                n_val_ep=2,
                n_test_ep=2,
                n_steps=16,
                epochs_mask=6,
                epochs_corr=8,
                epochs_b0=8,
            )
        else:
            cfg_g3r = RTWXO0G3RConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0g3r(args.output, config=cfg_g3r)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g3b":
        from .rtwx_o0g3b import RTWXO0G3BConfig, run_rtwx_o0g3b

        cfg_g3b = RTWXO0G3BConfig(
            output=args.output,
            g3r_cache=args.g3r_cache,
            robotwin_repo=args.robotwin_repo,
            smoke=bool(args.smoke),
        )
        result = run_rtwx_o0g3b(args.output, config=cfg_g3b)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g3a":
        from .rtwx_o0g3a import RTWXO0G3AConfig, run_rtwx_o0g3a

        cfg_g3a = RTWXO0G3AConfig(output=args.output, g3r_cache=args.g3r_cache, smoke=bool(args.smoke))
        result = run_rtwx_o0g3a(args.output, config=cfg_g3a)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g4":
        from .rtwx_o0g4 import RTWXO0G4Config, run_rtwx_o0g4

        if args.smoke:
            cfg_g4 = RTWXO0G4Config(output=args.output, backend="numpy", smoke=True, seeds=(41, 42))
        else:
            cfg_g4 = RTWXO0G4Config(
                output=args.output,
                g3r_cache=args.g3r_cache,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0g4(args.output, config=cfg_g4)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g5a":
        from .rtwx_o0g5a import RTWXO0G5AConfig, run_rtwx_o0g5a

        if args.smoke:
            cfg_g5a = RTWXO0G5AConfig(
                output=args.output,
                backend="numpy",
                smoke=True,
                seeds=(41,),
                n_test_ep=2,
                n_steps=16,
                rgb_size=32,
            )
        else:
            cfg_g5a = RTWXO0G5AConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
            )
        result = run_rtwx_o0g5a(args.output, config=cfg_g5a)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g5b":
        from .rtwx_o0g5b import RTWXO0G5BConfig, run_rtwx_o0g5b

        cfg_g5b = RTWXO0G5BConfig(
            output=args.output,
            g5a_cache=args.g5a_cache,
            robotwin_repo=args.robotwin_repo,
            smoke=bool(args.smoke),
        )
        result = run_rtwx_o0g5b(args.output, config=cfg_g5b)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g5c":
        from .rtwx_o0g5c import RTWXO0G5CConfig, run_rtwx_o0g5c

        cfg_g5c = RTWXO0G5CConfig(
            output=args.output,
            g5a_cache=args.g5a_cache,
            robotwin_repo=args.robotwin_repo,
            smoke=bool(args.smoke),
        )
        result = run_rtwx_o0g5c(args.output, config=cfg_g5c)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0g6a":
        from .rtwx_o0g6a import RTWXO0G6AConfig, run_rtwx_o0g6a

        cfg_g6a = RTWXO0G6AConfig(
            output=args.output,
            g5c_cache=args.g5c_cache,
            robotwin_repo=args.robotwin_repo,
            smoke=bool(args.smoke),
        )
        result = run_rtwx_o0g6a(args.output, config=cfg_g6a)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0a0":
        from .rtwx_o0a0 import RTWXO0A0Config, run_rtwx_o0a0

        result = run_rtwx_o0a0(
            args.output,
            config=RTWXO0A0Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0":
        from .rtwx_o0e0 import RTWXO0E0Config, run_rtwx_o0e0

        result = run_rtwx_o0e0(
            args.output,
            config=RTWXO0E0Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0p0":
        from .rtwx_o0e0p0 import RTWXO0E0P0Config, run_rtwx_o0e0p0

        result = run_rtwx_o0e0p0(
            args.output,
            config=RTWXO0E0P0Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0r0":
        from .rtwx_o0e0r0 import RTWXO0E0R0Config, run_rtwx_o0e0r0

        result = run_rtwx_o0e0r0(
            args.output,
            config=RTWXO0E0R0Config(
                output=args.output,
                p0_cache=args.p0_cache,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0r1":
        from .rtwx_o0e0r1 import RTWXO0E0R1Config, run_rtwx_o0e0r1

        result = run_rtwx_o0e0r1(
            args.output,
            config=RTWXO0E0R1Config(
                output=args.output,
                p0_cache=args.p0_cache,
                r0_cache=args.r0_cache,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0r2":
        from .rtwx_o0e0r2 import RTWXO0E0R2Config, run_rtwx_o0e0r2

        result = run_rtwx_o0e0r2(
            args.output,
            config=RTWXO0E0R2Config(
                output=args.output,
                p0_cache=args.p0_cache,
                natural_cache=args.natural_cache,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0r3":
        from .rtwx_o0e0r3 import RTWXO0E0R3Config, run_rtwx_o0e0r3

        result = run_rtwx_o0e0r3(
            args.output,
            config=RTWXO0E0R3Config(
                output=args.output,
                p0_cache=args.p0_cache,
                natural_cache=args.natural_cache,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0r4":
        from .rtwx_o0e0r4 import RTWXO0E0R4Config, run_rtwx_o0e0r4

        result = run_rtwx_o0e0r4(
            args.output,
            config=RTWXO0E0R4Config(
                output=args.output,
                p0_cache=args.p0_cache,
                natural_cache=args.natural_cache,
                r3_cache=args.r3_cache,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0r5":
        from .rtwx_o0e0r5 import RTWXO0E0R5Config, run_rtwx_o0e0r5

        result = run_rtwx_o0e0r5(
            args.output,
            config=RTWXO0E0R5Config(
                output=args.output,
                p0_cache=args.p0_cache,
                natural_cache=args.natural_cache,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0e0r6":
        from .rtwx_o0e0r6 import RTWXO0E0R6Config, run_rtwx_o0e0r6

        result = run_rtwx_o0e0r6(
            args.output,
            config=RTWXO0E0R6Config(
                output=args.output,
                r5_run=args.r5_run,
                p0_cache=args.p0_cache,
                natural_cache=args.natural_cache,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0rel0":
        from .rtwx_o0rel0 import RTWXO0REL0Config, run_rtwx_o0rel0

        result = run_rtwx_o0rel0(
            args.output,
            config=RTWXO0REL0Config(
                output=args.output,
                natural_cache=args.natural_cache,
                p0_cache=args.p0_cache,
                r5_run=args.r5_run,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0rel0a":
        from .rtwx_o0rel0a import RTWXO0REL0AConfig, run_rtwx_o0rel0a

        result = run_rtwx_o0rel0a(
            args.output,
            config=RTWXO0REL0AConfig(
                output=args.output,
                rel0_run=args.rel0_run,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0hyb0":
        from .rtwx_o0hyb0 import RTWXO0HYB0Config, run_rtwx_o0hyb0

        result = run_rtwx_o0hyb0(
            args.output,
            config=RTWXO0HYB0Config(
                output=args.output,
                natural_cache=args.natural_cache,
                p0_cache=args.p0_cache,
                r5_run=args.r5_run,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0seq0":
        from .rtwx_o0seq0 import RTWXO0SEQ0Config, run_rtwx_o0seq0

        result = run_rtwx_o0seq0(
            args.output,
            config=RTWXO0SEQ0Config(
                output=args.output,
                natural_cache=args.natural_cache,
                p0_cache=args.p0_cache,
                r5_run=args.r5_run,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0bel0":
        from .rtwx_o0bel0 import RTWXO0BEL0Config, run_rtwx_o0bel0

        result = run_rtwx_o0bel0(
            args.output,
            config=RTWXO0BEL0Config(
                output=args.output,
                natural_cache=args.natural_cache,
                p0_cache=args.p0_cache,
                r5_run=args.r5_run,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0rab0":
        from .rtwx_o0rab0 import RTWXO0RAB0Config, run_rtwx_o0rab0

        result = run_rtwx_o0rab0(
            args.output,
            config=RTWXO0RAB0Config(
                output=args.output,
                bel0_cache=args.bel0_cache,
                natural_cache=args.natural_cache,
                p0_cache=args.p0_cache,
                r5_run=args.r5_run,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0ra0":
        from .rtwx_o0ra0 import RTWXO0RA0Config, run_rtwx_o0ra0

        result = run_rtwx_o0ra0(
            args.output,
            config=RTWXO0RA0Config(
                output=args.output,
                p0_run=args.p0_run,
                natural_cache=args.natural_cache,
                p0_cache=args.p0_cache,
                r5_run=args.r5_run,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0memb0":
        from .rtwx_o0memb0 import RTWXO0MEMB0Config, run_rtwx_o0memb0

        result = run_rtwx_o0memb0(
            args.output,
            config=RTWXO0MEMB0Config(
                output=args.output,
                bel0_cache=args.bel0_cache,
                natural_cache=args.natural_cache,
                p0_cache=args.p0_cache,
                r5_run=args.r5_run,
                robotwin_repo=args.robotwin_repo,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-mr0":
        from .rtwx_mr0 import RTWXMR0Config, run_rtwx_mr0

        result = run_rtwx_mr0(
            args.output,
            config=RTWXMR0Config(
                output=args.output,
                bel0_cache=args.bel0_cache,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-mr0-p0":
        from .rtwx_mr0_p0 import RTWXMR0P0Config, run_rtwx_mr0_p0

        result = run_rtwx_mr0_p0(
            args.output,
            config=RTWXMR0P0Config(output=args.output, smoke=bool(args.smoke)),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-mr0-a":
        from .rtwx_mr0_a import RTWXMR0AConfig, run_rtwx_mr0_a

        result = run_rtwx_mr0_a(
            args.output,
            config=RTWXMR0AConfig(
                output=args.output,
                p0_run=args.p0_run,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-ac0":
        from .rtwx_ac0 import RTWXAC0Config, run_rtwx_ac0

        result = run_rtwx_ac0(
            args.output,
            config=RTWXAC0Config(
                output=args.output,
                smoke=bool(args.smoke),
                workers=int(args.workers) if args.workers else 8,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-ac1-d0":
        from .rtwx_ac1_d0 import RTWXAC1D0Config, run_rtwx_ac1_d0

        result = run_rtwx_ac1_d0(
            args.output,
            config=RTWXAC1D0Config(
                output=args.output,
                ac0_run=args.ac0_run,
                smoke=bool(args.smoke),
                workers=int(args.workers) if args.workers else 8,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-ac1-d1":
        from .rtwx_ac1_d1 import RTWXAC1D1Config, run_rtwx_ac1_d1

        result = run_rtwx_ac1_d1(
            args.output,
            config=RTWXAC1D1Config(output=args.output, ac0_run=args.ac0_run, smoke=bool(args.smoke)),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-ac2":
        from .rtwx_ac2 import RTWXAC2Config, run_rtwx_ac2

        result = run_rtwx_ac2(
            args.output,
            config=RTWXAC2Config(
                output=args.output,
                ac0_run=args.ac0_run,
                smoke=bool(args.smoke),
                workers=int(args.workers) if args.workers else 8,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-ac3":
        from .rtwx_ac3 import RTWXAC3Config, run_rtwx_ac3

        result = run_rtwx_ac3(
            args.output,
            config=RTWXAC3Config(
                output=args.output,
                ac0_run=args.ac0_run,
                ac2_run=args.ac2_run,
                smoke=bool(args.smoke),
                workers=int(args.workers) if args.workers else 8,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-ma0":
        from .rtwx_ma0 import RTWXMA0Config, run_rtwx_ma0

        result = run_rtwx_ma0(
            args.output,
            config=RTWXMA0Config(
                output=args.output,
                smoke=bool(args.smoke),
                p0_only=bool(args.p0_only),
                skip_train=bool(args.skip_train),
                conda_env=str(args.conda_env),
                gpu_id=str(args.gpu_id),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-task-x1":
        from .rtwx_task_x1 import RTWXTASKX1Config, run_rtwx_task_x1

        out = args.output
        if bool(args.stage_b_only) and args.output == "runs/rtwx_task_x1":
            out = "runs/rtwx_task_x1_sb"
        result = run_rtwx_task_x1(
            out,
            config=RTWXTASKX1Config(
                output=out,
                ac0_run=args.ac0_run,
                smoke=bool(args.smoke),
                workers=int(args.workers) if args.workers else 8,
                stage_b_only=bool(args.stage_b_only),
                b1_ckpt=str(args.b1_ckpt),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-task-x1-i0":
        from .rtwx_task_x1_i0 import RTWXTASKX1I0Config, run_rtwx_task_x1_i0

        result = run_rtwx_task_x1_i0(
            args.output,
            config=RTWXTASKX1I0Config(output=args.output, x1_run=args.x1_run, ac0_run=args.ac0_run),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-task-x1-i1":
        from .rtwx_task_x1_i1 import RTWXTASKX1I1Config, run_rtwx_task_x1_i1

        result = run_rtwx_task_x1_i1(
            args.output,
            config=RTWXTASKX1I1Config(
                output=args.output, x1_run=args.x1_run, ac0_run=args.ac0_run, smoke=bool(args.smoke)
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0q0":
        from .rtwx_o0q0 import RTWXO0Q0Config, run_rtwx_o0q0

        result = run_rtwx_o0q0(
            args.output,
            config=RTWXO0Q0Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0q0r0":
        from .rtwx_o0q0r0 import RTWXO0Q0R0Config, run_rtwx_o0q0r0

        result = run_rtwx_o0q0r0(
            args.output,
            config=RTWXO0Q0R0Config(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0q0r0b":
        from .rtwx_o0q0r0b import RTWXO0Q0R0BConfig, run_rtwx_o0q0r0b

        result = run_rtwx_o0q0r0b(
            args.output,
            config=RTWXO0Q0R0BConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0q0r0c":
        from .rtwx_o0q0r0c import RTWXO0Q0R0CConfig, run_rtwx_o0q0r0c

        result = run_rtwx_o0q0r0c(
            args.output,
            config=RTWXO0Q0R0CConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0q0r0d":
        from .rtwx_o0q0r0d import RTWXO0Q0R0DConfig, run_rtwx_o0q0r0d

        result = run_rtwx_o0q0r0d(
            args.output,
            config=RTWXO0Q0R0DConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "rtwx-o0q0r0e":
        from .rtwx_o0q0r0e import RTWXO0Q0R0EConfig, run_rtwx_o0q0r0e

        result = run_rtwx_o0q0r0e(
            args.output,
            config=RTWXO0Q0R0EConfig(
                output=args.output,
                robotwin_repo=args.robotwin_repo,
                backend=args.backend,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "sym-x0":
        from .symx_x0 import SYMX0Config, run_symx_x0

        result = run_symx_x0(args.output, config=SYMX0Config(output=args.output, smoke=bool(args.smoke)))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "sym-x1":
        from .symx_x1 import SYMX1Config, run_symx_x1

        result = run_symx_x1(
            args.output,
            config=SYMX1Config(output=args.output, x0_summary=args.x0_summary, smoke=bool(args.smoke)),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "sym-x2":
        from .symx_x2 import SYMX2Config, run_symx_x2

        result = run_symx_x2(
            args.output,
            config=SYMX2Config(
                output=args.output,
                x0_summary=args.x0_summary,
                x1_summary=args.x1_summary,
                smoke=bool(args.smoke),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "task-x0":
        from .task_x0 import N_DEMO, TASKX0Config, run_task_x0

        task = str(args.task)
        n_demo = N_DEMO if args.n_demo is None else int(args.n_demo)
        max_att = args.max_seed_attempts
        result = run_task_x0(
            args.output or f"runs/task_x0/{task}",
            config=TASKX0Config(
                task=task,
                output=args.output or f"runs/task_x0/{task}",
                robotwin_repo=args.robotwin_repo,
                n_demo=n_demo,
                seed_attempts=80 if max_att is None else int(max_att),
                max_seed_attempts=max_att,
                seed=int(args.seed),
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "plan-x0":
        from .plan_x0 import PLANX0Config, run_plan_x0

        result = run_plan_x0(
            args.output,
            config=PLANX0Config(
                n_train_scenes=args.train_scenes,
                n_val_scenes=args.val_scenes,
                n_test_scenes=args.test_scenes,
                targets_per_scene=args.targets_per_scene,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "plan-x1":
        from .plan_x1 import PLANX1Config, run_plan_x1

        result = run_plan_x1(
            args.output,
            config=PLANX1Config(x0_data=args.x0_data),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "plan-x15":
        from .plan_x15 import PLANX15Config, run_plan_x15

        result = run_plan_x15(
            args.output,
            config=PLANX15Config(
                x0_data=args.x0_data,
                skip_cem=bool(args.skip_cem),
                max_eval_conditions=args.max_eval_conditions,
            ),
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
