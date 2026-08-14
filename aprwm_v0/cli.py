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
from .r1_ms import run_preflight as run_r1_ms_preflight
from .r1_ms_io import R1MSIOConfig, run_r1_ms_io_smoke


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

    r1_ms_parser = subparsers.add_parser(
        "r1-ms-preflight",
        help="Freeze and inspect the Drawer-only ManiSkill R1-MS0 protocol",
    )
    r1_ms_parser.add_argument("--output", default="runs/r1_ms/preflight")
    r1_ms_parser.add_argument(
        "--python", default=".venv-maniskill/bin/python"
    )
    r1_ms_parser.add_argument(
        "--asset-root", default=".maniskill"
    )

    r1_ms_io_parser = subparsers.add_parser(
        "r1-ms-io-smoke",
        help="Run asset-free state replay/branching plumbing without gate authority",
    )
    r1_ms_io_parser.add_argument("--output", default="runs/r1_ms/io_smoke")
    r1_ms_io_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[8201, 8211, 8221]
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
    if args.command == "r1-ms-preflight":
        result = run_r1_ms_preflight(
            args.output, python=args.python, asset_root=args.asset_root
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "r1-ms-io-smoke":
        result = run_r1_ms_io_smoke(
            args.output, config=R1MSIOConfig(seeds=tuple(args.seeds))
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
