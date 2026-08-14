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
