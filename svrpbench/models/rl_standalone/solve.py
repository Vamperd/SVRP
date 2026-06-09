#!/usr/bin/env python
"""Solve CVRP benchmark instances with a standalone PyTorch RL checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from dataset import evaluate_route, instances_to_tensors, load_cvrp_instances, split_instances
from model import PointerPolicy, actions_to_route, rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CVRP routes with a standalone RL policy.")
    parser.add_argument("--data", required=True, help="Path to a single-depot CVRP .npz file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--output", default="results/solutions.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_dir", default=None)
    parser.add_argument(
        "--allow_duplicate_openmp",
        action="store_true",
        help="Windows workaround for libomp/libiomp plot conflicts. Use only if --plot fails.",
    )
    parser.add_argument("--capacity_penalty", type=float, default=100.0)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_policy(path: str | Path, device: torch.device) -> tuple[PointerPolicy, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PointerPolicy(embed_dim=int(checkpoint["embed_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def solve_instance(
    model: PointerPolicy,
    instance,
    *,
    device: torch.device,
    capacity_penalty: float,
) -> dict:
    batch = instances_to_tensors([instance], device=device)
    with torch.no_grad():
        out = rollout(model, batch, decode="greedy", capacity_penalty=capacity_penalty)
    route = actions_to_route(out.actions[0])
    result = evaluate_route(instance, route)
    result["instance"] = instance.index
    result["source"] = instance.source
    result["model_distance"] = float(out.distances[0].detach().cpu().item())
    return result


def save_plot(instance, result: dict, path: str | Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for --plot.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    loc = instance.locations
    route = result["routes"][0]
    route_xy = loc[route]

    plt.figure(figsize=(6, 6))
    plt.scatter(loc[1:, 0], loc[1:, 1], c="#277da1", label="customers")
    plt.scatter(loc[0:1, 0], loc[0:1, 1], c="#f94144", marker="s", label="depot")
    plt.plot(route_xy[:, 0], route_xy[:, 1], c="#43aa8b", linewidth=1.5)
    for node in route:
        plt.text(loc[node, 0], loc[node, 1], str(node), fontsize=8)
    plt.title(f"instance {instance.index} distance={result['total_distance']:.1f}")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_duplicate_openmp:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = resolve_device(args.device)
    model, checkpoint = load_policy(args.checkpoint, device)
    instances = load_cvrp_instances(args.data, limit=args.limit)
    if args.split != "all":
        split = split_instances(
            instances,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )
        instances = split[args.split]

    expected_customers = int(checkpoint.get("num_customers", instances[0].num_customers))
    for instance in instances:
        if instance.num_customers != expected_customers:
            raise ValueError(
                f"Checkpoint expects {expected_customers} customers, "
                f"but instance {instance.index} has {instance.num_customers}."
            )

    results = []
    for instance in instances:
        result = solve_instance(
            model,
            instance,
            device=device,
            capacity_penalty=args.capacity_penalty,
        )
        results.append(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else output.parent / "plots"
        for instance, result in zip(instances, results):
            save_plot(instance, result, plot_dir / f"instance_{instance.index}.png")

    feasible = sum(1 for result in results if result["feasible"])
    avg_distance = sum(result["total_distance"] for result in results) / max(1, len(results))
    print(
        "[DONE] "
        f"output={output} instances={len(results)} feasible={feasible}/{len(results)} "
        f"avg_distance={avg_distance:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
