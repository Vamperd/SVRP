#!/usr/bin/env python
"""Batch evaluation for static, hybrid, and traffic-aware TWCVRP RL."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dataset import load_instances, split_instances, instances_to_tensors
from decoder import decode_order
from evaluator import aggregate_metrics, evaluate_routes
from model import TWPointerPolicy, actions_to_order, rollout
from traffic import planning_matrix, sample_traffic_matrix, stable_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TWCVRP RL checkpoints.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--sizes", nargs="+", default=None)
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--decoder", default="strict_insert", choices=["strict_insert", "greedy_split"])
    parser.add_argument("--insert_top_k", type=int, default=30)
    parser.add_argument("--static_checkpoint", default=None)
    parser.add_argument("--traffic_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_buffer", type=float, default=0.50)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--output_json", default="results/comparison.json")
    parser.add_argument("--output_csv", default="results/comparison.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_checkpoint(template: str | None, size: str) -> Path | None:
    if not template:
        return None
    return Path(template.format(size=size))


def discover_split_sizes(split_root: str | Path) -> list[str]:
    root = Path(split_root) / "train"
    sizes = [path.name for path in root.glob("*") if path.is_dir()]
    return sorted(sizes, key=lambda value: int(value) if value.isdigit() else value)


def load_policy(path: Path, device: torch.device) -> TWPointerPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    context_dim = int(
        checkpoint.get(
            "context_dim",
            state.get("context_encoder.0.weight").shape[1],
        )
    )
    dynamic_feature_dim = int(
        checkpoint.get(
            "dynamic_feature_dim",
            state["dynamic_encoder.0.weight"].shape[1]
            if "dynamic_encoder.0.weight" in state
            else 0,
        )
    )
    model = TWPointerPolicy(
        embed_dim=int(checkpoint.get("embed_dim", 128)),
        feature_dim=int(checkpoint.get("feature_dim", 6)),
        context_dim=context_dim,
        dynamic_feature_dim=dynamic_feature_dim,
        model_version=checkpoint.get("model_version", "tw_pointer_legacy_v1"),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_eval_instances(args, size: str):
    if args.split_root:
        split_root = Path(args.split_root)
        if args.split == "all":
            items = []
            for split_name in ("train", "val", "test"):
                items.extend(
                    load_instances(
                        data_root=split_root / split_name,
                        size=size,
                        source="solomon",
                        pattern=args.pattern,
                        limit=args.limit,
                    )
                )
            return items
        return load_instances(
            data_root=split_root / args.split,
            size=size,
            source="solomon",
            pattern=args.pattern,
            limit=args.limit,
        )

    if not args.data_root:
        raise ValueError("Either --split_root or --data_root is required.")
    instances = load_instances(
        data_root=args.data_root,
        size=size,
        source=args.source,
        pattern=args.pattern,
        limit=args.limit,
    )
    split = split_instances(
        instances,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    return instances if args.split == "all" else split[args.split]


def policy_routes(model, instance, *, mode: str, args, device: torch.device) -> list[list[int]]:
    batch = instances_to_tensors([instance], device=device)
    with torch.no_grad():
        out = rollout(model, batch, decode="greedy")
    order = actions_to_order(out.actions[0])
    matrix = planning_matrix(
        instance,
        mode=mode,
        traffic_sigma=args.traffic_sigma,
        traffic_buffer=args.traffic_buffer,
    )
    return decode_order(
        instance,
        order,
        matrix,
        decoder=args.decoder,
        insert_top_k=args.insert_top_k,
    )


def evaluate_static(instance, routes, *, method: str, size: str) -> dict:
    matrix = planning_matrix(instance, mode="static")
    row = evaluate_routes(instance, routes, matrix)
    row["method"] = method
    row["size"] = str(size)
    return row


def evaluate_traffic_mc(instance, routes, *, method: str, size: str, args) -> dict:
    rows = []
    for sample_idx in range(max(1, args.mc_samples)):
        matrix = sample_traffic_matrix(
            instance,
            seed=stable_seed(instance.name, method, sample_idx, base_seed=args.traffic_seed),
            traffic_sigma=args.traffic_sigma,
        )
        rows.append(evaluate_routes(instance, routes, matrix))
    summary = aggregate_metrics(rows)
    summary.update(
        {
            "method": method,
            "size": str(size),
            "instance": instance.name,
            "source": instance.source,
            "routes": routes,
            "mc_samples": max(1, args.mc_samples),
            "feasible": bool(summary.get("feasibility", 0.0) >= 1.0),
        }
    )
    return summary


def flatten_summary(summary: dict) -> list[dict]:
    rows = []
    for size, methods in summary["aggregate"].items():
        for method, metrics in methods.items():
            row = {"size": size, "method": method}
            row.update(metrics)
            rows.append(row)
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row if key != "routes"})
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    if args.sizes:
        sizes = [str(size) for size in args.sizes]
    elif args.split_root:
        sizes = discover_split_sizes(args.split_root)
    else:
        raise ValueError("--sizes is required when --split_root is not provided.")
    summary = {"aggregate": {}, "per_instance": []}
    model_cache: dict[str, TWPointerPolicy] = {}

    for size in sizes:
        selected = load_eval_instances(args, str(size))
        summary["aggregate"].setdefault(str(size), {})

        static_model = None
        static_checkpoint = resolve_checkpoint(args.static_checkpoint, str(size))
        if static_checkpoint:
            cache_key = str(static_checkpoint)
            if cache_key not in model_cache:
                model_cache[cache_key] = load_policy(static_checkpoint, device)
            static_model = model_cache[cache_key]

        traffic_model = None
        traffic_checkpoint = resolve_checkpoint(args.traffic_checkpoint, str(size))
        if traffic_checkpoint:
            cache_key = str(traffic_checkpoint)
            if cache_key not in model_cache:
                model_cache[cache_key] = load_policy(traffic_checkpoint, device)
            traffic_model = model_cache[cache_key]

        method_rows: dict[str, list[dict]] = {"static_rl": [], "hybrid_rl": [], "traffic_rl": []}
        for instance in selected:
            if static_model is not None:
                static_routes = policy_routes(static_model, instance, mode="static", args=args, device=device)
                static_row = evaluate_static(instance, static_routes, method="static_rl", size=str(size))
                hybrid_row = evaluate_traffic_mc(
                    instance,
                    static_routes,
                    method="hybrid_rl",
                    size=str(size),
                    args=args,
                )
                method_rows["static_rl"].append(static_row)
                method_rows["hybrid_rl"].append(hybrid_row)
                summary["per_instance"].extend([static_row, hybrid_row])

            if traffic_model is not None:
                traffic_routes = policy_routes(
                    traffic_model,
                    instance,
                    mode="traffic",
                    args=args,
                    device=device,
                )
                traffic_row = evaluate_traffic_mc(
                    instance,
                    traffic_routes,
                    method="traffic_rl",
                    size=str(size),
                    args=args,
                )
                method_rows["traffic_rl"].append(traffic_row)
                summary["per_instance"].append(traffic_row)

        for method, rows in method_rows.items():
            if rows:
                summary["aggregate"][str(size)][method] = aggregate_metrics(rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_csv, flatten_summary(summary))
    print(f"[DONE] json={args.output_json} csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
