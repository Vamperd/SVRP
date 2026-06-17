#!/usr/bin/env python
"""Evaluate event-driven online recourse policies and baselines."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import ensure_dir, SOLOMON_TW_DIR  # noqa: F401
from dataset import load_instances, split_instances, instances_to_tensors
from decoder import decode_order
from env import EventDrivenTWEnv
from evaluator import evaluate_routes
from heuristic import run_heuristic
from model import TWPointerPolicy, actions_to_order, rollout as pointer_rollout
from policy import EventDrivenSTPolicy
from rollout import rollout_policy
from traffic import planning_matrix, sample_traffic_matrix, stable_seed


CSV_FIELDS = [
    "method",
    "size",
    "mode",
    "instances",
    "avg_cost",
    "single_customer_cost",
    "avg_cvr",
    "feasibility_rate",
    "late_minutes",
    "time_window_violations",
    "route_count",
    "forced_late_actions",
    "vehicles_excess",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate online recourse TWCVRP policies.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--sizes", nargs="+", default=["100"])
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--checkpoint", default=None, help="Event-driven recourse checkpoint.")
    parser.add_argument("--strict_checkpoint", default=None, help="Existing rl_solomon_tw pointer checkpoint.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["earliest_due", "min_late", "recourse"],
        choices=["earliest_due", "min_late", "recourse", "strict_insert", "sg_recourse"],
    )
    parser.add_argument("--mode", default="traffic", choices=["static", "traffic"])
    parser.add_argument("--strict_planning", default="static", choices=["static", "traffic"])
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_profile", default="additive", choices=["additive", "proportional"])
    parser.add_argument("--traffic_strength", type=float, default=1.0)
    parser.add_argument("--traffic_time_scale", default="depot_day", choices=["raw", "depot_day"])
    parser.add_argument("--mc_samples", type=int, default=5)
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask_late", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--insert_top_k", type=int, default=10)
    parser.add_argument("--allow_unseen_size", action="store_true")
    parser.add_argument("--output_json", default="results/recourse_eval.json")
    parser.add_argument("--output_csv", default="results/recourse_eval.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_eval_instances(args: argparse.Namespace) -> list:
    if args.input:
        instances = load_instances(input_path=args.input)
        if args.split == "all":
            return instances
        return split_instances(instances, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)[args.split]

    result = []
    for size in [str(size) for size in args.sizes]:
        if args.split_root:
            if args.split == "all":
                for split in ["train", "val", "test"]:
                    result.extend(load_instances(data_root=Path(args.split_root) / split, size=size, source=args.source))
            else:
                result.extend(load_instances(data_root=Path(args.split_root) / args.split, size=size, source=args.source))
        else:
            instances = load_instances(data_root=args.data_root, size=size, source=args.source)
            if args.split == "all":
                result.extend(instances)
            else:
                result.extend(
                    split_instances(instances, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)[
                        args.split
                    ]
                )
    return result


def load_recourse_policy(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    policy = EventDrivenSTPolicy(
        customer_feature_dim=int(config.get("customer_feature_dim", 14)),
        vehicle_feature_dim=int(config.get("vehicle_feature_dim", 9)),
        embed_dim=int(config.get("embed_dim", 128)),
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy, checkpoint


def load_strict_policy(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    context_weight = state.get("context_encoder.0.weight")
    dynamic_weight = state.get("dynamic_encoder.0.weight")
    policy = TWPointerPolicy(
        embed_dim=int(checkpoint.get("embed_dim", 128)),
        feature_dim=int(checkpoint.get("feature_dim", 6)),
        context_dim=int(checkpoint.get("context_dim", context_weight.shape[1] if context_weight is not None else 9)),
        dynamic_feature_dim=int(
            checkpoint.get("dynamic_feature_dim", dynamic_weight.shape[1] if dynamic_weight is not None else 0)
        ),
        model_version=checkpoint.get("model_version", "tw_pointer_dynamic_v2"),
    ).to(device)
    policy.load_state_dict(state)
    policy.eval()
    return policy, checkpoint


def matrix_for_eval(instance, args, *, sample_idx: int) -> np.ndarray:
    if args.mode == "static":
        return planning_matrix(instance, mode="static")
    return sample_traffic_matrix(
        instance,
        seed=stable_seed(instance.name, "eval", sample_idx, base_seed=args.traffic_seed),
        traffic_sigma=args.traffic_sigma,
        traffic_profile=args.traffic_profile,
        traffic_strength=args.traffic_strength,
    )


def strict_routes(policy, instance, args, device: torch.device) -> list[list[int]]:
    matrix = planning_matrix(
        instance,
        mode=args.strict_planning,
        traffic_sigma=args.traffic_sigma,
        traffic_profile=args.traffic_profile,
        traffic_strength=args.traffic_strength,
    )
    batch = instances_to_tensors([instance], device=device)
    with torch.no_grad():
        out = pointer_rollout(policy, batch, decode="greedy")
    order = actions_to_order(out.actions[0])
    return decode_order(instance, order, matrix, decoder="strict_insert", insert_top_k=args.insert_top_k)


def evaluate_method(method: str, instance, matrix, args, device, recourse_policy=None, strict_policy=None, sg_policy=None) -> dict:
    if method in {"earliest_due", "min_late"}:
        env = EventDrivenTWEnv(instance, matrix, mask_late=args.mask_late, traffic_time_scale=args.traffic_time_scale)
        row = run_heuristic(env, strategy=method)
    elif method == "recourse":
        if recourse_policy is None:
            raise ValueError("--checkpoint is required for method=recourse.")
        env = EventDrivenTWEnv(instance, matrix, mask_late=args.mask_late, traffic_time_scale=args.traffic_time_scale)
        with torch.no_grad():
            row = rollout_policy(recourse_policy, env, decode="greedy", device=device).metrics
    elif method == "sg_recourse":
        if sg_policy is None:
            raise ValueError("--checkpoint is required for method=sg_recourse.")
        env = EventDrivenTWEnv(instance, matrix, mask_late=args.mask_late, traffic_time_scale=args.traffic_time_scale)
        with torch.no_grad():
            row = rollout_policy(sg_policy, env, decode="greedy", device=device).metrics
    elif method == "strict_insert":
        if strict_policy is None:
            raise ValueError("--strict_checkpoint is required for method=strict_insert.")
        routes = strict_routes(strict_policy, instance, args, device)
        row = evaluate_routes(instance, routes, matrix)
        row["forced_late_actions"] = 0
    else:
        raise ValueError(f"Unknown method: {method}")
    row["method"] = method
    row["size"] = str(instance.num_customers)
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["size"]), []).append(row)
    output = []
    for (method, size), items in sorted(grouped.items(), key=lambda item: (int(item[0][1]), item[0][0])):
        size_int = int(size)
        avg_cost = float(np.mean([float(row.get("total_cost", 0.0)) for row in items]))
        output.append(
            {
                "method": method,
                "size": size,
                "mode": items[0].get("mode", ""),
                "instances": len(items),
                "avg_cost": avg_cost,
                "single_customer_cost": avg_cost / max(1, size_int),
                "avg_cvr": float(np.mean([float(row.get("cvr", 0.0)) for row in items])),
                "feasibility_rate": float(np.mean([1.0 if row.get("feasible") else 0.0 for row in items])),
                "late_minutes": float(np.mean([float(row.get("late_minutes", 0.0)) for row in items])),
                "time_window_violations": float(
                    np.mean([float(row.get("time_window_violations", 0.0)) for row in items])
                ),
                "route_count": float(np.mean([float(row.get("route_count", 0.0)) for row in items])),
                "forced_late_actions": float(np.mean([float(row.get("forced_late_actions", 0.0)) for row in items])),
                "vehicles_excess": float(np.mean([float(row.get("vehicles_excess", 0.0)) for row in items])),
            }
        )
    return output


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output = ensure_dir(path)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    recourse_policy = None
    strict_policy = None
    sg_policy = None
    if "recourse" in args.methods or "sg_recourse" in args.methods:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when methods includes recourse or sg_recourse.")
        if "sg_recourse" in args.methods:
            sg_policy, _ = load_recourse_policy(args.checkpoint, device)
        if "recourse" in args.methods:
            recourse_policy, recourse_checkpoint = load_recourse_policy(args.checkpoint, device)
            if not args.allow_unseen_size:
                supported = {str(size) for size in recourse_checkpoint.get("supported_sizes", [])}
                if supported:
                    unsupported = {str(i.num_customers) for i in load_eval_instances(args)} - supported
                    if unsupported:
                        raise ValueError(f"Recourse checkpoint does not support sizes: {sorted(unsupported)}")
    if "strict_insert" in args.methods:
        if not args.strict_checkpoint:
            raise ValueError("--strict_checkpoint is required when methods includes strict_insert.")
        strict_policy, _ = load_strict_policy(args.strict_checkpoint, device)

    instances = load_eval_instances(args)
    rows = []
    samples = 1 if args.mode == "static" else max(1, args.mc_samples)
    for instance in instances:
        for sample_idx in range(samples):
            matrix = matrix_for_eval(instance, args, sample_idx=sample_idx)
            for method in args.methods:
                row = evaluate_method(
                    method,
                    instance,
                    matrix,
                    args,
                    device,
                    recourse_policy=recourse_policy,
                    strict_policy=strict_policy,
                    sg_policy=sg_policy,
                )
                row["mode"] = args.mode
                row["sample_idx"] = sample_idx
                rows.append(row)

    aggregate_rows = aggregate(rows)
    output_json = ensure_dir(args.output_json)
    output_json.write_text(json.dumps({"aggregate": aggregate_rows, "per_instance": rows}, indent=2), encoding="utf-8")
    write_csv(args.output_csv, aggregate_rows)
    print(f"[DONE] json={args.output_json} csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

