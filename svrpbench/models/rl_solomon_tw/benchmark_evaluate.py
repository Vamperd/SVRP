#!/usr/bin/env python
"""Evaluate TWCVRP RL checkpoints with the PDF/OR-Tools benchmark metric style."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from dataset import load_instances, split_instances, instances_to_tensors
from decoder import decode_order
from evaluator import evaluate_routes
from model import TWPointerPolicy, actions_to_order, rollout
from traffic import planning_matrix, sample_traffic_matrix, stable_seed


CSV_FIELDS = [
    "eval_set",
    "model_name",
    "size",
    "mode",
    "metric_profile",
    "decoder",
    "post_opt",
    "traffic_profile",
    "traffic_strength",
    "traffic_time_scale",
    "instances",
    "avg_depot_due",
    "avg_traffic_edges",
    "avg_raw_current_time",
    "avg_scaled_current_time",
    "avg_delay",
    "avg_delay_ratio",
    "avg_cost",
    "single_customer_cost",
    "avg_waiting",
    "avg_cvr",
    "feasibility_rate",
    "avg_solver_runtime_s",
    "robustness_std",
    "avg_route_count",
    "avg_vehicles_excess",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TWCVRP checkpoints with PDF benchmark metrics.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--eval_files", nargs="+", default=None)
    parser.add_argument("--eval_set", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--sizes", nargs="+", default=None)
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--mode", default="static", choices=["static", "hybrid", "traffic"])
    parser.add_argument(
        "--metric_profile",
        default="pdf_compatible",
        choices=["pdf_compatible", "native"],
        help="pdf_compatible keeps the report metric style; native uses benchmark/project travel matrices.",
    )
    parser.add_argument(
        "--decoder",
        default="strict_insert",
        choices=["strict_insert", "deadline_aware_insert", "greedy_split"],
    )
    parser.add_argument("--insert_top_k", type=int, default=30)
    parser.add_argument("--post_opt", default="none", choices=["none", "time_window_repair"])
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more name=path checkpoint specs, e.g. static100=checkpoints/static.pt.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_buffer", type=float, default=0.50)
    parser.add_argument(
        "--traffic_profile",
        default="additive",
        choices=["additive", "proportional"],
        help="Traffic stress model for hybrid/traffic evaluation. additive preserves prior results; proportional scales delay by edge distance.",
    )
    parser.add_argument(
        "--traffic_strength",
        type=float,
        default=1.0,
        help="Multiplier for stochastic traffic delay in hybrid/traffic evaluation. 1.0 keeps the PDF-style baseline.",
    )
    parser.add_argument(
        "--traffic_time_scale",
        default="raw",
        choices=["raw", "depot_day"],
        help=(
            "Time coordinate used by PDF/SVRP traffic peaks. raw uses route current_time directly; "
            "depot_day maps depot [ready,due] to [0,1440]. Native matrix evaluation records this "
            "setting but does not use route-time traffic peaks."
        ),
    )
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--allow_unseen_size", action="store_true")
    parser.add_argument("--output_json", default="results/benchmark_metrics.json")
    parser.add_argument("--output_csv", default="results/benchmark_metrics.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_checkpoints(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Checkpoint spec must be name=path, got: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Checkpoint name is empty in spec: {spec}")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        parsed.append((name, path))
    return parsed


def load_policy(path: Path, device: torch.device) -> tuple[TWPointerPolicy, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    context_weight = state.get("context_encoder.0.weight")
    if "context_dim" in checkpoint:
        context_dim = int(checkpoint["context_dim"])
    elif context_weight is not None:
        context_dim = int(context_weight.shape[1])
    else:
        context_dim = 9
    dynamic_weight = state.get("dynamic_encoder.0.weight")
    if "dynamic_feature_dim" in checkpoint:
        dynamic_feature_dim = int(checkpoint["dynamic_feature_dim"])
    else:
        dynamic_feature_dim = int(dynamic_weight.shape[1] if dynamic_weight is not None else 0)
    model = TWPointerPolicy(
        embed_dim=int(checkpoint.get("embed_dim", 128)),
        feature_dim=int(checkpoint.get("feature_dim", 6)),
        context_dim=context_dim,
        dynamic_feature_dim=dynamic_feature_dim,
        model_version=checkpoint.get("model_version", "tw_pointer_legacy_v1"),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def discover_split_sizes(split_root: str | Path) -> list[str]:
    root = Path(split_root) / "train"
    sizes = [path.name for path in root.glob("*") if path.is_dir()]
    return sorted(sizes, key=lambda value: int(value) if value.isdigit() else value)


def resolve_sizes(args: argparse.Namespace) -> list[str]:
    if args.eval_files:
        return []
    if args.sizes:
        return [str(size) for size in args.sizes]
    if args.size:
        return [str(args.size)]
    if args.split_root:
        return discover_split_sizes(args.split_root)
    raise ValueError("Use --eval_files, --size, --sizes, or --split_root to select evaluation data.")


def load_eval_instances(args: argparse.Namespace) -> list:
    if args.eval_files:
        instances = []
        for index, path in enumerate(args.eval_files):
            loaded = load_instances(input_path=path)
            for instance in loaded:
                instance.index = index
            instances.extend(loaded)
        return instances[: args.limit] if args.limit is not None else instances

    selected = []
    for size in resolve_sizes(args):
        if args.split_root:
            split_root = Path(args.split_root)
            split_names = ("train", "val", "test") if args.split == "all" else (args.split,)
            for split_name in split_names:
                selected.extend(
                    load_instances(
                        data_root=split_root / split_name,
                        size=size,
                        source="solomon",
                        pattern=args.pattern,
                        limit=args.limit,
                    )
                )
            continue

        if not args.data_root:
            raise ValueError("Either --split_root, --data_root, or --eval_files is required.")
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
        selected.extend(instances if args.split == "all" else split[args.split])
    return selected


def default_eval_set(args: argparse.Namespace) -> str:
    if args.eval_set:
        return args.eval_set
    if args.eval_files:
        return "explicit_files"
    if args.split_root:
        return f"{Path(args.split_root).name}_{args.split}"
    return f"data_root_{args.split}"


def route_planning_mode(mode: str) -> str:
    if mode in {"static", "hybrid"}:
        return "static"
    if mode == "traffic":
        return "traffic"
    raise ValueError("mode must be static, hybrid, or traffic.")


def solve_routes(model, instance, args: argparse.Namespace, device: torch.device) -> tuple[list[list[int]], float]:
    start = time.perf_counter()
    batch = instances_to_tensors([instance], device=device)
    with torch.no_grad():
        out = rollout(model, batch, decode="greedy")
    order = actions_to_order(out.actions[0])
    matrix = planning_matrix(
        instance,
        mode=route_planning_mode(args.mode),
        traffic_sigma=args.traffic_sigma,
        traffic_buffer=args.traffic_buffer,
        traffic_profile=args.traffic_profile,
        traffic_strength=args.traffic_strength,
    )
    routes = decode_order(
        instance,
        order,
        matrix,
        decoder=args.decoder,
        insert_top_k=args.insert_top_k,
        post_opt=args.post_opt,
    )
    return routes, time.perf_counter() - start


def pdf_distance_matrix(instance) -> np.ndarray:
    """Euclidean distances used by the source PDF/OR-Tools implementation."""
    coords = np.asarray(instance.coords, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def pdf_static_time_matrix(instance) -> np.ndarray:
    """Static travel-time matrix: Euclidean distance rounded to nearest integer."""
    matrix = np.round(pdf_distance_matrix(instance)).astype(float)
    np.maximum(matrix, 0.0, out=matrix)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def normalize_routes(routes: list[list[int]]) -> list[list[int]]:
    normalized = []
    for route in routes:
        cleaned = [int(node) for node in route]
        if not cleaned:
            continue
        if cleaned[0] != 0:
            cleaned = [0] + cleaned
        if cleaned[-1] != 0:
            cleaned.append(0)
        normalized.append(cleaned)
    return normalized


def evaluate_pdf_routes(instance, routes: list[list[int]], travel_time_fn) -> dict:
    """Evaluate routes exactly in the source PDF style.

    Service time is intentionally ignored in time propagation. The reported
    benchmark cost is travel time plus waiting time, and CVR only includes time
    window and capacity violations.
    """
    routes = normalize_routes(routes)
    depot_nodes = {0}
    distance = pdf_distance_matrix(instance)
    all_visited: list[int] = []
    total_distance = 0.0
    total_travel_time = 0.0
    waiting_time = 0.0
    time_window_violations = 0
    capacity_violations = 0
    route_loads = []
    route_durations = []

    for route in routes:
        current_time = 0.0
        route_load = 0.0
        prev = route[0]
        if prev in depot_nodes:
            current_time = 0.0
        for node in route[1:]:
            step_time = float(travel_time_fn(prev, node, current_time))
            total_distance += float(distance[prev, node])
            total_travel_time += step_time
            current_time += step_time
            arrival = current_time

            if node not in depot_nodes:
                all_visited.append(int(node))
                route_load += float(instance.demands[node])
                ready = float(instance.ready_times[node])
                due = float(instance.due_times[node])
                if arrival < ready:
                    wait = ready - arrival
                    waiting_time += wait
                    current_time += wait
                elif arrival > due:
                    time_window_violations += 1
            prev = node

        if route_load > float(instance.vehicle_capacity) + 1e-9:
            capacity_violations += 1
        route_loads.append(route_load)
        route_durations.append(current_time)

    customers = set(range(1, int(instance.num_customers) + 1))
    visited_set = set(all_visited)
    missing_customers = sorted(customers - visited_set)
    duplicate_visits = max(0, len(all_visited) - len(visited_set))
    total_cost = total_travel_time + waiting_time
    benchmark_cvr = 100.0 * (time_window_violations + capacity_violations) / max(1, instance.num_customers)
    benchmark_feasible = (
        not missing_customers
        and duplicate_visits == 0
        and time_window_violations == 0
        and capacity_violations == 0
    )
    route_count = len(routes)
    vehicles_excess = max(0, route_count - int(instance.vehicle_count))
    ignored_service_time = float(
        sum(float(instance.service_times[node]) for node in all_visited)
        if hasattr(instance, "service_times")
        else 0.0
    )

    return {
        "total_distance": total_distance,
        "total_travel_time": total_travel_time,
        "waiting_time": waiting_time,
        "service_time": 0.0,
        "ignored_service_time": ignored_service_time,
        "total_cost": total_cost,
        "late_minutes": 0.0,
        "time_window_violations": time_window_violations,
        "capacity_violations": capacity_violations,
        "missing_customers": missing_customers,
        "duplicate_visits": duplicate_visits,
        "served_customers": all_visited,
        "route_loads": route_loads,
        "route_durations": route_durations,
        "vehicles_excess": vehicles_excess,
        "route_count": route_count,
        "vehicle_count": int(instance.vehicle_count),
        "feasible": bool(benchmark_feasible),
        "cvr": benchmark_cvr,
        "benchmark_total_cost": total_cost,
        "benchmark_cvr": benchmark_cvr,
        "benchmark_feasible": bool(benchmark_feasible),
        "project_total_cost": total_cost,
        "project_cvr": benchmark_cvr,
        "project_feasible": bool(benchmark_feasible),
        "metric_profile": "pdf_compatible",
    }


def _normal_pdf(x: float, mean: float, std_dev: float) -> float:
    return math.exp(-((x - mean) ** 2) / (2 * std_dev**2)) / (std_dev * math.sqrt(2 * math.pi))


def _time_factor(current_time: float) -> float:
    morning_peak = _normal_pdf(current_time, 480.0, 90.0)
    evening_peak = _normal_pdf(current_time, 1020.0, 90.0)
    return 0.5 + 2.0 * (morning_peak + evening_peak)


def _random_factor(current_time: float, rng: np.random.Generator) -> float:
    rush_hour_effect = _normal_pdf(current_time, 480.0, 90.0) + _normal_pdf(current_time, 1020.0, 90.0)
    mu = 0.0 + 0.1 * rush_hour_effect
    sigma = 0.3 + 0.2 * rush_hour_effect
    return rng.lognormal(mean=mu, sigma=sigma)


def _sample_accidents(current_time: float, rng: np.random.Generator) -> float:
    accident_rate = max(0.0, 0.05 * _normal_pdf(current_time, 1260.0, 120.0))
    num_accidents = rng.poisson(lam=accident_rate)
    if num_accidents <= 0:
        return 0.0
    return float(np.sum(rng.uniform(30.0, 120.0, size=num_accidents)))


def pdf_sample_travel_time(
    i: int,
    j: int,
    distance_matrix: np.ndarray,
    current_time: float,
    rng: np.random.Generator,
    traffic_strength: float = 1.0,
    traffic_profile: str = "additive",
) -> float:
    """Source-compatible traffic sample for one edge traversal."""
    if i == j:
        return 0.0
    distance = float(distance_matrix[i, j])
    if distance <= 0:
        return 0.0
    time_fac = _time_factor(current_time)
    distance_factor = 1.0 - math.exp(-distance / 50.0)
    delay_ratio = 0.25 * time_fac * distance_factor
    stochastic_delay = delay_ratio * _random_factor(current_time, rng)
    accident_delay = _sample_accidents(current_time, rng)
    strength = max(0.0, float(traffic_strength))
    if traffic_profile == "additive":
        return distance + strength * (stochastic_delay + accident_delay)
    if traffic_profile == "proportional":
        return distance * (1.0 + strength * stochastic_delay) + strength * accident_delay
    raise ValueError("traffic_profile must be 'additive' or 'proportional'.")


def scale_traffic_time(instance, current_time: float, traffic_time_scale: str) -> float:
    """Map route time into the time coordinate used by SVRP traffic peaks."""
    if traffic_time_scale == "raw":
        return float(current_time)
    if traffic_time_scale == "depot_day":
        start = float(instance.ready_times[0])
        end = float(instance.due_times[0])
        horizon = max(end - start, 1e-6)
        day_fraction = min(max((float(current_time) - start) / horizon, 0.0), 1.0)
        return day_fraction * 1440.0
    raise ValueError("traffic_time_scale must be 'raw' or 'depot_day'.")


def new_traffic_stats(instance) -> dict:
    return {
        "depot_due": float(instance.due_times[0]),
        "traffic_edges": 0,
        "raw_current_time_sum": 0.0,
        "scaled_current_time_sum": 0.0,
        "delay_sum": 0.0,
        "delay_ratio_sum": 0.0,
    }


def record_traffic_stats(
    stats: dict,
    *,
    distance: float,
    travel_time: float,
    raw_current_time: float,
    scaled_current_time: float,
) -> None:
    if distance <= 0:
        return
    delay = max(0.0, float(travel_time) - float(distance))
    stats["traffic_edges"] += 1
    stats["raw_current_time_sum"] += float(raw_current_time)
    stats["scaled_current_time_sum"] += float(scaled_current_time)
    stats["delay_sum"] += delay
    stats["delay_ratio_sum"] += delay / max(float(distance), 1e-6)


def summarize_traffic_stats(stats: dict) -> dict:
    edges = max(1, int(stats.get("traffic_edges", 0)))
    return {
        "depot_due": float(stats.get("depot_due", 0.0)),
        "traffic_edges": int(stats.get("traffic_edges", 0)),
        "raw_current_time_mean": float(stats.get("raw_current_time_sum", 0.0)) / edges,
        "scaled_current_time_mean": float(stats.get("scaled_current_time_sum", 0.0)) / edges,
        "delay_mean": float(stats.get("delay_sum", 0.0)) / edges,
        "delay_ratio_mean": float(stats.get("delay_ratio_sum", 0.0)) / edges,
    }


def native_benchmark_row(row: dict) -> dict:
    """Expose native evaluator metrics through the benchmark report fields."""
    row["benchmark_total_cost"] = float(row.get("total_cost", 0.0))
    row["benchmark_cvr"] = float(row.get("cvr", 0.0))
    row["benchmark_feasible"] = bool(row.get("feasible", False))
    row["project_total_cost"] = float(row.get("total_cost", 0.0))
    row["project_cvr"] = float(row.get("cvr", 0.0))
    row["project_feasible"] = bool(row.get("feasible", False))
    row["metric_profile"] = "native"
    return row


def evaluate_static(instance, routes, *, args: argparse.Namespace, solver_runtime_s: float) -> dict:
    if args.metric_profile == "native":
        matrix = planning_matrix(instance, mode="static")
        row = native_benchmark_row(evaluate_routes(instance, routes, matrix))
        row["solver_runtime_s"] = float(solver_runtime_s)
        return row

    matrix = pdf_static_time_matrix(instance)
    row = evaluate_pdf_routes(instance, routes, lambda i, j, _current_time: float(matrix[i, j]))
    row["solver_runtime_s"] = float(solver_runtime_s)
    return row


def evaluate_traffic_mc(instance, routes, *, model_name: str, args: argparse.Namespace, solver_runtime_s: float) -> dict:
    rows = []
    if args.metric_profile == "native":
        for sample_idx in range(max(1, args.mc_samples)):
            matrix = sample_traffic_matrix(
                instance,
                seed=stable_seed(instance.name, model_name, sample_idx, base_seed=args.traffic_seed),
                traffic_sigma=args.traffic_sigma,
                traffic_profile=args.traffic_profile,
                traffic_strength=args.traffic_strength,
            )
            row = native_benchmark_row(evaluate_routes(instance, routes, matrix))
            row["solver_runtime_s"] = float(solver_runtime_s)
            rows.append(row)
        return aggregate_benchmark_rows(rows, mode=args.mode, static_runtime_s=solver_runtime_s)

    distance = pdf_distance_matrix(instance)
    child_seeds = np.random.SeedSequence(args.traffic_seed).spawn(max(1, args.mc_samples))
    for sample_idx in range(max(1, args.mc_samples)):
        rng = np.random.default_rng(child_seeds[sample_idx])
        traffic_stats = new_traffic_stats(instance)

        def traffic_travel_time(i, j, current_time, rng=rng, stats=traffic_stats):
            raw_current_time = float(current_time)
            scaled_current_time = scale_traffic_time(instance, raw_current_time, args.traffic_time_scale)
            travel_time = pdf_sample_travel_time(
                i,
                j,
                distance,
                scaled_current_time,
                rng,
                traffic_strength=args.traffic_strength,
                traffic_profile=args.traffic_profile,
            )
            record_traffic_stats(
                stats,
                distance=float(distance[i, j]),
                travel_time=float(travel_time),
                raw_current_time=raw_current_time,
                scaled_current_time=scaled_current_time,
            )
            return travel_time

        row = evaluate_pdf_routes(
            instance,
            routes,
            traffic_travel_time,
        )
        row.update(summarize_traffic_stats(traffic_stats))
        row["solver_runtime_s"] = float(solver_runtime_s)
        rows.append(row)
    return aggregate_benchmark_rows(rows, mode=args.mode, static_runtime_s=solver_runtime_s)


def aggregate_benchmark_rows(rows: list[dict], *, mode: str, static_runtime_s: float | None = None) -> dict:
    if not rows:
        return {"instances": 0}
    result = {
        "instances": len(rows),
        "depot_due": mean(rows, "depot_due"),
        "traffic_edges": mean(rows, "traffic_edges"),
        "raw_current_time_mean": mean(rows, "raw_current_time_mean"),
        "scaled_current_time_mean": mean(rows, "scaled_current_time_mean"),
        "delay_mean": mean(rows, "delay_mean"),
        "delay_ratio_mean": mean(rows, "delay_ratio_mean"),
        "total_distance": mean(rows, "total_distance"),
        "total_travel_time": mean(rows, "total_travel_time"),
        "waiting_time": mean(rows, "waiting_time"),
        "service_time": mean(rows, "service_time"),
        "late_minutes": mean(rows, "late_minutes"),
        "time_window_violations": mean(rows, "time_window_violations"),
        "capacity_violations": mean(rows, "capacity_violations"),
        "vehicles_excess": mean(rows, "vehicles_excess"),
        "route_count": mean(rows, "route_count"),
        "project_total_cost": mean(rows, "project_total_cost"),
        "project_cvr": mean(rows, "project_cvr"),
        "project_feasibility": feasibility_mean(rows, "project_feasible", "project_feasibility"),
        "benchmark_total_cost": mean(rows, "benchmark_total_cost"),
        "benchmark_cvr": mean(rows, "benchmark_cvr"),
        "benchmark_feasibility": feasibility_mean(rows, "benchmark_feasible", "benchmark_feasibility", "feasible"),
        "solver_runtime_s": float(static_runtime_s if static_runtime_s is not None else mean(rows, "solver_runtime_s")),
    }
    values = [float(row.get("benchmark_total_cost", 0.0)) for row in rows]
    result["robustness_std"] = 0.0 if mode == "static" or len(values) <= 1 else float(np.std(values, ddof=1))
    result["customers_per_route"] = float(
        np.mean(
            [
                len(row.get("served_customers", [])) / max(1, int(row.get("route_count", 1)))
                for row in rows
            ]
        )
    )
    return result


def aggregate_for_report(rows: list[dict], *, eval_set: str, model_name: str, size: str, mode: str) -> dict:
    base = aggregate_benchmark_rows(rows, mode=mode)
    instances = int(base.get("instances", 0))
    size_int = int(size) if str(size).isdigit() else max(1, int(rows[0].get("num_customers", 1)))
    return {
        "eval_set": eval_set,
        "model_name": model_name,
        "size": str(size),
        "mode": mode,
        "metric_profile": rows[0].get("metric_profile", "pdf_compatible") if rows else "pdf_compatible",
        "decoder": rows[0].get("decoder", "strict_insert") if rows else "strict_insert",
        "post_opt": rows[0].get("post_opt", "none") if rows else "none",
        "traffic_time_scale": rows[0].get("traffic_time_scale", "raw") if rows else "raw",
        "instances": instances,
        "avg_depot_due": float(base.get("depot_due", 0.0)),
        "avg_traffic_edges": float(base.get("traffic_edges", 0.0)),
        "avg_raw_current_time": float(base.get("raw_current_time_mean", 0.0)),
        "avg_scaled_current_time": float(base.get("scaled_current_time_mean", 0.0)),
        "avg_delay": float(base.get("delay_mean", 0.0)),
        "avg_delay_ratio": float(base.get("delay_ratio_mean", 0.0)),
        "avg_cost": float(base.get("benchmark_total_cost", 0.0)),
        "single_customer_cost": float(base.get("benchmark_total_cost", 0.0)) / max(1, size_int),
        "avg_waiting": float(base.get("waiting_time", 0.0)),
        "avg_cvr": float(base.get("benchmark_cvr", 0.0)),
        "feasibility_rate": float(base.get("benchmark_feasibility", 0.0)),
        "avg_solver_runtime_s": float(base.get("solver_runtime_s", 0.0)),
        "robustness_std": float(base.get("robustness_std", 0.0)),
        "avg_route_count": float(base.get("route_count", 0.0)),
        "avg_vehicles_excess": float(base.get("vehicles_excess", 0.0)),
        "project_avg_cost": float(base.get("project_total_cost", 0.0)),
        "project_avg_cvr": float(base.get("project_cvr", 0.0)),
        "project_feasibility_rate": float(base.get("project_feasibility", 0.0)),
    }


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))


def feasibility_mean(rows: list[dict], key: str, *fallback_keys: str) -> float:
    values = []
    for row in rows:
        value = None
        for candidate in (key, *fallback_keys):
            if candidate in row:
                value = row[candidate]
                break
        if value is None:
            values.append(0.0)
        elif isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        else:
            values.append(float(value))
    return float(np.mean(values))


def check_checkpoint_size(model_name: str, checkpoint: dict, instance) -> None:
    size = str(instance.num_customers)
    supported_sizes = checkpoint.get("supported_sizes")
    if supported_sizes and size not in {str(item) for item in supported_sizes}:
        raise ValueError(f"{model_name} supports {supported_sizes}, but {instance.name} has {size} customers.")
    expected = checkpoint.get("num_customers")
    if not supported_sizes and expected is not None and int(expected) != instance.num_customers:
        raise ValueError(f"{model_name} expects {expected} customers, but {instance.name} has {size}.")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    checkpoints = parse_checkpoints(args.checkpoints)
    models = []
    for model_name, checkpoint_path in checkpoints:
        model, checkpoint = load_policy(checkpoint_path, device)
        models.append((model_name, checkpoint_path, model, checkpoint))

    instances = load_eval_instances(args)
    if not instances:
        raise ValueError("No evaluation instances found.")
    eval_set = default_eval_set(args)
    summary = {"aggregate": {}, "per_instance": []}
    report_rows = []

    for model_name, checkpoint_path, model, checkpoint in models:
        rows_by_size: dict[str, list[dict]] = {}
        for instance in instances:
            if not args.allow_unseen_size:
                check_checkpoint_size(model_name, checkpoint, instance)
            size = str(instance.num_customers)
            routes, solve_seconds = solve_routes(model, instance, args, device)
            if args.mode == "static":
                row = evaluate_static(instance, routes, args=args, solver_runtime_s=solve_seconds)
            else:
                row = evaluate_traffic_mc(
                    instance,
                    routes,
                    model_name=model_name,
                    args=args,
                    solver_runtime_s=solve_seconds,
                )
            row.update(
                {
                    "eval_set": eval_set,
                    "model_name": model_name,
                    "checkpoint": str(checkpoint_path),
                    "size": size,
                    "mode": args.mode,
                    "metric_profile": args.metric_profile,
                    "decoder": args.decoder,
                    "post_opt": args.post_opt,
                    "traffic_profile": args.traffic_profile,
                    "traffic_strength": float(args.traffic_strength),
                    "traffic_time_scale": args.traffic_time_scale,
                    "routes": routes,
                    "num_customers": instance.num_customers,
                    "source": instance.source,
                    "instance": instance.name,
                }
            )
            rows_by_size.setdefault(size, []).append(row)
            summary["per_instance"].append(row)

        for size, rows in sorted(rows_by_size.items(), key=lambda item: int(item[0])):
            aggregate_row = aggregate_for_report(
                rows,
                eval_set=eval_set,
                model_name=model_name,
                size=size,
                mode=args.mode,
            )
            aggregate_row["traffic_profile"] = args.traffic_profile
            aggregate_row["traffic_strength"] = float(args.traffic_strength)
            aggregate_row["traffic_time_scale"] = args.traffic_time_scale
            aggregate_row["metric_profile"] = args.metric_profile
            report_rows.append(aggregate_row)
            summary["aggregate"].setdefault(eval_set, {}).setdefault(model_name, {})[size] = aggregate_row

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_csv, report_rows)
    print(f"[DONE] json={args.output_json} csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
