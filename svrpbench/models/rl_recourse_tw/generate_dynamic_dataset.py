#!/usr/bin/env python
"""Generate dynamic dataset snapshots from static benchmark routes.

Applies time-window drift and traffic perturbation to pre-generated static
routes, producing a dataset of (traffic_matrix, drifted_due_times, expert_order)
snapshots suitable for Static-Guided Behavioral Cloning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import SOLOMON_TW_DIR  # noqa: F401
from dataset import load_instances
from traffic import sample_traffic_matrix, stable_seed, euclidean_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dynamic dataset from static routes.")
    parser.add_argument("--routes_dir", required=True, help="Directory with *_routes.json files from generate_static_routes.py.")
    parser.add_argument("--split_root", required=True, help="Data split root (for loading original instances).")
    parser.add_argument("--sizes", nargs="+", required=True, help="Sizes to process.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--snapshots_per_route", type=int, default=10, help="Traffic realizations per static route (M).")
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_profile", default="additive", choices=["additive", "proportional"])
    parser.add_argument("--traffic_strength", type=float, default=1.0)
    parser.add_argument("--drift_sigma", type=float, default=0.15, help="σ_shift for time-window drift.")
    parser.add_argument("--no_drift", action="store_true", help="Disable time-window drift (ablation).")
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data_cache/dynamic_dataset")
    return parser


def apply_time_window_drift(
    instance,
    drift_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    ready = np.asarray(instance.ready_times, dtype=np.float32)
    due = np.asarray(instance.due_times, dtype=np.float32)
    width = np.maximum(due - ready, 1e-6)
    drift = rng.normal(0.0, drift_sigma * width)
    drifted = np.maximum(due + drift, ready)
    drifted[0] = due[0]
    return drifted


def routes_to_order(routes: list[list[int]]) -> list[int]:
    order: list[int] = []
    for route in routes:
        for node in route:
            if node != 0 and node not in order:
                order.append(int(node))
    return order


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.traffic_seed)

    for size in args.sizes:
        size_dir = Path(args.split_root) / args.split / str(size)
        routes_subdir = Path(args.routes_dir) / str(size)
        if not routes_subdir.exists():
            print(f"[SKIP] size={size} routes dir not found: {routes_subdir}")
            continue

        instances = load_instances(data_root=size_dir, size=size, source=args.source)
        instance_map = {inst.name: inst for inst in instances}

        for inst_name, instance in instance_map.items():
            routes_file = routes_subdir / f"{inst_name}_routes.json"
            if not routes_file.exists():
                print(f"  [MISSING] {inst_name}")
                continue

            routes_data = json.loads(routes_file.read_text(encoding="utf-8"))
            static_routes = routes_data["routes"]
            if not static_routes:
                continue

            snapshots: list[dict] = []
            for route_idx, routes in enumerate(static_routes):
                expert_order = routes_to_order(routes)
                if not expert_order:
                    continue

                for snap in range(args.snapshots_per_route):
                    seed = stable_seed(inst_name, "sg_dynamic", route_idx, snap, base_seed=args.traffic_seed)
                    traf_matrix = sample_traffic_matrix(
                        instance,
                        seed=seed,
                        traffic_sigma=args.traffic_sigma,
                        traffic_profile=args.traffic_profile,
                        traffic_strength=args.traffic_strength,
                    )
                    if args.no_drift:
                        drifted_due = np.asarray(instance.due_times, dtype=np.float32)
                    else:
                        sample_rng = np.random.default_rng(int(seed))
                        drifted_due = apply_time_window_drift(instance, args.drift_sigma, sample_rng)

                    snapshots.append({
                        "traffic_matrix": traf_matrix,
                        "drifted_due_times": drifted_due,
                        "expert_order": expert_order,
                    })

            if not snapshots:
                continue

            out_dir = Path(args.output_dir) / str(size)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{inst_name}_snapshots.npz"

            stacked_matrices = np.stack([s["traffic_matrix"] for s in snapshots])
            stacked_drifts = np.stack([s["drifted_due_times"] for s in snapshots])

            max_order_len = max(len(s["expert_order"]) for s in snapshots)
            padded_orders = np.full((len(snapshots), max_order_len), -1, dtype=np.int32)
            for i, s in enumerate(snapshots):
                padded_orders[i, : len(s["expert_order"])] = s["expert_order"]

            np.savez_compressed(
                out_path,
                traffic_matrices=stacked_matrices,
                drifted_due_times=stacked_drifts,
                expert_orders=padded_orders,
                num_snapshots=len(snapshots),
                instance_name=inst_name,
                original_due_times=np.asarray(instance.due_times, dtype=np.float32),
                original_ready_times=np.asarray(instance.ready_times, dtype=np.float32),
            )
            print(f"  {inst_name}: {len(snapshots)} snapshots")

    print(f"[DONE] output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
