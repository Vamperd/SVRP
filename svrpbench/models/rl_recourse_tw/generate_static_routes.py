#!/usr/bin/env python
"""Generate S diverse benchmark routes per instance from a static checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import ensure_dir, SOLOMON_TW_DIR  # noqa: F401
from dataset import load_instances, instances_to_tensors
from decoder import decode_order
from evaluator import evaluate_routes
from model import TWPointerPolicy, actions_to_order, rollout
from traffic import planning_matrix, sample_traffic_matrix, stable_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate diverse static routes for dynamic dataset.")
    parser.add_argument("--split_root", required=True, help="Root of the data split (e.g. data_splits/universal_v1).")
    parser.add_argument("--sizes", nargs="+", required=True, help="Sizes to generate routes for.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True, help="Static RL checkpoint.")
    parser.add_argument("--num_routes", type=int, default=20, help="Number of diverse routes per instance.")
    parser.add_argument("--output_dir", default="data_cache/static_routes", help="Output directory for route files.")
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_policy(path: str | Path, device: torch.device) -> tuple[TWPointerPolicy, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    context_dim = int(
        checkpoint.get("context_dim", state.get("context_encoder.0.weight").shape[1])
    )
    dynamic_weight = state.get("dynamic_encoder.0.weight")
    dynamic_feature_dim = int(
        checkpoint.get(
            "dynamic_feature_dim",
            dynamic_weight.shape[1] if dynamic_weight is not None else 0,
        )
    )
    model = TWPointerPolicy(
        embed_dim=int(checkpoint.get("embed_dim", 128)),
        feature_dim=int(checkpoint.get("feature_dim", 6)),
        context_dim=context_dim,
        dynamic_feature_dim=dynamic_feature_dim,
        model_version=checkpoint.get("model_version", "tw_pointer_dynamic_v2"),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def generate_routes_for_instance(
    model,
    instance,
    device: torch.device,
    num_routes: int,
    rng: np.random.Generator,
) -> list[list[list[int]]]:
    batch = instances_to_tensors([instance], device=device)
    routes_set: set[tuple[tuple[int, ...], ...]] = set()
    all_routes: list[list[list[int]]] = []
    top_k_values = [10, 20, 30, 0]
    max_attempts = num_routes * 3

    for attempt in range(max_attempts):
        if len(all_routes) >= num_routes:
            break
        insert_top_k = int(top_k_values[attempt % len(top_k_values)])
        decode_mode = "greedy" if attempt < num_routes else "sample"
        if decode_mode == "sample":
            insert_top_k = int(rng.choice(top_k_values))

        with torch.no_grad():
            out = rollout(model, batch, decode=decode_mode)
        order = actions_to_order(out.actions[0])
        matrix = planning_matrix(instance, mode="static")
        routes = decode_order(
            instance, order, matrix, decoder="strict_insert", insert_top_k=max(1, insert_top_k)
        )
        normalized = normalize_routes_for_dedup(routes)
        route_hash = tuple(tuple(r) for r in normalized)
        if route_hash not in routes_set:
            routes_set.add(route_hash)
            all_routes.append(routes)

    return all_routes


def normalize_routes_for_dedup(routes: list[list[int]]) -> list[list[int]]:
    cleaned = []
    for route in routes:
        r = [int(n) for n in route]
        if len(r) <= 2:
            continue
        if r[0] == 0:
            r = r[1:]
        if r[-1] == 0:
            r = r[:-1]
        if r:
            cleaned.append(r)
    cleaned.sort(key=lambda r: (len(r), r[0] if r else 0))
    return cleaned


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    model, checkpoint = load_policy(args.checkpoint, device)
    print(f"[LOAD] checkpoint={args.checkpoint} supported_sizes={checkpoint.get('supported_sizes', [])}")

    for size in args.sizes:
        size_dir = Path(args.split_root) / args.split / size
        if not size_dir.exists():
            print(f"[SKIP] size={size} split_dir not found: {size_dir}")
            continue
        instances = load_instances(data_root=size_dir, size=size, source=args.source)
        print(f"[SIZE={size}] instances={len(instances)}")

        for instance in instances:
            routes = generate_routes_for_instance(
                model, instance, device, args.num_routes, rng
            )
            out_dir = Path(args.output_dir) / size
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{instance.name}_routes.json"
            out_path.write_text(
                json.dumps(
                    {
                        "instance_name": instance.name,
                        "num_customers": instance.num_customers,
                        "routes": routes,
                        "count": len(routes),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  {instance.name}: {len(routes)} routes")

    print(f"[DONE] output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
