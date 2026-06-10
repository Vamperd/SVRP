#!/usr/bin/env python
"""Solve Solomon TWCVRP instances with a standalone RL checkpoint."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from dataset import load_instances, instances_to_tensors
from decoder import decode_order
from evaluator import evaluate_routes
from model import TWPointerPolicy, actions_to_order, rollout
from traffic import planning_matrix


COLORS = ["#2f80ed", "#27ae60", "#f2994a", "#9b51e0", "#eb5757", "#00a6a6", "#7a5c00"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TWCVRP routes with a standalone RL policy.")
    parser.add_argument("--input", required=True, help="Solomon .txt/.TXT or TWCVRP .npz file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="static", choices=["static", "traffic"])
    parser.add_argument("--decoder", default="strict_insert", choices=["strict_insert", "greedy_split"])
    parser.add_argument("--insert_top_k", type=int, default=30)
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_buffer", type=float, default=0.50)
    parser.add_argument("--output", default="results/solution.json")
    parser.add_argument("--plot_svg", default=None)
    parser.add_argument("--label_nodes", action="store_true")
    parser.add_argument(
        "--allow_unseen_size",
        action="store_true",
        help="Allow solving a customer size not listed in checkpoint supported_sizes.",
    )
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
    return model, checkpoint


def solve_instance(model, instance, args, device: torch.device) -> dict:
    batch = instances_to_tensors([instance], device=device)
    with torch.no_grad():
        out = rollout(model, batch, decode="greedy")
    order = actions_to_order(out.actions[0])
    matrix = planning_matrix(
        instance,
        mode=args.mode,
        traffic_sigma=args.traffic_sigma,
        traffic_buffer=args.traffic_buffer,
    )
    routes = decode_order(
        instance,
        order,
        matrix,
        decoder=args.decoder,
        insert_top_k=args.insert_top_k,
    )
    result = evaluate_routes(instance, routes, matrix)
    result["mode"] = args.mode
    result["order"] = order
    return result


def save_svg(instance, routes: list[list[int]], path: str | Path, *, label_nodes: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(instance.coords, dtype=np.float32)
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    width = 900.0
    height = 900.0
    margin = 40.0

    def project(node: int) -> tuple[float, float]:
        xy = coords[node]
        x = margin + (float(xy[0] - min_xy[0]) / float(span[0])) * (width - 2 * margin)
        y = height - margin - (float(xy[1] - min_xy[1]) / float(span[1])) * (height - 2 * margin)
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="Arial" font-size="18" fill="#222">{html.escape(instance.name)}</text>',
    ]
    for route_idx, route in enumerate(routes):
        color = COLORS[route_idx % len(COLORS)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (project(node) for node in route))
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.0" stroke-opacity="0.85"/>'
        )

    for node in range(1, instance.num_customers + 1):
        x, y = project(node)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2f80ed" fill-opacity="0.75"/>')
        if label_nodes and instance.num_customers <= 200:
            lines.append(
                f'<text x="{x + 4:.1f}" y="{y - 4:.1f}" font-family="Arial" font-size="9" fill="#333">{node}</text>'
            )

    depot_x, depot_y = project(0)
    lines.append(
        f'<rect x="{depot_x - 7:.1f}" y="{depot_y - 7:.1f}" width="14" height="14" fill="#eb5757"/>'
    )
    lines.append(
        f'<text x="{depot_x + 9:.1f}" y="{depot_y - 9:.1f}" font-family="Arial" font-size="12" fill="#111">depot</text>'
    )
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, checkpoint = load_policy(args.checkpoint, device)
    instances = load_instances(input_path=args.input)
    supported_sizes = checkpoint.get("supported_sizes")
    for instance in instances:
        size = str(instance.num_customers)
        if supported_sizes and size not in {str(item) for item in supported_sizes} and not args.allow_unseen_size:
            raise ValueError(
                f"Checkpoint supports sizes {supported_sizes}, but {instance.name} has {size} customers. "
                "Use --allow_unseen_size for an explicit cross-size generalization test."
            )
        expected = checkpoint.get("num_customers")
        if (
            not supported_sizes
            and expected is not None
            and instance.num_customers != int(expected)
            and not args.allow_unseen_size
        ):
            raise ValueError(
                f"Checkpoint expects {expected} customers, but {instance.name} has {instance.num_customers}. "
                "Use --allow_unseen_size for an explicit cross-size generalization test."
            )

    results = [solve_instance(model, instance, args, device) for instance in instances]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.plot_svg and results:
        save_svg(instances[0], results[0]["routes"], args.plot_svg, label_nodes=args.label_nodes)
    feasible = sum(1 for row in results if row["feasible"])
    avg_cvr = sum(float(row["cvr"]) for row in results) / max(1, len(results))
    print(
        "[DONE] "
        f"output={output} instances={len(results)} feasible={feasible}/{len(results)} avg_cvr={avg_cvr:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
