"""Simple time-window-aware nearest-neighbor baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import load_instances
from decoder import decode_order
from evaluator import evaluate_routes
from traffic import planning_matrix


def nearest_order(instance, matrix) -> list[int]:
    unvisited = set(range(1, instance.num_customers + 1))
    order: list[int] = []
    current = 0
    current_time = 0.0
    remaining_capacity = float(instance.vehicle_capacity)

    while unvisited:
        best_node = None
        best_score = float("inf")
        for node in unvisited:
            demand = float(instance.demands[node])
            route_time = 0.0 if demand > remaining_capacity else current_time
            route_current = 0 if demand > remaining_capacity else current
            arrival = route_time + float(matrix[route_current, node])
            wait = max(0.0, float(instance.ready_times[node]) - arrival)
            start = arrival + wait
            late = max(0.0, start - float(instance.due_times[node]))
            score = late * 1000.0 + wait * 0.1 + float(matrix[route_current, node])
            if score < best_score:
                best_score = score
                best_node = node

        node = int(best_node)
        demand = float(instance.demands[node])
        if demand > remaining_capacity:
            current = 0
            current_time = 0.0
            remaining_capacity = float(instance.vehicle_capacity)
        arrival = current_time + float(matrix[current, node])
        wait = max(0.0, float(instance.ready_times[node]) - arrival)
        current_time = arrival + wait + float(instance.service_times[node])
        remaining_capacity -= demand
        current = node
        order.append(node)
        unvisited.remove(node)
    return order


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple TWCVRP heuristic on one input.")
    parser.add_argument("--input", required=True, help="Solomon .txt/.TXT or TWCVRP .npz file.")
    parser.add_argument("--mode", default="static", choices=["static", "traffic"])
    parser.add_argument("--decoder", default="strict_insert", choices=["strict_insert", "greedy_split"])
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_buffer", type=float, default=0.50)
    parser.add_argument("--output", default="results/heuristic_solution.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instances = load_instances(input_path=args.input)
    results = []
    for instance in instances:
        matrix = planning_matrix(
            instance,
            mode=args.mode,
            traffic_sigma=args.traffic_sigma,
            traffic_buffer=args.traffic_buffer,
        )
        order = nearest_order(instance, matrix)
        routes = decode_order(instance, order, matrix, decoder=args.decoder)
        results.append(evaluate_routes(instance, routes, matrix))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] output={output} instances={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
