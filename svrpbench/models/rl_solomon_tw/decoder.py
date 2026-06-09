"""Decode policy permutations into multi-vehicle TWCVRP routes."""
from __future__ import annotations

import numpy as np

from evaluator import evaluate_routes


DEFAULT_PENALTIES = {
    "late": 10.0,
    "time_window": 1000.0,
    "capacity": 1000.0,
    "missing": 500.0,
    "duplicate": 500.0,
    "vehicle": 10000.0,
    "route_count": 200.0,
    "feasible_bonus": 50000.0,
    "infeasible": 50000.0,
}


def decode_order(
    instance,
    order: list[int],
    planning_matrix: np.ndarray,
    *,
    decoder: str = "strict_insert",
    split_on_late: bool = True,
) -> list[list[int]]:
    """Split a customer order into vehicle routes using capacity and time windows."""
    if decoder == "greedy_split":
        return _decode_greedy_split(
            instance,
            order,
            planning_matrix,
            split_on_late=split_on_late,
        )
    if decoder == "strict_insert":
        return _decode_strict_insert(instance, order, planning_matrix)
    raise ValueError("decoder must be 'strict_insert' or 'greedy_split'.")


def _decode_greedy_split(
    instance,
    order: list[int],
    planning_matrix: np.ndarray,
    *,
    split_on_late: bool = True,
) -> list[list[int]]:
    """Legacy decoder: append customers and open a new route when late/capacity would fail."""
    clean_order = [int(node) for node in order if int(node) != 0]
    routes: list[list[int]] = []
    current_route = [0]
    current_node = 0
    current_time = 0.0
    current_load = 0.0

    for node in clean_order:
        demand = float(instance.demands[node])
        projected = _project_append(instance, planning_matrix, current_node, current_time, node)
        would_exceed_capacity = current_load + demand > float(instance.vehicle_capacity) + 1e-6
        would_be_late = projected["late"] > 1e-6
        should_split = len(current_route) > 1 and (
            would_exceed_capacity or (split_on_late and would_be_late)
        )

        if should_split:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0]
            current_node = 0
            current_time = 0.0
            current_load = 0.0
            projected = _project_append(instance, planning_matrix, current_node, current_time, node)

        current_route.append(node)
        current_node = node
        current_time = projected["depart"]
        current_load += demand

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)
    return routes


def _decode_strict_insert(instance, order: list[int], planning_matrix: np.ndarray) -> list[list[int]]:
    """Insert each customer into the best position while respecting the fleet limit."""
    clean_order = [int(node) for node in order if int(node) != 0]
    routes: list[list[int]] = []
    max_routes = max(1, int(instance.vehicle_count))

    for node in clean_order:
        candidates: list[list[list[int]]] = []
        for route_idx, route in enumerate(routes):
            for insert_pos in range(1, len(route)):
                candidate = [list(row) for row in routes]
                candidate[route_idx] = route[:insert_pos] + [node] + route[insert_pos:]
                candidates.append(candidate)

        if len(routes) < max_routes:
            candidates.append([list(row) for row in routes] + [[0, node, 0]])

        if not candidates:
            routes = [[0, node, 0]]
            continue

        routes = min(
            candidates,
            key=lambda candidate: _partial_route_score(instance, candidate, planning_matrix),
        )
    return routes


def score_order(
    instance,
    order: list[int],
    planning_matrix: np.ndarray,
    *,
    decoder: str = "strict_insert",
    penalties: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """Return REINFORCE reward and route metrics for a decoded order."""
    penalty = dict(DEFAULT_PENALTIES)
    if penalties:
        penalty.update(penalties)
    routes = decode_order(instance, order, planning_matrix, decoder=decoder)
    metrics = evaluate_routes(instance, routes, planning_matrix)
    reward = -float(metrics["total_cost"])
    reward += penalty["feasible_bonus"] if metrics["feasible"] else -penalty["infeasible"]
    reward -= penalty["late"] * float(metrics["late_minutes"])
    reward -= penalty["time_window"] * float(metrics["time_window_violations"])
    reward -= penalty["capacity"] * float(metrics["capacity_violations"])
    reward -= penalty["missing"] * len(metrics["missing_customers"])
    reward -= penalty["duplicate"] * float(metrics["duplicate_visits"])
    reward -= penalty["vehicle"] * float(metrics["vehicles_excess"])
    reward -= penalty["route_count"] * float(metrics["route_count"])
    return reward, metrics


def _partial_route_score(instance, routes: list[list[int]], planning_matrix: np.ndarray) -> tuple:
    metrics = evaluate_routes(instance, routes, planning_matrix)
    partial_violations = (
        10000 * int(metrics["vehicles_excess"])
        + 1000 * int(metrics["capacity_violations"])
        + 1000 * int(metrics["time_window_violations"])
        + 1000 * int(metrics["duplicate_visits"])
    )
    return (
        partial_violations,
        float(metrics["late_minutes"]),
        int(metrics["route_count"]),
        float(metrics["total_cost"]),
    )


def _project_append(instance, matrix: np.ndarray, current_node: int, current_time: float, node: int) -> dict:
    arrival = float(current_time) + float(matrix[current_node, node])
    ready = float(instance.ready_times[node])
    due = float(instance.due_times[node])
    wait = max(0.0, ready - arrival)
    service_start = arrival + wait
    service = float(instance.service_times[node])
    return {
        "arrival": arrival,
        "wait": wait,
        "late": max(0.0, service_start - due),
        "depart": service_start + service,
    }
