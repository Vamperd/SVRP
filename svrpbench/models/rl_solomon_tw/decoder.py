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
    insert_top_k: int = 30,
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
        return _decode_strict_insert(
            instance,
            order,
            planning_matrix,
            insert_top_k=insert_top_k,
        )
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


def _decode_strict_insert(
    instance,
    order: list[int],
    planning_matrix: np.ndarray,
    *,
    insert_top_k: int = 30,
) -> list[list[int]]:
    """Insert each customer into the best position while respecting the fleet limit.

    This is intentionally incremental: only the changed route is re-scored for
    each candidate. The earlier implementation re-evaluated every route for
    every possible insertion, which made training dominated by Python loops.
    """
    clean_order = [int(node) for node in order if int(node) != 0]
    routes: list[list[int]] = []
    route_scores: list[dict] = []
    max_routes = max(1, int(instance.vehicle_count))

    for node in clean_order:
        candidates: list[tuple[tuple, int | None, int | None, list[int], dict]] = []
        positions = _candidate_positions(routes, node, planning_matrix, insert_top_k)
        for _, route_idx, insert_pos in positions:
            route = routes[route_idx]
            candidate_route = route[:insert_pos] + [node] + route[insert_pos:]
            candidate_score = _score_single_route(instance, candidate_route, planning_matrix)
            old_score = route_scores[route_idx]
            candidates.append(
                (
                    _candidate_key(
                        candidate_score,
                        old_score=old_score,
                        route_count=len(routes),
                    ),
                    route_idx,
                    insert_pos,
                    candidate_route,
                    candidate_score,
                )
            )

        if len(routes) < max_routes:
            candidate_route = [0, node, 0]
            candidate_score = _score_single_route(instance, candidate_route, planning_matrix)
            candidates.append(
                (
                    _candidate_key(
                        candidate_score,
                        old_score=None,
                        route_count=len(routes) + 1,
                    ),
                    None,
                    None,
                    candidate_route,
                    candidate_score,
                )
            )

        if not candidates:
            routes = [[0, node, 0]]
            route_scores = [_score_single_route(instance, routes[0], planning_matrix)]
            continue

        _, route_idx, _, selected_route, selected_score = min(candidates, key=lambda item: item[0])
        if route_idx is None:
            routes.append(selected_route)
            route_scores.append(selected_score)
        else:
            routes[route_idx] = selected_route
            route_scores[route_idx] = selected_score
    return routes


def score_order(
    instance,
    order: list[int],
    planning_matrix: np.ndarray,
    *,
    decoder: str = "strict_insert",
    insert_top_k: int = 30,
    penalties: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """Return REINFORCE reward and route metrics for a decoded order."""
    penalty = dict(DEFAULT_PENALTIES)
    if penalties:
        penalty.update(penalties)
    routes = decode_order(
        instance,
        order,
        planning_matrix,
        decoder=decoder,
        insert_top_k=insert_top_k,
    )
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


def _candidate_positions(
    routes: list[list[int]],
    node: int,
    matrix: np.ndarray,
    insert_top_k: int,
) -> list[tuple[float, int, int]]:
    positions: list[tuple[float, int, int]] = []
    for route_idx, route in enumerate(routes):
        for insert_pos in range(1, len(route)):
            prev_node = route[insert_pos - 1]
            next_node = route[insert_pos]
            delta = float(matrix[prev_node, node] + matrix[node, next_node] - matrix[prev_node, next_node])
            positions.append((delta, route_idx, insert_pos))
    positions.sort(key=lambda item: item[0])
    if insert_top_k and insert_top_k > 0:
        return positions[:insert_top_k]
    return positions


def _candidate_key(
    candidate_score: dict,
    *,
    old_score: dict | None,
    route_count: int,
) -> tuple:
    old_cost = float(old_score["cost"]) if old_score is not None else 0.0
    old_late = float(old_score["late_minutes"]) if old_score is not None else 0.0
    old_violations = int(old_score["violations"]) if old_score is not None else 0
    return (
        int(candidate_score["violations"]),
        max(0, int(candidate_score["violations"]) - old_violations),
        float(candidate_score["late_minutes"]),
        max(0.0, float(candidate_score["late_minutes"]) - old_late),
        int(route_count),
        float(candidate_score["cost"]) - old_cost,
        float(candidate_score["cost"]),
    )


def _score_single_route(instance, route: list[int], matrix: np.ndarray) -> dict:
    route_load = 0.0
    current_time = 0.0
    travel_time = 0.0
    waiting_time = 0.0
    late_minutes = 0.0
    time_window_violations = 0
    for prev, node in zip(route[:-1], route[1:]):
        step_time = float(matrix[prev, node])
        travel_time += step_time
        arrival = current_time + step_time
        if node == 0:
            current_time = arrival
            continue

        route_load += float(instance.demands[node])
        ready = float(instance.ready_times[node])
        due = float(instance.due_times[node])
        wait = max(0.0, ready - arrival)
        waiting_time += wait
        service_start = arrival + wait
        late = max(0.0, service_start - due)
        if late > 1e-6:
            time_window_violations += 1
            late_minutes += late
        current_time = service_start + float(instance.service_times[node])

    capacity_violations = int(route_load > float(instance.vehicle_capacity) + 1e-6)
    violations = capacity_violations + time_window_violations
    return (
        {
            "cost": travel_time + waiting_time + late_minutes,
            "travel_time": travel_time,
            "waiting_time": waiting_time,
            "late_minutes": late_minutes,
            "time_window_violations": time_window_violations,
            "capacity_violations": capacity_violations,
            "violations": violations,
            "route_load": route_load,
            "duration": current_time,
        }
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
