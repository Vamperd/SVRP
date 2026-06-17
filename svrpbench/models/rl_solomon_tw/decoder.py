"""Decode policy permutations into multi-vehicle TWCVRP routes."""
from __future__ import annotations

import numpy as np

from evaluator import evaluate_routes


DEFAULT_PENALTIES = {
    "objective": "feasibility",
    "late": 10.0,
    "time_window": 1000.0,
    "capacity": 1000.0,
    "missing": 500.0,
    "duplicate": 500.0,
    "vehicle": 10000.0,
    "route_count": 200.0,
    "route_overuse": 0.0,
    "target_customers_per_route": 9.0,
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
    post_opt: str = "none",
) -> list[list[int]]:
    """Split a customer order into vehicle routes using capacity and time windows."""
    if decoder == "greedy_split":
        routes = _decode_greedy_split(
            instance,
            order,
            planning_matrix,
            split_on_late=split_on_late,
        )
    elif decoder == "strict_insert":
        routes = _decode_strict_insert(
            instance,
            order,
            planning_matrix,
            insert_top_k=insert_top_k,
        )
    elif decoder == "deadline_aware_insert":
        routes = _decode_deadline_aware_insert(
            instance,
            order,
            planning_matrix,
            insert_top_k=insert_top_k,
        )
    else:
        raise ValueError("decoder must be 'strict_insert', 'deadline_aware_insert', or 'greedy_split'.")
    return apply_post_opt(instance, routes, planning_matrix, post_opt=post_opt)


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


def _decode_deadline_aware_insert(
    instance,
    order: list[int],
    planning_matrix: np.ndarray,
    *,
    insert_top_k: int = 30,
) -> list[list[int]]:
    """Insert customers with a stronger preference for time-window slack.

    Unlike ``strict_insert``, this decoder scores candidate insertions with the
    full route timing state before applying ``insert_top_k``. It is slower, but
    avoids pruning a time-window-friendly insertion only because its travel
    delta is not among the cheapest.
    """
    clean_order = [int(node) for node in order if int(node) != 0]
    routes: list[list[int]] = []
    route_scores: list[dict] = []
    max_routes = max(1, int(instance.vehicle_count))

    for node in clean_order:
        candidates: list[tuple[tuple, int | None, int | None, list[int], dict]] = []
        for route_idx, route in enumerate(routes):
            old_score = route_scores[route_idx]
            for insert_pos in range(1, len(route)):
                candidate_route = route[:insert_pos] + [node] + route[insert_pos:]
                candidate_score = _score_single_route(instance, candidate_route, planning_matrix)
                candidates.append(
                    (
                        _deadline_candidate_key(
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
                    _deadline_candidate_key(
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

        if insert_top_k and insert_top_k > 0:
            candidates = sorted(candidates, key=lambda item: item[0])[:insert_top_k]
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
    post_opt: str = "none",
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
        post_opt=post_opt,
    )
    metrics = evaluate_routes(instance, routes, planning_matrix)
    annotate_objective_metrics(instance, metrics, penalty)
    if penalty.get("objective") == "robust_cvr":
        reward = _robust_cvr_reward(instance, metrics, penalty)
    else:
        reward = _feasibility_reward(metrics, penalty)
    return reward, metrics


def annotate_objective_metrics(instance, metrics: dict, penalties: dict[str, float] | None = None) -> dict:
    penalty = dict(DEFAULT_PENALTIES)
    if penalties:
        penalty.update(penalties)
    num_customers = max(1, int(instance.num_customers))
    target_customers_per_route = float(penalty.get("target_customers_per_route", 0.0) or 0.0)
    if target_customers_per_route > 0:
        target_route_count = int(np.ceil(num_customers / target_customers_per_route))
    else:
        target_route_count = 0
    route_count = int(metrics.get("route_count", 0))
    route_overuse = max(0, route_count - target_route_count) if target_route_count > 0 else 0
    metrics["target_route_count"] = target_route_count
    metrics["route_overuse"] = route_overuse
    metrics["cost_per_customer"] = float(metrics.get("total_cost", 0.0)) / num_customers
    metrics["late_per_customer"] = float(metrics.get("late_minutes", 0.0)) / num_customers
    metrics["time_window_violations_per_customer"] = (
        float(metrics.get("time_window_violations", 0.0)) / num_customers
    )
    metrics["capacity_violations_per_customer"] = (
        float(metrics.get("capacity_violations", 0.0)) / num_customers
    )
    return metrics


def _feasibility_reward(metrics: dict, penalty: dict) -> float:
    reward = -float(metrics["total_cost"])
    reward += penalty["feasible_bonus"] if metrics["feasible"] else -penalty["infeasible"]
    reward -= penalty["late"] * float(metrics["late_minutes"])
    reward -= penalty["time_window"] * float(metrics["time_window_violations"])
    reward -= penalty["capacity"] * float(metrics["capacity_violations"])
    reward -= penalty["missing"] * len(metrics["missing_customers"])
    reward -= penalty["duplicate"] * float(metrics["duplicate_visits"])
    reward -= penalty["vehicle"] * float(metrics["vehicles_excess"])
    reward -= penalty["route_count"] * float(metrics["route_count"])
    return reward


def _robust_cvr_reward(instance, metrics: dict, penalty: dict) -> float:
    num_customers = max(1, int(instance.num_customers))
    reward = -float(metrics["cost_per_customer"])
    if metrics["feasible"]:
        reward += float(penalty["feasible_bonus"]) / num_customers
    else:
        reward -= float(penalty["infeasible"]) / num_customers
    reward -= float(penalty["late"]) * float(metrics["late_per_customer"])
    reward -= (
        float(penalty["time_window"])
        * float(metrics.get("time_window_violations", 0.0))
        / num_customers
    )
    reward -= (
        float(penalty["capacity"])
        * float(metrics.get("capacity_violations", 0.0))
        / num_customers
    )
    reward -= float(penalty["missing"]) * len(metrics["missing_customers"]) / num_customers
    reward -= float(penalty["duplicate"]) * float(metrics["duplicate_visits"]) / num_customers
    reward -= float(penalty["vehicle"]) * float(metrics["vehicles_excess"]) / num_customers
    reward -= float(penalty["route_count"]) * float(metrics["route_count"]) / num_customers
    reward -= float(penalty.get("route_overuse", 0.0)) * float(metrics["route_overuse"]) / num_customers
    return reward


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


def _deadline_candidate_key(
    candidate_score: dict,
    *,
    old_score: dict | None,
    route_count: int,
) -> tuple:
    old_cost = float(old_score["cost"]) if old_score is not None else 0.0
    old_late = float(old_score["late_minutes"]) if old_score is not None else 0.0
    old_wait = float(old_score["waiting_time"]) if old_score is not None else 0.0
    old_violations = int(old_score["violations"]) if old_score is not None else 0
    min_slack = float(candidate_score.get("min_slack", 0.0))
    return (
        int(candidate_score["violations"]),
        max(0, int(candidate_score["violations"]) - old_violations),
        float(candidate_score["late_minutes"]),
        max(0.0, float(candidate_score["late_minutes"]) - old_late),
        int(route_count),
        -min_slack,
        max(0.0, float(candidate_score["waiting_time"]) - old_wait),
        float(candidate_score["cost"]) - old_cost,
        float(candidate_score["cost"]),
    )


def apply_post_opt(
    instance,
    routes: list[list[int]],
    planning_matrix: np.ndarray,
    *,
    post_opt: str = "none",
) -> list[list[int]]:
    """Apply optional route repair after decoding."""
    normalized = _normalize_routes(routes)
    if post_opt in {"none", "", None}:
        return normalized
    if post_opt == "time_window_repair":
        return _time_window_repair(instance, normalized, planning_matrix)
    raise ValueError("post_opt must be 'none' or 'time_window_repair'.")


def _time_window_repair(
    instance,
    routes: list[list[int]],
    planning_matrix: np.ndarray,
    *,
    max_passes: int = 4,
) -> list[list[int]]:
    """Local repair focused on time-window violations.

    The move set is deliberately small and deterministic: relocate late
    customers and their immediate neighbors, then swap those focused customers
    with other customers. This keeps the repair useful for narrow time-window
    instances without turning evaluation into an unbounded search.
    """
    best_routes = _normalize_routes(routes)
    best_metrics = evaluate_routes(instance, best_routes, planning_matrix)
    best_key = _repair_key(best_metrics)
    if best_metrics.get("time_window_violations", 0) <= 0:
        return best_routes

    for _ in range(max_passes):
        improved = False
        focus_nodes = _repair_focus_nodes(instance, best_routes, planning_matrix)
        if not focus_nodes:
            break

        candidate = _best_relocate_move(
            instance,
            best_routes,
            planning_matrix,
            focus_nodes=focus_nodes,
            current_key=best_key,
        )
        if candidate is not None:
            best_routes, best_metrics, best_key = candidate
            improved = True

        candidate = _best_swap_move(
            instance,
            best_routes,
            planning_matrix,
            focus_nodes=focus_nodes,
            current_key=best_key,
        )
        if candidate is not None:
            best_routes, best_metrics, best_key = candidate
            improved = True

        if not improved or best_metrics.get("time_window_violations", 0) <= 0:
            break
    return best_routes


def _best_relocate_move(
    instance,
    routes: list[list[int]],
    planning_matrix: np.ndarray,
    *,
    focus_nodes: set[int],
    current_key: tuple,
) -> tuple[list[list[int]], dict, tuple] | None:
    best_routes: list[list[int]] | None = None
    best_metrics: dict | None = None
    best_key = current_key
    max_routes = max(1, int(instance.vehicle_count))

    for route_idx, pos, node in _customer_positions(routes):
        if node not in focus_nodes:
            continue
        base_routes = _remove_position(routes, route_idx, pos)
        for dest_idx, dest_route in enumerate(base_routes):
            for insert_pos in range(1, len(dest_route)):
                candidate_routes = _copy_routes(base_routes)
                candidate_routes[dest_idx] = (
                    candidate_routes[dest_idx][:insert_pos]
                    + [node]
                    + candidate_routes[dest_idx][insert_pos:]
                )
                candidate_routes = _normalize_routes(candidate_routes)
                metrics = evaluate_routes(instance, candidate_routes, planning_matrix)
                key = _repair_key(metrics)
                if key < best_key:
                    best_routes, best_metrics, best_key = candidate_routes, metrics, key

        if len(base_routes) < max_routes:
            candidate_routes = _normalize_routes(base_routes + [[0, node, 0]])
            metrics = evaluate_routes(instance, candidate_routes, planning_matrix)
            key = _repair_key(metrics)
            if key < best_key:
                best_routes, best_metrics, best_key = candidate_routes, metrics, key

    if best_routes is None or best_metrics is None:
        return None
    return best_routes, best_metrics, best_key


def _best_swap_move(
    instance,
    routes: list[list[int]],
    planning_matrix: np.ndarray,
    *,
    focus_nodes: set[int],
    current_key: tuple,
) -> tuple[list[list[int]], dict, tuple] | None:
    positions = _customer_positions(routes)
    best_routes: list[list[int]] | None = None
    best_metrics: dict | None = None
    best_key = current_key

    for idx_a, (route_a, pos_a, node_a) in enumerate(positions):
        for route_b, pos_b, node_b in positions[idx_a + 1 :]:
            if node_a not in focus_nodes and node_b not in focus_nodes:
                continue
            candidate_routes = _copy_routes(routes)
            candidate_routes[route_a][pos_a] = node_b
            candidate_routes[route_b][pos_b] = node_a
            candidate_routes = _normalize_routes(candidate_routes)
            metrics = evaluate_routes(instance, candidate_routes, planning_matrix)
            key = _repair_key(metrics)
            if key < best_key:
                best_routes, best_metrics, best_key = candidate_routes, metrics, key

    if best_routes is None or best_metrics is None:
        return None
    return best_routes, best_metrics, best_key


def _repair_key(metrics: dict) -> tuple:
    return (
        len(metrics.get("missing_customers", [])),
        int(metrics.get("duplicate_visits", 0)),
        int(metrics.get("vehicles_excess", 0)),
        int(metrics.get("capacity_violations", 0)),
        int(metrics.get("time_window_violations", 0)),
        float(metrics.get("late_minutes", 0.0)),
        float(metrics.get("total_cost", 0.0)),
        int(metrics.get("route_count", 0)),
    )


def _repair_focus_nodes(instance, routes: list[list[int]], matrix: np.ndarray) -> set[int]:
    focus: set[int] = set()
    for route in routes:
        late_positions = _late_positions(instance, route, matrix)
        for pos in late_positions:
            for neighbor_pos in (pos - 1, pos, pos + 1):
                if 0 < neighbor_pos < len(route) - 1:
                    focus.add(int(route[neighbor_pos]))
    return focus


def _late_positions(instance, route: list[int], matrix: np.ndarray) -> list[int]:
    current_time = 0.0
    late_positions: list[int] = []
    for pos, (prev, node) in enumerate(zip(route[:-1], route[1:]), start=1):
        arrival = current_time + float(matrix[prev, node])
        if node == 0:
            current_time = arrival
            continue
        wait = max(0.0, float(instance.ready_times[node]) - arrival)
        service_start = arrival + wait
        if service_start - float(instance.due_times[node]) > 1e-6:
            late_positions.append(pos)
        current_time = service_start + float(instance.service_times[node])
    return late_positions


def _customer_positions(routes: list[list[int]]) -> list[tuple[int, int, int]]:
    positions: list[tuple[int, int, int]] = []
    for route_idx, route in enumerate(routes):
        for pos in range(1, len(route) - 1):
            node = int(route[pos])
            if node != 0:
                positions.append((route_idx, pos, node))
    return positions


def _remove_position(routes: list[list[int]], route_idx: int, pos: int) -> list[list[int]]:
    candidate = _copy_routes(routes)
    del candidate[route_idx][pos]
    return _normalize_routes(candidate)


def _copy_routes(routes: list[list[int]]) -> list[list[int]]:
    return [list(route) for route in routes]


def _normalize_routes(routes: list[list[int]]) -> list[list[int]]:
    normalized: list[list[int]] = []
    for route in routes:
        customers = [int(node) for node in route if int(node) != 0]
        if not customers:
            continue
        normalized.append([0] + customers + [0])
    return normalized


def _score_single_route(instance, route: list[int], matrix: np.ndarray) -> dict:
    route_load = 0.0
    current_time = 0.0
    travel_time = 0.0
    waiting_time = 0.0
    late_minutes = 0.0
    time_window_violations = 0
    min_slack = float("inf")
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
        min_slack = min(min_slack, due - service_start)
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
            "min_slack": min_slack if min_slack != float("inf") else 0.0,
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
