"""Route evaluation for TWCVRP static and traffic-aware experiments."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from traffic import euclidean_matrix


def normalize_routes(routes: Iterable[Iterable[int]]) -> list[list[int]]:
    normalized: list[list[int]] = []
    for route in routes:
        row = [int(node) for node in route]
        if not row:
            continue
        if row[0] != 0:
            row.insert(0, 0)
        if row[-1] != 0:
            row.append(0)
        if len(row) > 1:
            normalized.append(row)
    return normalized


def evaluate_routes(instance, routes: list[list[int]], travel_time_matrix: np.ndarray) -> dict:
    """Evaluate capacity, time windows, travel, waiting, and service metrics."""
    start = time.perf_counter()
    routes = normalize_routes(routes)
    travel = np.asarray(travel_time_matrix, dtype=np.float32)
    distance_matrix = euclidean_matrix(instance.coords)

    total_distance = 0.0
    total_travel_time = 0.0
    waiting_time = 0.0
    service_time = 0.0
    late_minutes = 0.0
    time_window_violations = 0
    capacity_violations = 0
    route_loads: list[float] = []
    route_durations: list[float] = []
    served: list[int] = []

    for route in routes:
        current_time = 0.0
        route_load = 0.0
        for prev, node in zip(route[:-1], route[1:]):
            total_distance += float(distance_matrix[prev, node])
            step_time = float(travel[prev, node])
            total_travel_time += step_time
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
            service = float(instance.service_times[node])
            service_time += service
            current_time = service_start + service
            served.append(int(node))

        if route_load > float(instance.vehicle_capacity) + 1e-6:
            capacity_violations += 1
        route_loads.append(route_load)
        route_durations.append(current_time)

    unique_served = sorted(set(served))
    expected = set(range(1, instance.num_customers + 1))
    missing = sorted(expected - set(unique_served))
    duplicate_visits = len(served) - len(unique_served)
    vehicles_excess = max(0, len(routes) - int(instance.vehicle_count))
    violation_events = (
        time_window_violations
        + capacity_violations
        + len(missing)
        + duplicate_visits
        + vehicles_excess
    )
    feasible = violation_events == 0

    return {
        "instance": instance.name,
        "source": instance.source,
        "routes": routes,
        "total_distance": total_distance,
        "total_travel_time": total_travel_time,
        "waiting_time": waiting_time,
        "service_time": service_time,
        "total_cost": total_travel_time + waiting_time + late_minutes,
        "late_minutes": late_minutes,
        "time_window_violations": time_window_violations,
        "capacity_violations": capacity_violations,
        "vehicles_excess": vehicles_excess,
        "route_count": len(routes),
        "route_loads": route_loads,
        "route_durations": route_durations,
        "served_customers": unique_served,
        "missing_customers": missing,
        "duplicate_visits": duplicate_visits,
        "vehicle_capacity": float(instance.vehicle_capacity),
        "vehicle_count": int(instance.vehicle_count),
        "feasible": bool(feasible),
        "cvr": 100.0 * violation_events / max(1, instance.num_customers),
        "runtime": time.perf_counter() - start,
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"instances": 0}

    numeric_keys = [
        "total_distance",
        "total_travel_time",
        "waiting_time",
        "service_time",
        "total_cost",
        "late_minutes",
        "time_window_violations",
        "capacity_violations",
        "vehicles_excess",
        "route_count",
        "route_overuse",
        "cost_per_customer",
        "late_per_customer",
        "time_window_violations_per_customer",
        "capacity_violations_per_customer",
        "cvr",
        "runtime",
    ]
    result = {"instances": len(rows)}
    for key in numeric_keys:
        result[key] = float(np.mean([float(row.get(key, 0.0)) for row in rows]))
    result["feasibility"] = float(np.mean([1.0 if row.get("feasible") else 0.0 for row in rows]))
    if any("robustness_std" in row for row in rows):
        result["robustness_std"] = float(
            np.mean([float(row.get("robustness_std", 0.0)) for row in rows])
        )
    else:
        result["robustness_std"] = float(np.std([float(row.get("total_cost", 0.0)) for row in rows]))
    result["customers_per_route"] = float(
        np.mean(
            [
                len(row.get("served_customers", [])) / max(1, int(row.get("route_count", 1)))
                for row in rows
            ]
        )
    )
    return result
