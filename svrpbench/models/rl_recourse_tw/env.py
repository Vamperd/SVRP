"""Event-driven TWCVRP environment with online recourse decisions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import SOLOMON_TW_DIR  # noqa: F401  (ensures sibling imports work)
from evaluator import evaluate_routes


@dataclass
class DecisionState:
    vehicle_idx: int
    current_node: int
    current_time: float
    remaining_capacity: float
    legal_mask: np.ndarray
    relaxed_mask: np.ndarray
    forced_late: bool


class EventDrivenTWEnv:
    """Asynchronous multi-vehicle TWCVRP simulator.

    Each step advances to the earliest available vehicle with at least one
    capacity-feasible unserved customer. The policy then chooses one customer
    for that single vehicle, which keeps the action space at O(customers).
    """

    def __init__(
        self,
        instance,
        travel_time_matrix: np.ndarray,
        *,
        mask_late: bool = True,
        traffic_time_scale: str = "raw",
        drifted_due_times: np.ndarray | None = None,
    ):
        self.instance = instance
        self.travel = np.asarray(travel_time_matrix, dtype=np.float32)
        self.mask_late = bool(mask_late)
        self.traffic_time_scale = traffic_time_scale
        self.num_customers = int(instance.num_customers)
        self.vehicle_count = max(1, int(instance.vehicle_count))
        self.capacity = float(instance.vehicle_capacity)
        if drifted_due_times is not None:
            self.drifted_due_times = np.asarray(drifted_due_times, dtype=np.float32)
            self._elastic = True
        else:
            self.drifted_due_times = np.asarray(self.instance.due_times, dtype=np.float32)
            self._elastic = False
        self.reset()

    def reset(self) -> None:
        self.current_nodes = np.zeros(self.vehicle_count, dtype=np.int64)
        self.available_times = np.zeros(self.vehicle_count, dtype=np.float32)
        self.remaining_capacities = np.full(self.vehicle_count, self.capacity, dtype=np.float32)
        self.active = np.ones(self.vehicle_count, dtype=bool)
        self.served = np.zeros(self.num_customers + 1, dtype=bool)
        self.served[0] = True
        self.routes: list[list[int]] = [[0] for _ in range(self.vehicle_count)]
        self.forced_late_actions = 0
        self.steps = 0

    @property
    def done(self) -> bool:
        return bool(np.all(self.served[1:]))

    @property
    def horizon(self) -> float:
        return float(max(np.max(self.instance.due_times), self.instance.due_times[0], 1.0))

    def decision_state(self) -> DecisionState | None:
        if self.done:
            return None

        for vehicle_idx in np.argsort(self.available_times):
            vehicle_idx = int(vehicle_idx)
            if not self.active[vehicle_idx]:
                continue
            relaxed_mask = self._capacity_mask(vehicle_idx)
            if not relaxed_mask.any():
                self._retire_vehicle(vehicle_idx)
                continue
            legal_mask = self._time_mask(vehicle_idx, relaxed_mask) if self.mask_late else relaxed_mask.copy()
            forced_late = False
            if not legal_mask.any():
                legal_mask = relaxed_mask.copy()
                forced_late = True
            return DecisionState(
                vehicle_idx=vehicle_idx,
                current_node=int(self.current_nodes[vehicle_idx]),
                current_time=float(self.available_times[vehicle_idx]),
                remaining_capacity=float(self.remaining_capacities[vehicle_idx]),
                legal_mask=legal_mask,
                relaxed_mask=relaxed_mask,
                forced_late=forced_late,
            )
        return None

    def step(self, customer: int, state: DecisionState | None = None) -> None:
        if state is None:
            state = self.decision_state()
        if state is None:
            return

        node = int(customer)
        vehicle_idx = int(state.vehicle_idx)
        prev = int(self.current_nodes[vehicle_idx])
        travel_time = float(self.travel[prev, node])
        arrival = float(self.available_times[vehicle_idx]) + travel_time
        ready = float(self.instance.ready_times[node])
        wait = max(0.0, ready - arrival)
        service_start = arrival + wait
        depart = service_start + float(self.instance.service_times[node])

        self.available_times[vehicle_idx] = depart
        self.current_nodes[vehicle_idx] = node
        self.remaining_capacities[vehicle_idx] -= float(self.instance.demands[node])
        self.served[node] = True
        self.routes[vehicle_idx].append(node)
        self.steps += 1
        if state.forced_late:
            self.forced_late_actions += 1

    def finish_routes(self) -> list[list[int]]:
        finished: list[list[int]] = []
        for route in self.routes:
            if len(route) <= 1:
                continue
            if route[-1] != 0:
                route = route + [0]
            finished.append(route)
        return finished

    def metrics(self) -> dict:
        row = evaluate_routes(self.instance, self.finish_routes(), self.travel)
        row["forced_late_actions"] = int(self.forced_late_actions)
        row["env_steps"] = int(self.steps)
        return row

    def candidate_features(self, state: DecisionState) -> np.ndarray:
        customers = np.arange(1, self.num_customers + 1)
        current = int(state.current_node)
        current_time = float(state.current_time)
        travel = self.travel[current, customers].astype(np.float32)
        arrival = current_time + travel
        ready = self.instance.ready_times[customers].astype(np.float32)
        due = self.drifted_due_times[customers].astype(np.float32)
        original_due = self.instance.due_times[customers].astype(np.float32)
        wait = np.maximum(0.0, ready - arrival)
        service_start = arrival + wait
        late = np.maximum(0.0, service_start - due)
        slack = due - service_start
        width = np.maximum(due - ready, 0.0)
        demand = self.instance.demands[customers].astype(np.float32)
        coords = self.instance.coords[customers].astype(np.float32)
        map_size = max(float(self.instance.map_size), 1.0)
        horizon = max(self.horizon, 1e-6)
        scaled_arrival = np.asarray(
            [self.scale_time(value) for value in arrival],
            dtype=np.float32,
        )
        angle = 2.0 * np.pi * scaled_arrival / 1440.0
        drift_offset = np.maximum(due - original_due, 0.0) / horizon
        return np.stack(
            [
                coords[:, 0] / map_size,
                coords[:, 1] / map_size,
                demand / max(self.capacity, 1e-6),
                ready / horizon,
                due / horizon,
                width / horizon,
                travel / horizon,
                arrival / horizon,
                wait / horizon,
                late / horizon,
                np.clip(slack / horizon, -1.0, 1.0),
                np.sin(angle),
                np.cos(angle),
                drift_offset,
            ],
            axis=-1,
        ).astype(np.float32)

    def vehicle_features(self, state: DecisionState) -> np.ndarray:
        current_xy = self.instance.coords[int(state.current_node)].astype(np.float32)
        depot_xy = self.instance.coords[0].astype(np.float32)
        map_size = max(float(self.instance.map_size), 1.0)
        horizon = max(self.horizon, 1e-6)
        scaled_time = self.scale_time(float(state.current_time))
        angle = 2.0 * np.pi * scaled_time / 1440.0
        remaining_customers = float(np.sum(~self.served[1:])) / max(1, self.num_customers)
        return np.asarray(
            [
                current_xy[0] / map_size,
                current_xy[1] / map_size,
                depot_xy[0] / map_size,
                depot_xy[1] / map_size,
                float(state.current_time) / horizon,
                float(state.remaining_capacity) / max(self.capacity, 1e-6),
                remaining_customers,
                np.sin(angle),
                np.cos(angle),
            ],
            dtype=np.float32,
        )

    def scale_time(self, current_time: float) -> float:
        if self.traffic_time_scale == "raw":
            return float(current_time)
        if self.traffic_time_scale == "depot_day":
            start = float(self.instance.ready_times[0])
            end = float(self.instance.due_times[0])
            frac = min(max((float(current_time) - start) / max(end - start, 1e-6), 0.0), 1.0)
            return frac * 1440.0
        raise ValueError("traffic_time_scale must be 'raw' or 'depot_day'.")

    def _capacity_mask(self, vehicle_idx: int) -> np.ndarray:
        mask = np.zeros(self.num_customers + 1, dtype=bool)
        remaining = float(self.remaining_capacities[vehicle_idx])
        for node in range(1, self.num_customers + 1):
            if self.served[node]:
                continue
            if float(self.instance.demands[node]) <= remaining + 1e-6:
                mask[node] = True
        return mask

    def _time_mask(self, vehicle_idx: int, base_mask: np.ndarray) -> np.ndarray:
        mask = base_mask.copy()
        current = int(self.current_nodes[vehicle_idx])
        current_time = float(self.available_times[vehicle_idx])
        due_times = self.drifted_due_times if self._elastic else self.instance.due_times
        for node in np.where(base_mask)[0]:
            arrival = current_time + float(self.travel[current, int(node)])
            if arrival - float(due_times[int(node)]) > 1e-6:
                mask[int(node)] = False
        return mask

    def _retire_vehicle(self, vehicle_idx: int) -> None:
        self.active[vehicle_idx] = False
        if len(self.routes[vehicle_idx]) > 1 and self.routes[vehicle_idx][-1] != 0:
            current = int(self.current_nodes[vehicle_idx])
            self.available_times[vehicle_idx] += float(self.travel[current, 0])
            self.routes[vehicle_idx].append(0)

