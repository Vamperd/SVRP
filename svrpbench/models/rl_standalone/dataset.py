"""Dataset and metric helpers for standalone CVRP reinforcement learning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass
class CVRPInstance:
    """One single-depot CVRP instance from ``vrp_benchmark``."""

    locations: np.ndarray
    demands: np.ndarray
    vehicle_capacity: float
    num_vehicles: int
    map_size: float
    source: str
    index: int

    @property
    def num_customers(self) -> int:
        return int(self.locations.shape[0] - 1)

    @property
    def total_customer_demand(self) -> float:
        return float(np.sum(self.demands[1:]))


def load_cvrp_instances(path: str | Path, limit: Optional[int] = None) -> list[CVRPInstance]:
    """Load single-depot CVRP instances from a SVRPBench ``.npz`` file."""
    source = Path(path)
    with np.load(source, allow_pickle=True) as raw:
        num_depots = _as_scalar(raw.get("num_depots", 1), int)
        if num_depots != 1:
            raise ValueError(
                f"{source} has num_depots={num_depots}; standalone RL v1 supports only one depot."
            )

        locations = raw["locations"]
        demands = raw["demands"]
        capacities = raw["vehicle_capacities"]
        num_vehicles = raw.get("num_vehicles")
        map_size = _map_size(raw.get("map_size", 1000.0))
        count = len(locations) if limit is None else min(limit, len(locations))

        instances: list[CVRPInstance] = []
        for idx in range(count):
            loc = np.asarray(locations[idx], dtype=np.float32)
            dem = np.asarray(demands[idx], dtype=np.float32)
            if loc.ndim != 2 or loc.shape[1] != 2:
                raise ValueError(f"{source}[{idx}] locations must have shape (n_nodes, 2).")
            if dem.ndim != 1 or dem.shape[0] != loc.shape[0]:
                raise ValueError(f"{source}[{idx}] demands must match locations.")

            cap = _first_capacity(capacities[idx])
            nveh = int(_as_scalar(num_vehicles[idx], int)) if num_vehicles is not None else 1
            instances.append(
                CVRPInstance(
                    locations=loc,
                    demands=dem,
                    vehicle_capacity=cap,
                    num_vehicles=nveh,
                    map_size=map_size,
                    source=str(source),
                    index=idx,
                )
            )

    return instances


def infer_num_customers(instances: Iterable[CVRPInstance]) -> int:
    sizes = {instance.num_customers for instance in instances}
    if len(sizes) != 1:
        raise ValueError(f"Expected one customer count per training file, got {sorted(sizes)}.")
    return next(iter(sizes))


def split_instances(
    instances: list[CVRPInstance],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 1234,
) -> dict[str, list[CVRPInstance]]:
    """Deterministically split benchmark instances by instance index."""
    ordered = list(instances)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    if n <= 1:
        return {"train": ordered, "val": [], "test": []}
    if n == 2:
        return {"train": ordered[:1], "val": [], "test": ordered[1:]}

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    split = {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }
    return {key: sorted(value, key=lambda inst: inst.index) for key, value in split.items()}


def make_batch(
    instances: list[CVRPInstance],
    batch_size: int,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample a training batch with replacement."""
    if not instances:
        raise ValueError("No instances loaded.")
    selected = rng.integers(0, len(instances), size=batch_size)
    batch_instances = [instances[int(i)] for i in selected]
    return instances_to_tensors(batch_instances, device=device)


def instances_to_tensors(
    instances: list[CVRPInstance],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack instances into tensors for model rollout."""
    map_sizes = np.asarray([inst.map_size for inst in instances], dtype=np.float32)
    locations = np.stack([inst.locations for inst in instances], axis=0).astype(np.float32)
    demands = np.stack([inst.demands for inst in instances], axis=0).astype(np.float32)
    capacities = np.asarray([inst.vehicle_capacity for inst in instances], dtype=np.float32)

    return {
        "locations": torch.tensor(locations, dtype=torch.float32, device=device),
        "locations_norm": torch.tensor(
            locations / map_sizes[:, None, None],
            dtype=torch.float32,
            device=device,
        ),
        "demands": torch.tensor(demands, dtype=torch.float32, device=device),
        "demands_norm": torch.tensor(
            demands / np.maximum(capacities[:, None], 1e-6),
            dtype=torch.float32,
            device=device,
        ),
        "capacity": torch.tensor(capacities, dtype=torch.float32, device=device),
    }


def route_distance(locations: np.ndarray, route: list[int]) -> float:
    if len(route) < 2:
        return 0.0
    coords = locations[np.asarray(route, dtype=np.int64)]
    steps = coords[1:] - coords[:-1]
    return float(np.linalg.norm(steps, axis=1).sum())


def evaluate_route(instance: CVRPInstance, route: list[int]) -> dict:
    """Return simple route metrics for one CVRP solution."""
    customers = [node for node in route if node != 0]
    unique_customers = sorted(set(customers))
    capacity_used = float(np.sum(instance.demands[unique_customers])) if unique_customers else 0.0
    duplicate_visits = len(customers) - len(unique_customers)
    missing = sorted(set(range(1, instance.num_customers + 1)) - set(unique_customers))
    feasible = (
        duplicate_visits == 0
        and not missing
        and capacity_used <= instance.vehicle_capacity * max(1, instance.num_vehicles) + 1e-6
    )
    return {
        "routes": [route],
        "total_distance": route_distance(instance.locations, route),
        "feasible": bool(feasible),
        "capacity_used": [capacity_used],
        "served_customers": unique_customers,
        "missing_customers": missing,
        "duplicate_visits": duplicate_visits,
        "vehicle_capacity": instance.vehicle_capacity,
        "num_vehicles": instance.num_vehicles,
    }


def _as_scalar(value, cast):
    arr = np.asarray(value)
    if arr.shape == ():
        return cast(arr.item())
    return cast(arr.reshape(-1)[0])


def _first_capacity(value) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("vehicle_capacities entry is empty.")
    return float(arr[0])


def _map_size(value) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 1000.0
    return float(np.max(arr))
