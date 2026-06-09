"""Shared helpers for SVRPBench RL4CO reproduction.

The functions in this module intentionally avoid importing RL4CO, torch, or
tensordict at module import time. This keeps the benchmark package usable when
the optional RL dependencies have not yet been installed in the user's conda
``svrp`` environment.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


RL_DEPENDENCY_MESSAGE = (
    "RL dependencies are not available. Activate the conda environment with "
    "'conda activate svrp' and install the packages listed in "
    "svrpbench/models/rl/requirements-rl.txt. See "
    "svrpbench/models/rl/README_RL_REPRO.md for the manual steps."
)


def require_tensor_libs():
    """Import torch and TensorDict only when an RL operation needs them."""
    try:
        import torch
        from tensordict import TensorDict
    except ModuleNotFoundError as exc:
        raise RuntimeError(RL_DEPENDENCY_MESSAGE) from exc
    return torch, TensorDict


def infer_variant(instance: Any) -> str:
    """Infer the RL4CO problem variant from an ``Instance``-like object."""
    return "twvrp" if getattr(instance, "time_windows", None) is not None else "cvrp"


def ensure_single_depot(instance: Any) -> None:
    """Fail fast for unsupported multi-depot instances."""
    num_depots = int(getattr(instance, "num_depots", 1))
    if num_depots != 1:
        raise ValueError(
            "RL reproduction v1 supports exactly one depot. "
            f"Received {num_depots} depots from {getattr(instance, 'metadata', {})}."
        )


def num_customers(instance: Any) -> int:
    ensure_single_depot(instance)
    return int(getattr(instance, "num_nodes")) - 1


def get_map_size(instance: Any, default: float = 1000.0) -> float:
    """Return the scalar map size used to normalize coordinates."""
    metadata = getattr(instance, "metadata", {}) or {}
    if "map_size" in metadata:
        return _scalar_map_size(metadata["map_size"], default)

    source = metadata.get("source")
    if source:
        try:
            with np.load(source, allow_pickle=True) as data:
                if "map_size" in data:
                    return _scalar_map_size(data["map_size"], default)
        except Exception:
            pass

    return float(default)


def instance_to_tensordict(
    instance: Any,
    *,
    variant: Optional[str] = None,
    map_size: Optional[float] = None,
    max_time: float = 1440.0,
):
    """Convert one benchmark ``Instance`` into the RL4CO TensorDict format."""
    ensure_single_depot(instance)
    variant = variant or infer_variant(instance)
    map_size = float(map_size or get_map_size(instance))

    locations = np.asarray(instance.locations, dtype=np.float32)[None, ...]
    demands = np.asarray(instance.demands, dtype=np.float32)[None, ...]
    capacities = _first_capacities(np.asarray([instance.vehicle_capacities], dtype=object))

    if variant == "cvrp":
        return _arrays_to_cvrp_tensordict(
            locations=locations,
            demands=demands,
            capacities=capacities,
            map_size=map_size,
        )
    if variant in {"twvrp", "vrptw"}:
        if instance.time_windows is None:
            raise ValueError("TWVRP conversion requires instance.time_windows.")
        time_windows = np.asarray(instance.time_windows, dtype=np.float32)[None, ...]
        return _arrays_to_tw_tensordict(
            locations=locations,
            demands=demands,
            capacities=capacities,
            time_windows=time_windows,
            map_size=map_size,
            max_time=max_time,
        )
    raise ValueError(f"Unsupported RL variant: {variant!r}")


def npz_dict_to_tensordict(
    data: dict,
    *,
    variant: Optional[str] = None,
    map_size: float = 1000.0,
    max_time: float = 1440.0,
):
    """Convert a raw SVRPBench ``.npz`` dict into a batched RL4CO TensorDict."""
    variant = variant or ("twvrp" if "time_windows" in data else "cvrp")
    num_depots = int(np.asarray(data.get("num_depots", 1)).item())
    if num_depots != 1:
        raise ValueError(
            "RL reproduction v1 supports raw .npz conversion only for single-depot files."
        )

    locations = _stack_field(data.get("locations", data.get("locs")), np.float32, 3)
    demands = _stack_field(data.get("demands", data.get("demand")), np.float32, 2)
    capacities = _first_capacities(data.get("vehicle_capacities", data.get("capacity")))

    if variant == "cvrp":
        return _arrays_to_cvrp_tensordict(locations, demands, capacities, map_size)
    if variant in {"twvrp", "vrptw"}:
        time_windows = _stack_field(data["time_windows"], np.float32, 3)
        return _arrays_to_tw_tensordict(
            locations,
            demands,
            capacities,
            time_windows,
            map_size=map_size,
            max_time=max_time,
        )
    raise ValueError(f"Unsupported RL variant: {variant!r}")


def actions_to_routes(
    actions: Sequence[int],
    *,
    num_nodes: int,
    depot_index: int = 0,
) -> list[list[int]]:
    """Convert an RL4CO action sequence into benchmark routes.

    RL4CO uses node index 0 for the depot and customer indices 1..N. Consecutive
    depot actions and empty routes are removed, while customer duplicates are
    preserved so the benchmark feasibility checker can penalize them.
    """
    routes: list[list[int]] = []
    current = [depot_index]

    for raw_action in actions:
        node = int(raw_action)
        if node < 0 or node >= num_nodes:
            continue
        if node == depot_index:
            if len(current) > 1:
                current.append(depot_index)
                routes.append(current)
                current = [depot_index]
            continue
        current.append(node)

    if len(current) > 1:
        current.append(depot_index)
        routes.append(current)

    return routes or [[depot_index, depot_index]]


def best_action_sequence(policy_output: dict) -> list[int]:
    """Extract the best action sequence from a greedy RL4CO policy output."""
    torch, _ = require_tensor_libs()
    actions = policy_output["actions"].detach().cpu()
    rewards = policy_output.get("reward")
    rewards = rewards.detach().cpu() if rewards is not None else None

    if actions.ndim == 1:
        return [int(x) for x in actions.tolist()]

    if actions.ndim == 2:
        return [int(x) for x in actions[0].tolist()]

    batch_actions = actions[0]
    selected = 0
    if rewards is not None:
        reward_view = rewards[0] if rewards.ndim > 1 else rewards
        if reward_view.ndim > 0 and reward_view.numel() == batch_actions.shape[0]:
            selected = int(torch.argmax(reward_view).item())

    return [int(x) for x in batch_actions[selected].reshape(-1).tolist()]


def resolve_checkpoint(
    checkpoint_root: str | Path,
    *,
    variant: str,
    algo: str,
    num_loc: int,
) -> Path:
    """Find an exact or nearest-size checkpoint for a variant/algorithm pair."""
    root = Path(checkpoint_root)
    exact_dir = root / variant / f"{algo}_{num_loc}"
    exact = _best_checkpoint_in_dir(exact_dir)
    if exact is not None:
        return exact

    variant_dir = root / variant
    candidates: list[tuple[int, Path]] = []
    if variant_dir.exists():
        pattern = re.compile(rf"^{re.escape(algo)}_(\d+)$")
        for child in variant_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if not match:
                continue
            ckpt = _best_checkpoint_in_dir(child)
            if ckpt is not None:
                candidates.append((int(match.group(1)), ckpt))

    if candidates:
        smaller = [(size, ckpt) for size, ckpt in candidates if size <= num_loc]
        pool = smaller or candidates
        return max(pool, key=lambda item: item[0])[1]

    raise FileNotFoundError(
        "No RL checkpoint found. Expected files such as "
        f"{exact_dir / 'last.ckpt'} or epoch checkpoints under {variant_dir}."
    )


def _arrays_to_cvrp_tensordict(
    locations: np.ndarray,
    demands: np.ndarray,
    capacities: np.ndarray,
    map_size: float,
):
    torch, TensorDict = require_tensor_libs()
    locs_norm = torch.tensor(locations / map_size, dtype=torch.float32)
    cap = torch.tensor(capacities, dtype=torch.float32)
    demand = torch.tensor(demands[:, 1:] / capacities[:, None], dtype=torch.float32)

    return TensorDict(
        {
            "locs": locs_norm[:, 1:, :],
            "depot": locs_norm[:, 0, :],
            "demand": demand,
            "capacity": cap,
        },
        batch_size=torch.Size([locations.shape[0]]),
    )


def _arrays_to_tw_tensordict(
    locations: np.ndarray,
    demands: np.ndarray,
    capacities: np.ndarray,
    time_windows: np.ndarray,
    *,
    map_size: float,
    max_time: float,
):
    torch, TensorDict = require_tensor_libs()
    batch_size, num_nodes, _ = locations.shape
    locs_norm = torch.tensor(locations / map_size, dtype=torch.float32)
    demand_norm = torch.tensor(demands[:, 1:] / capacities[:, None], dtype=torch.float32)

    return TensorDict(
        {
            "locs": locs_norm[:, 1:, :],
            "depot": locs_norm[:, 0:1, :],
            "demand_linehaul": demand_norm,
            "demand_backhaul": torch.zeros_like(demand_norm),
            "distance_limit": torch.full((batch_size,), float("inf"), dtype=torch.float32),
            "time_windows": torch.tensor(time_windows / max_time, dtype=torch.float32),
            "service_time": torch.zeros((batch_size, num_nodes), dtype=torch.float32),
            "vehicle_capacity": torch.tensor(capacities, dtype=torch.float32),
            "capacity_original": torch.tensor(capacities, dtype=torch.float32),
            "open_route": torch.zeros((batch_size,), dtype=torch.bool),
            "speed": torch.ones((batch_size, num_nodes, num_nodes), dtype=torch.float32),
        },
        batch_size=torch.Size([batch_size]),
    )


def _stack_field(value: Any, dtype: Any, expected_ndim: int) -> np.ndarray:
    if value is None:
        raise KeyError("Required dataset field is missing.")
    raw = np.asarray(value)
    if raw.dtype == object:
        arr = np.stack([np.asarray(item, dtype=dtype) for item in raw], axis=0)
    else:
        arr = raw.astype(dtype)
    if arr.ndim == expected_ndim - 1:
        arr = arr[None, ...]
    return arr


def _first_capacities(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype == object:
        out = []
        for item in raw:
            flat = np.asarray(item, dtype=np.float32).reshape(-1)
            if flat.size == 0:
                raise ValueError("Vehicle capacity array is empty.")
            out.append(float(flat[0]))
        return np.asarray(out, dtype=np.float32)
    arr = raw.astype(np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr[:, 0]


def _scalar_map_size(value: Any, default: float) -> float:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size:
            return float(np.max(arr))
    except Exception:
        pass
    return float(default)


def _best_checkpoint_in_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    preferred = [path / "last.ckpt"]
    preferred.extend(sorted(path.glob("last-v*.ckpt"), reverse=True))
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    checkpoints = list(path.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item.stat().st_mtime)

