"""Data loading and deterministic splits for standalone TWCVRP RL."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass
class TWInstance:
    """One single-depot VRPTW/TWCVRP instance."""

    name: str
    source: str
    index: int
    vehicle_count: int
    vehicle_capacity: float
    coords: np.ndarray
    demands: np.ndarray
    ready_times: np.ndarray
    due_times: np.ndarray
    service_times: np.ndarray
    map_size: float
    static_time_matrix: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def num_customers(self) -> int:
        return int(self.num_nodes - 1)

    @property
    def horizon(self) -> float:
        return float(max(np.max(self.due_times), self.due_times[0], 1.0))

    @property
    def group(self) -> str:
        return instance_group(self.name)


def load_instances(
    *,
    data_root: str | Path | None = None,
    size: int | str | None = None,
    input_path: str | Path | None = None,
    source: str = "auto",
    pattern: str = "*",
    limit: Optional[int] = None,
) -> list[TWInstance]:
    """Load TWCVRP instances from Solomon text files or benchmark ``.npz`` files."""
    if input_path is not None:
        path = Path(input_path)
        if path.suffix.lower() == ".npz":
            return load_npz_instances(path, limit=limit)
        return [load_solomon_file(path, index=0)]

    if data_root is None:
        raise ValueError("Either data_root or input_path must be provided.")
    if size is None:
        raise ValueError("size is required when loading from data_root.")

    root = Path(data_root)
    if source not in {"auto", "solomon", "npz"}:
        raise ValueError("source must be one of: auto, solomon, npz.")

    if source in {"auto", "solomon"}:
        size_dir = root / str(size)
        if size_dir.exists():
            files = list_instance_files(size_dir, pattern=pattern)
            if limit is not None:
                files = files[:limit]
            return [load_solomon_file(path, index=i) for i, path in enumerate(files)]

    if source in {"auto", "npz"}:
        candidates = sorted(root.glob(f"*{size}*single_depot*.npz"))
        if candidates:
            return load_npz_instances(candidates[0], limit=limit)

    raise FileNotFoundError(f"No TWCVRP data found under {root} for size={size}.")


def list_instance_files(size_dir: str | Path, *, pattern: str = "*") -> list[Path]:
    """Return Solomon/Homberger text files in stable natural order."""
    root = Path(size_dir)
    files = [
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".txt"
    ]
    return sorted(files, key=lambda path: _natural_key(path.name.lower()))


def load_solomon_file(path: str | Path, *, index: int = 0) -> TWInstance:
    """Parse a Solomon/Homberger text file."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    name = non_empty[0] if non_empty else source.stem

    vehicle_count, capacity = _parse_vehicle_header(lines, source)
    rows: list[list[float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and _is_number(parts[0]):
            rows.append([float(value) for value in parts[:7]])
    if not rows:
        raise ValueError(f"{source} has no customer rows.")

    rows_arr = np.asarray(rows, dtype=np.float32)
    rows_arr = rows_arr[np.argsort(rows_arr[:, 0])]
    coords = rows_arr[:, 1:3].astype(np.float32)
    demands = rows_arr[:, 3].astype(np.float32)
    ready = rows_arr[:, 4].astype(np.float32)
    due = rows_arr[:, 5].astype(np.float32)
    service = rows_arr[:, 6].astype(np.float32)

    return TWInstance(
        name=source.stem,
        source=str(source),
        index=index,
        vehicle_count=int(vehicle_count),
        vehicle_capacity=float(capacity),
        coords=coords,
        demands=demands,
        ready_times=ready,
        due_times=due,
        service_times=service,
        map_size=float(max(np.max(coords), 1.0)),
    )


def load_npz_instances(path: str | Path, *, limit: Optional[int] = None) -> list[TWInstance]:
    """Load SVRPBench real_twcvrp ``.npz`` files."""
    source = Path(path)
    instances: list[TWInstance] = []
    with np.load(source, allow_pickle=True) as raw:
        locations = raw["locations"]
        demands = raw["demands"]
        time_windows = raw["time_windows"]
        capacities = raw["vehicle_capacities"]
        num_vehicles = raw.get("num_vehicles")
        time_matrix = raw.get("time_matrix")
        map_size = _map_size(raw.get("map_size", 1000.0))
        count = len(locations) if limit is None else min(limit, len(locations))

        for idx in range(count):
            coords = np.asarray(locations[idx], dtype=np.float32)
            dem = np.asarray(demands[idx], dtype=np.float32)
            tw = np.asarray(time_windows[idx], dtype=np.float32)
            matrix = None
            if time_matrix is not None:
                matrix = np.asarray(time_matrix[idx], dtype=np.float32)
            instances.append(
                TWInstance(
                    name=f"{source.stem}_{idx}",
                    source=str(source),
                    index=idx,
                    vehicle_count=_scalar_index(num_vehicles, idx, default=1, cast=int),
                    vehicle_capacity=_first_capacity(capacities[idx]),
                    coords=coords,
                    demands=dem,
                    ready_times=tw[:, 0],
                    due_times=tw[:, 1],
                    service_times=np.zeros(coords.shape[0], dtype=np.float32),
                    map_size=map_size,
                    static_time_matrix=matrix,
                )
            )
    return instances


def split_instances(
    instances: list[TWInstance],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 1234,
    stratified: bool = True,
) -> dict[str, list[TWInstance]]:
    """Split instances into train/val/test with deterministic file-level splits."""
    if not instances:
        return {"train": [], "val": [], "test": []}

    if not stratified:
        return _split_one_group(instances, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    groups: dict[str, list[TWInstance]] = {}
    for instance in instances:
        groups.setdefault(instance.group, []).append(instance)

    n_total = len(instances)
    n_train, n_val = _target_counts(n_total, train_ratio=train_ratio, val_ratio=val_ratio)
    group_names = sorted(groups)
    group_sizes = {name: len(groups[name]) for name in group_names}
    train_counts = _proportional_counts(group_sizes, n_train)
    remaining_sizes = {name: group_sizes[name] - train_counts[name] for name in group_names}
    val_counts = _proportional_counts(remaining_sizes, n_val)

    split = {"train": [], "val": [], "test": []}
    for group_name in group_names:
        group_seed = seed + sum(ord(char) for char in group_name)
        ordered = list(groups[group_name])
        np.random.default_rng(group_seed).shuffle(ordered)
        n_group_train = train_counts[group_name]
        n_group_val = val_counts[group_name]
        split["train"].extend(ordered[:n_group_train])
        split["val"].extend(ordered[n_group_train : n_group_train + n_group_val])
        split["test"].extend(ordered[n_group_train + n_group_val :])

    for key in split:
        split[key] = sorted(split[key], key=lambda inst: (inst.source, inst.index))
    return split


def instances_to_tensors(instances: list[TWInstance], *, device: torch.device) -> dict[str, torch.Tensor]:
    """Stack same-size instances into policy input tensors."""
    if not instances:
        raise ValueError("No instances provided.")
    sizes = {instance.num_nodes for instance in instances}
    if len(sizes) != 1:
        raise ValueError(f"Batch requires same node count, got {sorted(sizes)}.")

    coords = np.stack([instance.coords for instance in instances]).astype(np.float32)
    demands = np.stack([instance.demands for instance in instances]).astype(np.float32)
    ready = np.stack([instance.ready_times for instance in instances]).astype(np.float32)
    due = np.stack([instance.due_times for instance in instances]).astype(np.float32)
    service = np.stack([instance.service_times for instance in instances]).astype(np.float32)
    map_sizes = np.asarray([instance.map_size for instance in instances], dtype=np.float32)
    capacities = np.asarray([instance.vehicle_capacity for instance in instances], dtype=np.float32)
    horizons = np.asarray([instance.horizon for instance in instances], dtype=np.float32)
    vehicle_counts = np.asarray([instance.vehicle_count for instance in instances], dtype=np.float32)

    coords_norm = coords / np.maximum(map_sizes[:, None, None], 1e-6)
    features = np.concatenate(
        [
            coords_norm,
            demands[:, :, None] / np.maximum(capacities[:, None, None], 1e-6),
            ready[:, :, None] / np.maximum(horizons[:, None, None], 1e-6),
            due[:, :, None] / np.maximum(horizons[:, None, None], 1e-6),
            service[:, :, None] / np.maximum(horizons[:, None, None], 1e-6),
        ],
        axis=-1,
    )

    return {
        "features": torch.tensor(features, dtype=torch.float32, device=device),
        "coords_norm": torch.tensor(coords_norm, dtype=torch.float32, device=device),
        "demands": torch.tensor(demands, dtype=torch.float32, device=device),
        "demands_norm": torch.tensor(
            demands / np.maximum(capacities[:, None], 1e-6),
            dtype=torch.float32,
            device=device,
        ),
        "capacity": torch.tensor(capacities, dtype=torch.float32, device=device),
        "ready_norm": torch.tensor(
            ready / np.maximum(horizons[:, None], 1e-6),
            dtype=torch.float32,
            device=device,
        ),
        "due_norm": torch.tensor(
            due / np.maximum(horizons[:, None], 1e-6),
            dtype=torch.float32,
            device=device,
        ),
        "service_norm": torch.tensor(
            service / np.maximum(horizons[:, None], 1e-6),
            dtype=torch.float32,
            device=device,
        ),
        "vehicle_count": torch.tensor(vehicle_counts, dtype=torch.float32, device=device),
    }


def sample_batch(
    instances: list[TWInstance],
    batch_size: int,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[TWInstance], dict[str, torch.Tensor]]:
    if not instances:
        raise ValueError("Training split is empty.")
    selected = rng.integers(0, len(instances), size=batch_size)
    batch_instances = [instances[int(idx)] for idx in selected]
    return batch_instances, instances_to_tensors(batch_instances, device=device)


def instance_group(name: str) -> str:
    stem = Path(name).stem.lower()
    match = re.match(r"^([a-z]+[0-9]+)_", stem)
    if match:
        return match.group(1)
    match = re.match(r"^(rc|c|r)([0-9])", stem)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return stem[:2] or "unknown"


def _split_one_group(
    instances: list[TWInstance],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[TWInstance]]:
    ordered = list(instances)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    if n == 1:
        return {"train": ordered, "val": [], "test": []}
    if n == 2:
        return {"train": ordered[:1], "val": [], "test": ordered[1:]}

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def _target_counts(n: int, *, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n <= 1:
        return n, 0
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(0, int(round(n * val_ratio)))
    if n >= 3:
        n_val = max(1, n_val)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    return n_train, n_val


def _proportional_counts(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0:
        return {name: 0 for name in group_sizes}
    total = sum(group_sizes.values())
    if total <= 0:
        return {name: 0 for name in group_sizes}

    raw = {name: target * size / total for name, size in group_sizes.items()}
    counts = {name: min(group_sizes[name], int(np.floor(value))) for name, value in raw.items()}
    remaining = target - sum(counts.values())
    order = sorted(
        group_sizes,
        key=lambda name: (raw[name] - np.floor(raw[name]), group_sizes[name], name),
        reverse=True,
    )
    for name in order:
        if remaining <= 0:
            break
        if counts[name] < group_sizes[name]:
            counts[name] += 1
            remaining -= 1
    return counts


def _parse_vehicle_header(lines: list[str], source: Path) -> tuple[int, float]:
    for i, line in enumerate(lines):
        if "CAPACITY" not in line.upper():
            continue
        for candidate in lines[i + 1 : i + 5]:
            parts = candidate.split()
            if len(parts) >= 2 and _is_number(parts[0]) and _is_number(parts[1]):
                return int(float(parts[0])), float(parts[1])
    raise ValueError(f"{source} does not contain a VEHICLE NUMBER/CAPACITY header.")


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _first_capacity(value) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("vehicle_capacities entry is empty.")
    return float(arr[0])


def _scalar_index(values, idx: int, *, default, cast):
    if values is None:
        return default
    arr = np.asarray(values[idx])
    if arr.shape == ():
        return cast(arr.item())
    return cast(arr.reshape(-1)[0])


def _map_size(value) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 1000.0
    return float(np.max(arr))
