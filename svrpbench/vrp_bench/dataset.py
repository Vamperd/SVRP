"""Loading SVRPBench ``.npz`` files into ``Instance`` objects.

Each ``.npz`` produced by the benchmark holds a *batch* of instances: every
field has a leading instance dimension. ``load_instances`` slices that batch
into a list of ``Instance`` objects so downstream code never deals with the
batched representation directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np

from vrp_bench.core.instance import Instance

PathLike = Union[str, Path]


def load_npz(path: PathLike) -> dict:
    """Return the raw batched dict stored in ``path``."""
    return dict(np.load(path, allow_pickle=True))


def _slice(arr, i: int):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return np.asarray(arr[i])
    return np.asarray(arr[i])


def load_instances(path: PathLike, limit: Optional[int] = None) -> List[Instance]:
    """Load up to ``limit`` instances from an SVRPBench ``.npz`` file."""
    raw = load_npz(path)

    locs = raw.get("locations", raw.get("locs"))
    if locs is None:
        raise KeyError(f"{path}: no 'locations' or 'locs' field found")
    n = len(locs) if limit is None else min(limit, len(locs))

    demands = raw.get("demands", raw.get("demand"))
    caps = raw.get("vehicle_capacities", raw.get("capacity"))
    nveh = raw.get("num_vehicles")
    tw = raw.get("time_windows")
    tm = raw.get("time_matrix")
    appear = raw.get("appear_times", raw.get("appear_time"))

    out: List[Instance] = []
    for i in range(n):
        cap_i = _slice(caps, i)
        if cap_i.ndim == 0:
            cap_i = np.array([float(cap_i)])
        out.append(
            Instance(
                locations=_slice(locs, i),
                demands=_slice(demands, i),
                vehicle_capacities=cap_i,
                num_vehicles=int(np.asarray(nveh[i]).item()) if nveh is not None
                              else int(cap_i.shape[0]),
                time_windows=_slice(tw, i) if tw is not None else None,
                time_matrix=_slice(tm, i) if tm is not None else None,
                appear_times=_slice(appear, i) if appear is not None else None,
                metadata={"source": str(path), "index": i},
            )
        )
    return out


def iter_instances(paths: List[PathLike]) -> Iterator[Instance]:
    for p in paths:
        yield from load_instances(p)
