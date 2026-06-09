from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Instance:
    """A single VRP problem instance.

    All array shapes are per-instance (no batch dimension). Fields beyond
    ``locations``/``demands``/``vehicle_capacities`` are optional so the same
    type can describe CVRP, TWCVRP and stochastic variants without a class
    hierarchy.
    """

    locations: np.ndarray            # (n_nodes, 2) — node coordinates, depots first
    demands: np.ndarray              # (n_nodes,)   — depot demand is 0
    vehicle_capacities: np.ndarray   # (n_vehicles,)
    num_vehicles: int

    time_windows: Optional[np.ndarray] = None  # (n_nodes, 2) — (open, close)
    time_matrix: Optional[np.ndarray] = None   # (n_nodes, n_nodes)
    appear_times: Optional[np.ndarray] = None  # (n_nodes,)
    metadata: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.locations.shape[0])

    @property
    def num_depots(self) -> int:
        return int(np.sum(self.demands == 0))

    @property
    def has_time_windows(self) -> bool:
        return self.time_windows is not None

    def to_legacy_dict(self) -> dict:
        """Adapter for legacy solvers that expect a batched dict."""
        d = {
            "locations": np.asarray([self.locations]),
            "demands": np.asarray([self.demands]),
            "num_vehicles": np.asarray([self.num_vehicles]),
            "vehicle_capacities": np.asarray([self.vehicle_capacities]),
        }
        if self.time_windows is not None:
            d["time_windows"] = np.asarray([self.time_windows])
        if self.time_matrix is not None:
            d["time_matrix"] = np.asarray([self.time_matrix])
        if self.appear_times is not None:
            d["appear_times"] = np.asarray([self.appear_times])
        return d
