from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Solution:
    """A VRP solution: routes plus the metrics the solver reports.

    A route is a list of node indices. The depot endpoints are *not* required;
    consumers should treat each route as the ordered customer visits performed
    by one vehicle starting and ending at its depot.
    """

    routes: List[List[int]]
    total_cost: float
    runtime: float = 0.0
    feasibility: float = 1.0       # fraction of demand served within constraints
    cvr: float = 0.0               # constraint-violation rate (%)
    waiting_time: float = 0.0
    robustness: float = 0.0        # variance across stochastic realizations
    extras: dict = field(default_factory=dict)

    def as_metrics(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "runtime": self.runtime,
            "feasibility": self.feasibility,
            "cvr": self.cvr,
            "waiting_time": self.waiting_time,
            "robustness": self.robustness,
        }

    @classmethod
    def from_metrics(cls, metrics: dict, routes: List[List[int]] | None = None) -> "Solution":
        return cls(
            routes=routes or [],
            total_cost=float(metrics.get("total_cost", 0.0)),
            runtime=float(metrics.get("runtime", 0.0)),
            feasibility=float(metrics.get("feasibility", 1.0)),
            cvr=float(metrics.get("cvr", 0.0)),
            waiting_time=float(metrics.get("waiting_time", 0.0)),
            robustness=float(metrics.get("robustness", 0.0)),
            extras={k: v for k, v in metrics.items()
                    if k not in {"total_cost", "runtime", "feasibility",
                                 "cvr", "waiting_time", "robustness"}},
        )
