"""High-level evaluation: run a solver across many instances and aggregate."""
from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Iterable, List

from vrp_bench.core import Instance, Solution, Solver


def evaluate(
    solver: Solver,
    instances: Iterable[Instance],
    *,
    num_realizations: int = 1,
) -> dict:
    """Run ``solver`` on each instance and return per-instance + aggregate metrics."""
    per_instance: List[Solution] = []
    for inst in instances:
        per_instance.append(solver.solve(inst, num_realizations=num_realizations))

    if not per_instance:
        return {"per_instance": [], "aggregate": {}}

    keys = ["total_cost", "runtime", "feasibility", "cvr", "waiting_time", "robustness"]
    aggregate = {k: mean(getattr(s, k) for s in per_instance) for k in keys}
    aggregate["n_instances"] = len(per_instance)

    return {
        "per_instance": [asdict(s) for s in per_instance],
        "aggregate": aggregate,
    }
