"""Adapter glue between new ``Solver`` interface and legacy ``VRPSolverBase``.

The legacy solvers in ``vrp_bench/*_solver.py`` were written before the
package layout existed: they import sibling modules by bare name (``from
city import Map``). To keep them working unchanged, we add this directory
to ``sys.path`` once on first import.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Type

from vrp_bench.core import Instance, Solution, Solver

_LEGACY_DIR = Path(__file__).resolve().parent.parent
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))


class LegacyAdapter(Solver):
    """Wraps a ``VRPSolverBase`` subclass behind the new ``Solver`` API."""

    legacy_cls: Type  # set by subclasses

    def __init__(self, **legacy_kwargs):
        self._legacy_kwargs = legacy_kwargs

    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        legacy = self.legacy_cls(instance.to_legacy_dict(), **self._legacy_kwargs) \
            if self._legacy_kwargs else self.legacy_cls(instance.to_legacy_dict())
        t0 = time.time()
        metrics = legacy.solve_instance(0, num_realizations=num_realizations)
        if "runtime" not in metrics:
            metrics["runtime"] = time.time() - t0
        return Solution.from_metrics(metrics, routes=metrics.get("routes", []))
