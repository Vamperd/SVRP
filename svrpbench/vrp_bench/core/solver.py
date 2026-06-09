from __future__ import annotations

from abc import ABC, abstractmethod

from vrp_bench.core.instance import Instance
from vrp_bench.core.solution import Solution


class Solver(ABC):
    """Abstract base class every SVRPBench solver implements.

    The contract is intentionally narrow (single responsibility): take one
    ``Instance``, return one ``Solution``. Batch evaluation, dataset loading
    and metric aggregation are handled elsewhere.

    Construction-time configuration (hyperparameters, time budget, ...) is
    passed via ``__init__`` kwargs by the subclass — never via ``solve``.
    This keeps the call site uniform across solvers (open/closed: add new
    solvers without changing the runner).
    """

    name: str = "solver"

    @abstractmethod
    def solve(self, instance: Instance, *, num_realizations: int = 1) -> Solution:
        """Return a ``Solution`` for ``instance``.

        ``num_realizations`` controls how many stochastic travel-time samples
        the solver should average over (1 = deterministic).
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
