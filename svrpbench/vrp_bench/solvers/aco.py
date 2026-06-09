"""Ant Colony Optimization solver."""
from vrp_bench.core import register_solver
from vrp_bench.solvers._legacy import LegacyAdapter

from aco_solver import ACOSolver as _ACO  # noqa: E402


@register_solver("aco")
class ACO(LegacyAdapter):
    """ACO with NN+2opt warm start. See ``aco_solver.py`` for hyperparameters."""
    legacy_cls = _ACO
