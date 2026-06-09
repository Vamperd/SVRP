"""Nearest-neighbor construction with 2-opt local search."""
from vrp_bench.core import register_solver
from vrp_bench.solvers._legacy import LegacyAdapter

from nn_2opt_solver import NN2optSolver as _NN2opt  # noqa: E402  (sys.path patched in _legacy)


@register_solver("nn2opt")
class NN2opt(LegacyAdapter):
    """Greedy nearest-neighbor route, refined by 2-opt swaps. Fast baseline."""
    legacy_cls = _NN2opt
