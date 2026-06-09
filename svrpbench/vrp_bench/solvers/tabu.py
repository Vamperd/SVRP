"""Tabu Search solver."""
from vrp_bench.core import register_solver
from vrp_bench.solvers._legacy import LegacyAdapter

from tabu_search_solver import TabuSearchSolver as _Tabu  # noqa: E402


@register_solver("tabu")
class Tabu(LegacyAdapter):
    """Tabu search with neighborhood moves over an NN+2opt seed."""
    legacy_cls = _Tabu
