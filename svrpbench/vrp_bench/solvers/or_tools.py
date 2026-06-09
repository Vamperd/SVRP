"""Google OR-Tools VRP solver wrapper."""
from vrp_bench.core import register_solver
from vrp_bench.solvers._legacy import LegacyAdapter

from or_tools_solver import ORToolsSolver as _ORTools, OR_TOOLS_AVAILABLE  # noqa: E402

__all__ = ["ORTools", "OR_TOOLS_AVAILABLE"]


@register_solver("or-tools")
class ORTools(LegacyAdapter):
    """OR-Tools routing engine. Requires ``ortools`` installed."""
    legacy_cls = _ORTools
