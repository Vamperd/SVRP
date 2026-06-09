"""Built-in solvers.

Each solver lives in its own module and registers itself with the central
registry on import. Adding a new solver is a three-line exercise:

    from vrp_bench.core import Solver, register_solver

    @register_solver("my-solver")
    class MySolver(Solver):
        def solve(self, instance, *, num_realizations=1):
            ...

Then add an import below so it gets registered at package load time.
"""
# Importing each module triggers its @register_solver decorator.
from vrp_bench.solvers import nn2opt as _nn2opt    # noqa: F401
from vrp_bench.solvers import aco as _aco          # noqa: F401
from vrp_bench.solvers import tabu as _tabu        # noqa: F401
from vrp_bench.solvers import or_tools as _ortools # noqa: F401
from vrp_bench.solvers import rl as _rl            # noqa: F401
