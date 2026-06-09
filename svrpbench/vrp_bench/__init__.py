"""SVRPBench: Stochastic Vehicle Routing Problem benchmark."""
from vrp_bench.core import (
    Instance,
    Solution,
    Solver,
    register_solver,
    get_solver,
    list_solvers,
)
from vrp_bench.dataset import load_instances, load_npz

# Importing the solvers package triggers solver registration via decorators.
from vrp_bench import solvers  # noqa: F401

__all__ = [
    "Instance",
    "Solution",
    "Solver",
    "register_solver",
    "get_solver",
    "list_solvers",
    "load_instances",
    "load_npz",
]
