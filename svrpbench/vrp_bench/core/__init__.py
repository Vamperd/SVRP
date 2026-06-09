"""Core abstractions for the SVRPBench framework.

These are the only types a new solver or evaluator needs to understand:

- ``Instance``  — a single VRP problem (locations, demands, capacities, etc.).
- ``Solution`` — a set of routes plus solver-reported metrics.
- ``Solver``    — abstract base class. Subclass and decorate with
                  ``@register_solver("my-name")`` to make it discoverable.
"""
from vrp_bench.core.instance import Instance
from vrp_bench.core.solution import Solution
from vrp_bench.core.solver import Solver
from vrp_bench.core.registry import register_solver, get_solver, list_solvers

__all__ = [
    "Instance",
    "Solution",
    "Solver",
    "register_solver",
    "get_solver",
    "list_solvers",
]
