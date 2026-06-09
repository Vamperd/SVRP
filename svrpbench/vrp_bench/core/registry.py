"""Solver registry — discover solvers by name without import-time coupling.

Usage:

    @register_solver("aco")
    class ACO(Solver):
        ...

    solver_cls = get_solver("aco")
"""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from vrp_bench.core.solver import Solver

_REGISTRY: Dict[str, Type[Solver]] = {}


def register_solver(name: str) -> Callable[[Type[Solver]], Type[Solver]]:
    """Decorator that registers a ``Solver`` subclass under ``name``."""
    def _wrap(cls: Type[Solver]) -> Type[Solver]:
        if not issubclass(cls, Solver):
            raise TypeError(f"{cls.__name__} must subclass Solver")
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Solver name {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_solver(name: str) -> Type[Solver]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown solver {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_solvers() -> List[str]:
    return sorted(_REGISTRY)
