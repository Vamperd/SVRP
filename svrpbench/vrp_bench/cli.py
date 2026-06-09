"""Command-line interface: ``python -m vrp_bench``.

Examples:

    python -m vrp_bench list
    python -m vrp_bench solve --solver aco --data path/to/instances.npz
    python -m vrp_bench solve --solver or-tools --data file.npz --limit 5 --realizations 3
"""
from __future__ import annotations

import argparse
import json
import sys

from vrp_bench import dataset, evaluation
from vrp_bench.core import get_solver, list_solvers


def _cmd_list(_args) -> int:
    print("Registered solvers:")
    for name in list_solvers():
        print(f"  - {name}")
    return 0


def _cmd_solve(args) -> int:
    instances = dataset.load_instances(args.data, limit=args.limit)
    if not instances:
        print(f"No instances found in {args.data}", file=sys.stderr)
        return 1
    solver = get_solver(args.solver)()
    result = evaluation.evaluate(solver, instances, num_realizations=args.realizations)
    out = json.dumps(result["aggregate"], indent=2)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output}")
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vrp_bench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registered solvers").set_defaults(func=_cmd_list)

    s = sub.add_parser("solve", help="Run a solver over a dataset")
    s.add_argument("--solver", required=True, help=f"one of: {', '.join(list_solvers())}")
    s.add_argument("--data", required=True, help="path to .npz dataset")
    s.add_argument("--limit", type=int, default=None, help="max instances to evaluate")
    s.add_argument("--realizations", type=int, default=1,
                   help="stochastic travel-time realizations to average")
    s.add_argument("--output", help="optional JSON file for full per-instance results")
    s.set_defaults(func=_cmd_solve)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
