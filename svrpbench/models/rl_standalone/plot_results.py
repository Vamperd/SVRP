#!/usr/bin/env python
"""Plot standalone RL CVRP JSON results without importing torch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dataset import load_cvrp_instances


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CVRP routes from solve.py JSON output.")
    parser.add_argument("--data", required=True, help="Path to the original CVRP .npz file.")
    parser.add_argument("--solutions", required=True, help="Path to solve.py JSON output.")
    parser.add_argument("--plot_dir", default="results/plots")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow_duplicate_openmp",
        action="store_true",
        help="Windows workaround for OpenMP runtime conflicts.",
    )
    return parser


def import_pyplot():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_plot(plt, instance, result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    loc = instance.locations
    route = result["routes"][0]
    route_xy = loc[np.asarray(route, dtype=np.int64)]

    plt.figure(figsize=(6, 6))
    plt.scatter(loc[1:, 0], loc[1:, 1], c="#277da1", label="customers")
    plt.scatter(loc[0:1, 0], loc[0:1, 1], c="#f94144", marker="s", label="depot")
    plt.plot(route_xy[:, 0], route_xy[:, 1], c="#43aa8b", linewidth=1.5)
    for node in route:
        plt.text(loc[node, 0], loc[node, 1], str(node), fontsize=8)
    plt.title(f"instance {instance.index} distance={result['total_distance']:.1f}")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.allow_duplicate_openmp:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    instances = load_cvrp_instances(args.data, limit=args.limit)
    solutions = json.loads(Path(args.solutions).read_text(encoding="utf-8"))
    by_index = {int(item["instance"]): item for item in solutions}
    plt = import_pyplot()

    plot_dir = Path(args.plot_dir)
    count = 0
    for instance in instances:
        result = by_index.get(instance.index)
        if result is None:
            continue
        save_plot(plt, instance, result, plot_dir / f"instance_{instance.index}.png")
        count += 1

    print(f"[DONE] plots={plot_dir} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

