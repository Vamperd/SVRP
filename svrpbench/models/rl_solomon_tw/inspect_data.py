#!/usr/bin/env python
"""Inspect Solomon or benchmark TWCVRP data and print deterministic splits."""
from __future__ import annotations

import argparse
from collections import Counter

from dataset import load_instances, split_instances


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect TWCVRP data files.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sizes", nargs="+", required=True)
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("size files customers vehicles capacity horizon groups train val test")
    for size in args.sizes:
        instances = load_instances(
            data_root=args.data_root,
            size=size,
            source=args.source,
            pattern=args.pattern,
            limit=args.limit,
        )
        split = split_instances(
            instances,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        groups = Counter(instance.group for instance in instances)
        first = instances[0]
        print(
            f"{size} {len(instances)} {first.num_customers} {first.vehicle_count} "
            f"{first.vehicle_capacity:g} {first.horizon:g} {dict(groups)} "
            f"{len(split['train'])} {len(split['val'])} {len(split['test'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

