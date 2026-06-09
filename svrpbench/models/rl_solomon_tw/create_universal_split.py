#!/usr/bin/env python
"""Create a multi-size Solomon split for universal TWCVRP training."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from dataset import instance_group, list_instance_files, load_solomon_file


PRIMARY_SIZES = {"100", "200", "400"}
ADAPTATION_SIZES = {"600", "800", "1000"}
DEFAULT_SIZES = ["100", "200", "400", "600", "800", "1000"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create universal_v1 Solomon train/val/test folders.")
    parser.add_argument("--source_root", required=True, help="Original solomon root.")
    parser.add_argument("--output_root", default="data_splits/universal_v1")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--large_train_per_group", type=int, default=2)
    parser.add_argument("--large_val_total", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_root} already exists and is not empty. "
            "Use --overwrite to copy over existing files."
        )

    manifest = {
        "name": "universal_v1",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": args.seed,
        "sizes": [str(size) for size in args.sizes],
        "rules": {
            "primary_sizes": sorted(PRIMARY_SIZES),
            "adaptation_sizes": sorted(ADAPTATION_SIZES),
            "primary_train_ratio": args.train_ratio,
            "primary_val_ratio": args.val_ratio,
            "large_train_per_group": args.large_train_per_group,
            "large_val_total": args.large_val_total,
        },
        "splits": {},
    }

    for size in [str(size) for size in args.sizes]:
        files = list_instance_files(source_root / size)
        if not files:
            raise FileNotFoundError(f"No Solomon files found for size={size} under {source_root}.")
        size_split = split_size(
            files,
            size=size,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            large_train_per_group=args.large_train_per_group,
            large_val_total=args.large_val_total,
        )
        manifest["splits"][size] = {}
        for split_name, split_files in size_split.items():
            dest_dir = output_root / split_name / size
            dest_dir.mkdir(parents=True, exist_ok=True)
            manifest["splits"][size][split_name] = []
            for src in split_files:
                dest = dest_dir / src.name
                shutil.copy2(src, dest)
                inst = load_solomon_file(dest)
                manifest["splits"][size][split_name].append(
                    {
                        "name": inst.name,
                        "group": inst.group,
                        "customers": inst.num_customers,
                        "source": str(src.resolve()),
                        "path": str(dest.relative_to(output_root)),
                    }
                )

    manifest_path = output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] split_root={output_root} manifest={manifest_path}")
    return 0


def split_size(
    files: list[Path],
    *,
    size: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    large_train_per_group: int,
    large_val_total: int,
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(instance_group(path.stem), []).append(path)
    for group_name in groups:
        rng = np.random.default_rng(stable_group_seed(seed, size, group_name))
        groups[group_name] = list(groups[group_name])
        rng.shuffle(groups[group_name])

    if size in ADAPTATION_SIZES:
        return split_large_adaptation(groups, large_train_per_group, large_val_total)
    return split_primary(groups, train_ratio, val_ratio)


def split_primary(
    groups: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[Path]]:
    total = sum(len(items) for items in groups.values())
    n_train, n_val = target_counts(total, train_ratio=train_ratio, val_ratio=val_ratio)
    group_sizes = {name: len(items) for name, items in groups.items()}
    train_counts = proportional_counts(group_sizes, n_train)
    remaining_sizes = {name: group_sizes[name] - train_counts[name] for name in groups}
    val_counts = proportional_counts(remaining_sizes, n_val)
    return materialize_split(groups, train_counts, val_counts)


def split_large_adaptation(
    groups: dict[str, list[Path]],
    train_per_group: int,
    val_total: int,
) -> dict[str, list[Path]]:
    train_counts = {name: min(train_per_group, len(items)) for name, items in groups.items()}
    remaining_sizes = {name: len(groups[name]) - train_counts[name] for name in groups}
    val_counts = proportional_counts(remaining_sizes, val_total)
    return materialize_split(groups, train_counts, val_counts)


def materialize_split(
    groups: dict[str, list[Path]],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
) -> dict[str, list[Path]]:
    split = {"train": [], "val": [], "test": []}
    for group_name in sorted(groups):
        items = groups[group_name]
        n_train = train_counts.get(group_name, 0)
        n_val = val_counts.get(group_name, 0)
        split["train"].extend(items[:n_train])
        split["val"].extend(items[n_train : n_train + n_val])
        split["test"].extend(items[n_train + n_val :])
    for split_name in split:
        split[split_name] = sorted(split[split_name], key=lambda path: path.name.lower())
    return split


def target_counts(n: int, *, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    return n_train, n_val


def proportional_counts(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0:
        return {name: 0 for name in group_sizes}
    total = sum(group_sizes.values())
    if total <= 0:
        return {name: 0 for name in group_sizes}
    raw = {name: target * size / total for name, size in group_sizes.items()}
    counts = {name: min(group_sizes[name], int(np.floor(value))) for name, value in raw.items()}
    remaining = target - sum(counts.values())
    order = sorted(
        group_sizes,
        key=lambda name: (raw[name] - np.floor(raw[name]), group_sizes[name], name),
        reverse=True,
    )
    for name in order:
        if remaining <= 0:
            break
        if counts[name] < group_sizes[name]:
            counts[name] += 1
            remaining -= 1
    return counts


def stable_group_seed(seed: int, size: str, group_name: str) -> int:
    return int(seed) + int(size) * 17 + sum(ord(char) for char in group_name)


if __name__ == "__main__":
    raise SystemExit(main())

