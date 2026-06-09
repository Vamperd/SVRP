#!/usr/bin/env python
"""Inspect universal TWCVRP split folders."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset import instance_group, list_instance_files, load_solomon_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a train/val/test split root.")
    parser.add_argument("--split_root", default="data_splits/universal_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_root = Path(args.split_root)
    manifest_path = split_root / "split_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sizes = sorted(
        {
            path.name
            for split_name in ("train", "val", "test")
            for path in (split_root / split_name).glob("*")
            if path.is_dir()
        },
        key=lambda value: int(value) if value.isdigit() else value,
    )
    print("size train val test customers train_groups val_groups test_groups")
    for size in sizes:
        rows = {}
        groups = {}
        customers = ""
        for split_name in ("train", "val", "test"):
            files = list_instance_files(split_root / split_name / size)
            rows[split_name] = len(files)
            groups[split_name] = dict(Counter(instance_group(path.stem) for path in files))
            if files and not customers:
                customers = str(load_solomon_file(files[0]).num_customers)
        print(
            f"{size} {rows['train']} {rows['val']} {rows['test']} {customers} "
            f"{groups['train']} {groups['val']} {groups['test']}"
        )

    if manifest:
        print(f"manifest={manifest_path}")
        print(f"seed={manifest.get('seed')} rules={manifest.get('rules')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

