#!/usr/bin/env python
"""Optionally pre-generate traffic perturbation summaries or full matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import load_instances
from traffic import sample_traffic_matrix, stable_seed, traffic_cache_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare synthetic traffic perturbations.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sizes", nargs="+", required=True)
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data_cache/traffic")
    parser.add_argument(
        "--store",
        default="summary",
        choices=["summary", "full"],
        help="summary writes metadata only; full writes every sampled matrix and can be very large.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest = {
        "traffic_sigma": args.traffic_sigma,
        "traffic_seed": args.traffic_seed,
        "mc_samples": args.mc_samples,
        "store": args.store,
        "items": [],
    }

    for size in args.sizes:
        instances = load_instances(
            data_root=args.data_root,
            size=size,
            source=args.source,
            pattern=args.pattern,
            limit=args.limit,
        )
        for instance in instances:
            item = {
                "size": str(size),
                "instance": instance.name,
                "customers": instance.num_customers,
                "source": instance.source,
                "matrix_shape": [instance.num_nodes, instance.num_nodes],
            }
            if args.store == "full":
                path = traffic_cache_path(output_dir, size=size, instance_name=instance.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                matrices = np.stack(
                    [
                        sample_traffic_matrix(
                            instance,
                            seed=stable_seed(instance.name, sample_idx, base_seed=args.traffic_seed),
                            traffic_sigma=args.traffic_sigma,
                        )
                        for sample_idx in range(args.mc_samples)
                    ],
                    axis=0,
                ).astype(np.float32)
                np.savez_compressed(path, matrices=matrices)
                item["path"] = str(path)
                item["bytes_estimate"] = int(matrices.nbytes)
            manifest["items"].append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "traffic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] manifest={manifest_path} items={len(manifest['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

