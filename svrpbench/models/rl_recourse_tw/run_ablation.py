#!/usr/bin/env python
"""Run ablation experiments on static-guided dynamic recourse components.

Generates four dataset variants and optionally trains/evaluates each:

  1. full:       benchmark_routes + time_window_drift + traffic
  2. no_drift:   benchmark_routes + traffic (no drift)
  3. no_routes:  time_window_drift + traffic (no benchmark guidance)
  4. baseline:   traffic only (equivalent to old pure-online recourse)

Each variant writes its dataset to a subdirectory, which can then be
fed to train_sg.py for training and evaluate.py for testing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATION_VARIANTS = {
    "full":       {"no_drift": False, "no_routes": False},
    "no_drift":   {"no_drift": True,  "no_routes": False},
    "no_routes":  {"no_drift": False, "no_routes": True},
    "baseline":   {"no_drift": True,  "no_routes": True},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation study for static-guided recourse.")
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--static_checkpoint", required=True)
    parser.add_argument("--sizes", nargs="+", default=["400", "600", "800", "1000"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_routes", type=int, default=20)
    parser.add_argument("--snapshots_per_route", type=int, default=10)
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_profile", default="additive")
    parser.add_argument("--traffic_strength", type=float, default=1.0)
    parser.add_argument("--drift_sigma", type=float, default=0.15)
    parser.add_argument("--output_base", default="data_cache/ablation")
    parser.add_argument("--variants", nargs="+", default=["full", "no_drift", "no_routes", "baseline"])
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    return parser


def run_step(cmd: list[str], description: str) -> int:
    print(f"\n{'=' * 60}")
    print(f"[STEP] {description}")
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] {description}")
        return result.returncode
    print(f"[OK] {description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sizes_arg = list(args.sizes)

    for variant_name in args.variants:
        if variant_name not in ABLATION_VARIANTS:
            print(f"[SKIP] Unknown variant: {variant_name}")
            continue

        config = ABLATION_VARIANTS[variant_name]
        variant_dir = Path(args.output_base) / variant_name

        step1_cmd = [
            args.python, "-m", "rl_recourse_tw.generate_static_routes",
            "--split_root", args.split_root,
            "--sizes", *sizes_arg,
            "--split", args.split,
            "--checkpoint", args.static_checkpoint,
            "--num_routes", str(args.num_routes),
            "--output_dir", str(variant_dir / "static_routes"),
        ]

        step2_cmd = [
            args.python, "-m", "rl_recourse_tw.generate_dynamic_dataset",
            "--routes_dir", str(variant_dir / "static_routes"),
            "--split_root", args.split_root,
            "--sizes", *sizes_arg,
            "--split", args.split,
            "--snapshots_per_route", str(args.snapshots_per_route),
            "--traffic_sigma", str(args.traffic_sigma),
            "--traffic_profile", args.traffic_profile,
            "--traffic_strength", str(args.traffic_strength),
            "--drift_sigma", str(args.drift_sigma),
            "--output_dir", str(variant_dir / "dynamic_dataset"),
        ]
        if config["no_drift"]:
            step2_cmd.append("--no_drift")

        print(f"\n{'#' * 60}")
        print(f"# ABLATION VARIANT: {variant_name}")
        print(f"#  no_drift={config['no_drift']} no_routes={config['no_routes']}")
        print(f"{'#' * 60}")

        if config["no_routes"]:
            print(f"[SKIP] step1 (no benchmark routes for {variant_name})")
        else:
            rc = run_step(step1_cmd, f"{variant_name}: generate_static_routes")
            if rc != 0:
                return rc

        rc = run_step(step2_cmd, f"{variant_name}: generate_dynamic_dataset")
        if rc != 0:
            return rc

    print(f"\n[DONE] All ablation datasets generated under {args.output_base}")
    print(f"  Train each with train_sg.py --dataset_dir <variant>/dynamic_dataset")
    print(f"  Evaluate with evaluate.py --methods sg_recourse --checkpoint <checkpoint>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
