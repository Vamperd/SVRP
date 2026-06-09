#!/usr/bin/env python
"""Train RL4CO Attention/POMO models for SVRPBench reproduction.

Run this script manually from the conda ``svrp`` environment. See
``README_RL_REPRO.md`` for the exact commands.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

import utils as rl_utils


SCRIPT_DIR = Path(__file__).resolve().parent
SVRPBENCH_ROOT = SCRIPT_DIR.parents[1]
VRP_BENCH_DIR = SVRPBENCH_ROOT / "vrp_bench"
DEFAULT_CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints"
DEFAULT_MAP_SIZE = 1000
DEFAULT_MAX_TIME = 1440


def add_repo_paths() -> None:
    for path in (SVRPBENCH_ROOT, VRP_BENCH_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def import_training_deps():
    try:
        from lightning.pytorch.callbacks import ModelCheckpoint
        from rl4co.models import AttentionModelPolicy, POMO, REINFORCE
        from rl4co.utils.trainer import RL4COTrainer
    except (ImportError, ModuleNotFoundError) as exc:
        message = rl_utils.RL_DEPENDENCY_MESSAGE
        if "BoundedTensorSpec" in str(exc):
            message = (
                "RL4CO 0.4.0 is incompatible with the installed TorchRL version. "
                "Activate conda env 'svrp' and run: "
                "pip install --force-reinstall --no-deps torchrl==0.6.0 "
                "tensordict==0.6.0"
            )
        raise RuntimeError(message) from exc

    try:
        from rl4co.envs.routing import CVRPEnv
    except ImportError:
        try:
            from rl4co.envs import CVRPEnv
        except ImportError as exc:
            raise RuntimeError("Could not import CVRPEnv from the installed RL4CO.") from exc

    return {
        "AttentionModelPolicy": AttentionModelPolicy,
        "CVRPEnv": CVRPEnv,
        "ModelCheckpoint": ModelCheckpoint,
        "POMO": POMO,
        "REINFORCE": REINFORCE,
        "RL4COTrainer": RL4COTrainer,
    }


def patch_rl4co_tensordict_compat() -> None:
    """Patch RL4CO 0.4.0 dataset helpers for newer TensorDict constructors.

    RL4CO 0.4.0 passes ``_run_checks=False`` into ``TensorDict(...)``. Recent
    TensorDict versions no longer accept that private keyword and interpret it
    as a data kwarg, which raises:
    ``ValueError: Either a dictionary or a sequence of kwargs must be provided``.
    """
    try:
        import torch
        import rl4co.data.dataset as dataset_module
        from tensordict.tensordict import TensorDict
    except Exception:
        return

    def collate_fn(batch):
        return TensorDict(
            {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
            batch_size=torch.Size([len(batch)]),
        )

    def fast_getitems(self, index):
        return TensorDict(
            {key: item[index] for key, item in self.data.items()},
            batch_size=torch.Size([len(index)]),
        )

    dataset_module.TensorDictDataset.collate_fn = staticmethod(collate_fn)
    dataset_module.TensorDictDatasetFastGeneration.__getitems__ = fast_getitems


def import_mtvrp_deps():
    try:
        from rl4co.envs.routing.mtvrp.env import MTVRPEnv
        from rl4co.envs.routing.mtvrp.generator import MTVRPGenerator
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The installed RL4CO package does not expose "
            "rl4co.envs.routing.mtvrp. CVRP training can still run, but TWVRP "
            "training needs an RL4CO build/version that includes MTVRP/VRPTW."
        ) from exc
    return MTVRPEnv, MTVRPGenerator


def build_cvrp_env(CVRPEnv, num_loc: int):
    try:
        return CVRPEnv(generator_params={"num_loc": num_loc})
    except TypeError:
        return CVRPEnv(num_loc=num_loc)


def build_twvrp_env(MTVRPEnv, MTVRPGenerator, num_loc: int, max_time: int, map_size: int):
    add_repo_paths()
    from time_windows_generator import sample_time_window
    from travel_time_generator import get_distances, sample_travel_time

    def demand_sampler(inst, idx):
        return int(np.asarray(inst["demands"])[idx])

    def appear_time_sampler(inst, idx):
        appear_times = inst.get("appear_time", inst.get("appear_times"))
        return float(np.asarray(appear_times)[idx])

    def travel_time_sampler(inst, i, j):
        distances = get_distances(inst["map_instance"])
        return sample_travel_time(i, j, distances, random.randint(0, max_time))

    def time_window_sampler(inst, idx):
        appear_time = appear_time_sampler(inst, idx)
        return sample_time_window(random.randint(0, 1), appear_time)

    generator = MTVRPGenerator(
        num_loc=num_loc,
        variant_preset="vrptw",
        max_time=max_time,
        map_size=(map_size, map_size),
        num_cities=max(1, num_loc // 50),
        num_depots=1,
        demand_sampler=demand_sampler,
        appear_time_sampler=appear_time_sampler,
        travel_time_sampler=travel_time_sampler,
        time_window_sampler=time_window_sampler,
    )
    return MTVRPEnv(generator)


def resolve_batch_size(args) -> int:
    if args.batch_size is not None:
        return args.batch_size
    if args.smoke:
        return 8
    defaults = {10: 512, 20: 256, 50: 128, 100: 64, 200: 64, 500: 32}
    return defaults.get(args.num_loc, 32)


def resolve_train_size(args) -> int:
    if args.train_data_size is not None:
        return args.train_data_size
    if args.smoke:
        return 32
    return 100_000 if args.variant == "cvrp" else 1_000_000


def resolve_val_size(args) -> int:
    if args.val_data_size is not None:
        return args.val_data_size
    if args.smoke:
        return 8
    return 1_000


def resolve_epochs(args) -> int:
    if args.max_epochs is not None:
        return args.max_epochs
    return 1 if args.smoke else 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RL4CO baselines for SVRPBench.")
    parser.add_argument("--variant", choices=["cvrp", "twvrp"], required=True)
    parser.add_argument("--algo", choices=["attention", "pomo"], required=True)
    parser.add_argument("--num_loc", type=int, required=True, help="Number of customers.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_data_size", type=int, default=None)
    parser.add_argument("--val_data_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--checkpoint_root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument("--map_size", type=int, default=DEFAULT_MAP_SIZE)
    parser.add_argument("--monitor", default="val/reward")
    parser.add_argument("--monitor_mode", default="max")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny defaults for a manual environment smoke test.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deps = import_training_deps()
    patch_rl4co_tensordict_compat()

    if args.variant == "cvrp":
        env = build_cvrp_env(deps["CVRPEnv"], args.num_loc)
    else:
        MTVRPEnv, MTVRPGenerator = import_mtvrp_deps()
        env = build_twvrp_env(
            MTVRPEnv,
            MTVRPGenerator,
            args.num_loc,
            args.max_time,
            args.map_size,
        )

    policy = deps["AttentionModelPolicy"](
        env_name=env.name,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
    )

    model_cls = deps["REINFORCE"] if args.algo == "attention" else deps["POMO"]
    baseline = "rollout" if args.algo == "attention" else "shared"
    batch_size = resolve_batch_size(args)
    train_size = resolve_train_size(args)
    val_size = resolve_val_size(args)
    max_epochs = resolve_epochs(args)

    model = model_cls(
        env,
        policy,
        baseline=baseline,
        batch_size=batch_size,
        train_data_size=train_size,
        val_data_size=val_size,
        optimizer_kwargs={"lr": args.lr},
    )

    checkpoint_dir = Path(args.checkpoint_root) / args.variant / f"{args.algo}_{args.num_loc}"
    checkpoint_callback = deps["ModelCheckpoint"](
        dirpath=str(checkpoint_dir),
        filename="epoch_{epoch:03d}",
        save_top_k=1,
        save_last=True,
        monitor=args.monitor,
        mode=args.monitor_mode,
    )

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback],
    }
    if args.devices is not None:
        trainer_kwargs["devices"] = args.devices
    trainer = deps["RL4COTrainer"](**trainer_kwargs)

    print(
        "[START] "
        f"variant={args.variant} algo={args.algo} num_loc={args.num_loc} "
        f"batch_size={batch_size} train_data_size={train_size} "
        f"val_data_size={val_size} max_epochs={max_epochs} "
        f"checkpoint_dir={checkpoint_dir}"
    )
    trainer.fit(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
