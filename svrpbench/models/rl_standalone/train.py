#!/usr/bin/env python
"""Train the standalone CVRP RL policy with plain PyTorch."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset import (
    evaluate_route,
    infer_num_customers,
    instances_to_tensors,
    load_cvrp_instances,
    make_batch,
    split_instances,
)
from model import PointerPolicy, actions_to_route, rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a standalone PyTorch RL policy for CVRP.")
    parser.add_argument("--data", required=True, help="Path to a single-depot CVRP .npz file.")
    parser.add_argument("--checkpoint", default="checkpoints/cvrp.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--capacity_penalty", type=float, default=100.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional instance limit.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_split", action="store_true", help="Use deterministic train/val/test split.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    *,
    model: PointerPolicy,
    args: argparse.Namespace,
    num_customers: int,
    map_size: float,
    history: list[dict],
    split: dict[str, list] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "num_customers": num_customers,
        "embed_dim": model.embed_dim,
        "map_size": map_size,
        "config": {
            "data": str(args.data),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "capacity_penalty": args.capacity_penalty,
            "seed": args.seed,
            "use_split": args.use_split,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
        },
        "split": {
            key: [int(instance.index) for instance in value]
            for key, value in (split or {}).items()
        },
        "history": history,
    }
    torch.save(payload, path)

    metadata_path = path.with_suffix(".json")
    metadata = dict(payload)
    metadata.pop("model_state_dict")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def evaluate_policy(
    model: PointerPolicy,
    instances,
    *,
    device: torch.device,
    capacity_penalty: float,
) -> dict:
    if not instances:
        return {"instances": 0}
    model.eval()
    rows = []
    with torch.no_grad():
        for instance in instances:
            batch = instances_to_tensors([instance], device=device)
            out = rollout(model, batch, decode="greedy", capacity_penalty=capacity_penalty)
            route = actions_to_route(out.actions[0])
            rows.append(evaluate_route(instance, route))
    return {
        "instances": len(rows),
        "avg_distance": float(np.mean([row["total_distance"] for row in rows])),
        "feasibility": float(np.mean([1.0 if row["feasible"] else 0.0 for row in rows])),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    instances = load_cvrp_instances(args.data, limit=args.limit)
    split = split_instances(
        instances,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    train_instances = split["train"] if args.use_split else instances
    val_instances = split["val"] if args.use_split else []
    num_customers = infer_num_customers(instances)
    model = PointerPolicy(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history: list[dict] = []

    print(
        "[START] "
        f"instances={len(instances)} train={len(train_instances)} val={len(val_instances)} "
        f"num_customers={num_customers} "
        f"device={device} epochs={args.epochs} batch_size={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch = make_batch(train_instances, args.batch_size, rng=rng, device=device)
        out = rollout(
            model,
            batch,
            decode="sample",
            capacity_penalty=args.capacity_penalty,
        )
        advantage = out.rewards - out.rewards.mean()
        loss = -(advantage.detach() * out.log_probs.sum(dim=1)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu().item()),
            "reward": float(out.rewards.mean().detach().cpu().item()),
            "distance": float(out.distances.mean().detach().cpu().item()),
            "capacity_violation": float(out.capacity_violation.mean().detach().cpu().item()),
        }
        if val_instances and (
            epoch == 1 or epoch == args.epochs or epoch % max(1, args.val_every) == 0
        ):
            val_summary = evaluate_policy(
                model,
                val_instances,
                device=device,
                capacity_penalty=args.capacity_penalty,
            )
            row["val_distance"] = val_summary.get("avg_distance", 0.0)
            row["val_feasibility"] = val_summary.get("feasibility", 0.0)
        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.print_every) == 0:
            print(
                f"epoch={epoch:04d} loss={row['loss']:.4f} "
                f"reward={row['reward']:.2f} distance={row['distance']:.2f} "
                f"cap_violation={row['capacity_violation']:.2f}"
                + (
                    f" val_distance={row['val_distance']:.2f} val_feas={row['val_feasibility']:.2f}"
                    if "val_distance" in row
                    else ""
                )
            )

    save_checkpoint(
        args.checkpoint,
        model=model,
        args=args,
        num_customers=num_customers,
        map_size=instances[0].map_size,
        history=history,
        split=split if args.use_split else None,
    )
    print(f"[DONE] checkpoint={args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
