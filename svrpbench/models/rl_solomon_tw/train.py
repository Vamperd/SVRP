#!/usr/bin/env python
"""Train a standalone PyTorch REINFORCE policy for Solomon TWCVRP."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset import load_instances, sample_batch, split_instances, instances_to_tensors
from decoder import annotate_objective_metrics, decode_order, score_order
from evaluator import aggregate_metrics, evaluate_routes
from heuristic import nearest_order
from model import TWPointerPolicy, actions_to_order, order_log_probs, rollout
from traffic import planning_matrix, sample_traffic_matrix, stable_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train standalone RL for Solomon TWCVRP.")
    parser.add_argument("--data_root", default=None, help="Root containing solomon/<size> folders.")
    parser.add_argument("--split_root", default=None, help="Root with train/val/test/<size> split folders.")
    parser.add_argument("--input", default=None, help="Optional single Solomon .txt/.TXT or .npz file.")
    parser.add_argument("--size", default=None, help="Customer count folder, e.g. 100.")
    parser.add_argument("--sizes", nargs="+", default=None, help="Customer sizes for mixed training.")
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--size_sampling", default="balanced", choices=["balanced", "proportional"])
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--mode", default="static", choices=["static", "traffic"])
    parser.add_argument("--objective", default="feasibility", choices=["feasibility", "robust_cvr"])
    parser.add_argument(
        "--decoder",
        default="strict_insert",
        choices=["strict_insert", "deadline_aware_insert", "greedy_split"],
    )
    parser.add_argument("--insert_top_k", type=int, default=30)
    parser.add_argument("--post_opt", default="none", choices=["none", "time_window_repair"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--checkpoint", default="checkpoints/tw_policy.pt")
    parser.add_argument("--best_checkpoint", default=None)
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Optional checkpoint used to initialize model weights before fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_buffer", type=float, default=0.50)
    parser.add_argument("--traffic_profile", default="additive", choices=["additive", "proportional"])
    parser.add_argument("--traffic_strength", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--late_penalty", type=float, default=10.0)
    parser.add_argument("--time_window_penalty", type=float, default=1000.0)
    parser.add_argument("--capacity_penalty", type=float, default=1000.0)
    parser.add_argument("--missing_penalty", type=float, default=500.0)
    parser.add_argument("--duplicate_penalty", type=float, default=500.0)
    parser.add_argument("--vehicle_penalty", type=float, default=10000.0)
    parser.add_argument("--route_count_penalty", type=float, default=200.0)
    parser.add_argument("--route_overuse_penalty", type=float, default=0.0)
    parser.add_argument("--target_customers_per_route", type=float, default=9.0)
    parser.add_argument("--feasible_bonus", type=float, default=50000.0)
    parser.add_argument("--infeasible_penalty", type=float, default=50000.0)
    parser.add_argument("--imitation_epochs", type=int, default=0)
    parser.add_argument("--imitation_weight", type=float, default=1.0)
    parser.add_argument(
        "--baseline_momentum",
        type=float,
        default=0.90,
        help="EMA momentum for the per-size reward baseline used when batch_size=1.",
    )
    parser.add_argument(
        "--advantage_clip",
        type=float,
        default=10.0,
        help="Clip normalized REINFORCE advantages to this absolute value; set <=0 to disable.",
    )
    parser.add_argument(
        "--robust_val_samples",
        type=int,
        default=1,
        help="Number of traffic samples per validation route for robust_cvr validation.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use at most 4 files unless --limit is set.")
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


def penalty_config(args: argparse.Namespace) -> dict[str, float]:
    return {
        "objective": args.objective,
        "late": args.late_penalty,
        "time_window": args.time_window_penalty,
        "capacity": args.capacity_penalty,
        "missing": args.missing_penalty,
        "duplicate": args.duplicate_penalty,
        "vehicle": args.vehicle_penalty,
        "route_count": args.route_count_penalty,
        "route_overuse": args.route_overuse_penalty,
        "target_customers_per_route": args.target_customers_per_route,
        "feasible_bonus": args.feasible_bonus,
        "infeasible": args.infeasible_penalty,
    }


def load_data(args: argparse.Namespace):
    limit = args.limit
    if args.smoke and limit is None:
        limit = 4
    return load_instances(
        data_root=args.data_root,
        input_path=args.input,
        size=args.size,
        source=args.source,
        pattern=args.pattern,
        limit=limit,
    )


def discover_split_sizes(split_root: str | Path) -> list[str]:
    train_root = Path(split_root) / "train"
    sizes = [path.name for path in train_root.glob("*") if path.is_dir()]
    return sorted(sizes, key=lambda value: int(value) if value.isdigit() else value)


def load_split_manifest(split_root: str | Path) -> dict:
    manifest_path = Path(split_root) / "split_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_mixed_data(args: argparse.Namespace):
    if not args.split_root:
        instances = load_data(args)
        split = split_instances(
            instances,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        return {
            "instances": instances,
            "train_by_size": {str(instances[0].num_customers): split["train"] or instances},
            "val_instances": split["val"],
            "split": split,
            "manifest": {},
            "supported_sizes": [str(instances[0].num_customers)],
        }

    split_root = Path(args.split_root)
    sizes = [str(size) for size in (args.sizes or discover_split_sizes(split_root))]
    limit = args.limit
    if args.smoke and limit is None:
        limit = 2
    train_by_size: dict[str, list] = {}
    val_instances = []
    all_instances = []
    split = {"train": [], "val": [], "test": []}
    for size in sizes:
        train_items = load_instances(
            data_root=split_root / "train",
            size=size,
            source="solomon",
            pattern=args.pattern,
            limit=limit,
        )
        val_items = load_instances(
            data_root=split_root / "val",
            size=size,
            source="solomon",
            pattern=args.pattern,
            limit=limit,
        )
        if not train_items:
            raise ValueError(f"No train instances found for size={size} in {split_root}.")
        train_by_size[size] = train_items
        val_instances.extend(val_items)
        all_instances.extend(train_items)
        all_instances.extend(val_items)
        split["train"].extend(train_items)
        split["val"].extend(val_items)
    return {
        "instances": all_instances,
        "train_by_size": train_by_size,
        "val_instances": val_instances,
        "split": split,
        "manifest": load_split_manifest(split_root),
        "supported_sizes": sizes,
    }


def matrix_for_training(instance, args: argparse.Namespace, *, epoch: int, row: int) -> np.ndarray:
    if args.mode == "static":
        return planning_matrix(instance, mode="static")
    return sample_traffic_matrix(
        instance,
        seed=stable_seed(instance.name, epoch, row, base_seed=args.traffic_seed),
        traffic_sigma=args.traffic_sigma,
        traffic_profile=args.traffic_profile,
        traffic_strength=args.traffic_strength,
    )


def epoch_size_schedule(
    train_by_size: dict[str, list],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[str]:
    sizes = list(train_by_size)
    if not sizes:
        raise ValueError("No training size buckets available.")
    default_steps = len(sizes) if args.split_root else 1
    steps = args.steps_per_epoch or default_steps
    if args.size_sampling == "proportional":
        weights = np.asarray([len(train_by_size[size]) for size in sizes], dtype=np.float64)
        weights = weights / np.maximum(weights.sum(), 1.0)
        return [str(size) for size in rng.choice(sizes, size=steps, replace=True, p=weights)]

    schedule: list[str] = []
    while len(schedule) < steps:
        shuffled = list(rng.permutation(sizes))
        schedule.extend(str(size) for size in shuffled)
    return schedule[:steps]


def evaluate_policy(policy, instances, args: argparse.Namespace, device: torch.device) -> dict:
    if not instances:
        return {"instances": 0}
    if args.val_limit is not None and args.val_limit > 0:
        instances = instances[: args.val_limit]
    rows = []
    penalties = penalty_config(args)
    policy.eval()
    with torch.no_grad():
        for instance in instances:
            batch = instances_to_tensors([instance], device=device)
            out = rollout(policy, batch, decode="greedy")
            order = actions_to_order(out.actions[0])
            matrix = planning_matrix(
                instance,
                mode=args.mode,
                traffic_sigma=args.traffic_sigma,
                traffic_buffer=args.traffic_buffer,
                traffic_profile=args.traffic_profile,
                traffic_strength=args.traffic_strength,
            )
            if args.mode == "traffic" and args.robust_val_samples > 1:
                routes = decode_order(
                    instance,
                    order,
                    matrix,
                    decoder=args.decoder,
                    insert_top_k=args.insert_top_k,
                    post_opt=args.post_opt,
                )
                for sample_idx in range(max(1, args.robust_val_samples)):
                    sample_matrix = sample_traffic_matrix(
                        instance,
                        seed=stable_seed(
                            instance.name,
                            "val",
                            sample_idx,
                            base_seed=args.traffic_seed,
                        ),
                        traffic_sigma=args.traffic_sigma,
                        traffic_profile=args.traffic_profile,
                        traffic_strength=args.traffic_strength,
                    )
                    metrics = evaluate_routes(instance, routes, sample_matrix)
                    annotate_objective_metrics(instance, metrics, penalties)
                    rows.append(metrics)
            else:
                _, metrics = score_order(
                    instance,
                    order,
                    matrix,
                    decoder=args.decoder,
                    insert_top_k=args.insert_top_k,
                    post_opt=args.post_opt,
                    penalties=penalties,
                )
                rows.append(metrics)
    return aggregate_metrics(rows)


def best_key(row: dict, args: argparse.Namespace) -> tuple[float, ...]:
    if args.objective == "robust_cvr":
        return (
            -float(row.get("val_cvr", float("inf"))),
            float(row.get("val_feasibility", 0.0)),
            -float(row.get("val_route_overuse", float("inf"))),
            -float(row.get("val_total_cost", float("inf"))),
        )
    return (
        float(row.get("val_feasibility", 0.0)),
        -float(row.get("val_cvr", float("inf"))),
        -float(row.get("val_total_cost", float("inf"))),
    )


def reinforce_advantage(
    reward_tensor: torch.Tensor,
    *,
    size: str,
    baseline_state: dict[str, dict[str, float | None]],
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return a normalized REINFORCE advantage, including batch_size=1 support."""
    if reward_tensor.numel() > 1:
        advantage = (reward_tensor - reward_tensor.mean()) / torch.clamp(
            reward_tensor.std(unbiased=False),
            min=1.0,
        )
    else:
        state = baseline_state.setdefault(str(size), {"mean": None, "var": 1.0})
        reward_value = float(reward_tensor.detach().cpu().item())
        if state["mean"] is None:
            state["mean"] = reward_value
            state["var"] = 1.0
            advantage = torch.zeros_like(reward_tensor)
        else:
            baseline = float(state["mean"])
            variance = max(float(state["var"] or 1.0), 1.0)
            advantage = (reward_tensor - baseline) / (variance**0.5)
            delta = reward_value - baseline
            momentum = min(max(float(args.baseline_momentum), 0.0), 0.999)
            state["mean"] = momentum * baseline + (1.0 - momentum) * reward_value
            state["var"] = momentum * variance + (1.0 - momentum) * (delta * delta)

    if args.advantage_clip and args.advantage_clip > 0:
        advantage = torch.clamp(advantage, -float(args.advantage_clip), float(args.advantage_clip))
    return advantage


def expert_orders_tensor(
    batch_instances,
    args: argparse.Namespace,
    device: torch.device,
    expert_cache: dict[str, list[int]],
) -> torch.Tensor:
    orders = []
    for instance in batch_instances:
        cache_key = (
            f"{instance.source}|{args.mode}|{args.traffic_sigma}|{args.traffic_buffer}|"
            f"{args.traffic_profile}|{args.traffic_strength}"
        )
        if cache_key not in expert_cache:
            matrix = planning_matrix(
                instance,
                mode=args.mode,
                traffic_sigma=args.traffic_sigma,
                traffic_buffer=args.traffic_buffer,
                traffic_profile=args.traffic_profile,
                traffic_strength=args.traffic_strength,
            )
            expert_cache[cache_key] = nearest_order(instance, matrix)
        orders.append(expert_cache[cache_key])
    return torch.tensor(orders, dtype=torch.long, device=device)


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[TWPointerPolicy, dict | None]:
    """Create a policy, optionally initialized from a trusted local checkpoint."""
    if not args.init_checkpoint:
        return TWPointerPolicy(embed_dim=args.embed_dim).to(device), None

    checkpoint_path = Path(args.init_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init_checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    context_weight = state.get("context_encoder.0.weight")
    if "context_dim" in checkpoint:
        context_dim = int(checkpoint["context_dim"])
    elif context_weight is not None:
        context_dim = int(context_weight.shape[1])
    else:
        context_dim = 9
    dynamic_weight = state.get("dynamic_encoder.0.weight")
    if "dynamic_feature_dim" in checkpoint:
        dynamic_feature_dim = int(checkpoint["dynamic_feature_dim"])
    else:
        dynamic_feature_dim = int(dynamic_weight.shape[1] if dynamic_weight is not None else 0)
    model = TWPointerPolicy(
        embed_dim=int(checkpoint.get("embed_dim", args.embed_dim)),
        feature_dim=int(checkpoint.get("feature_dim", 6)),
        context_dim=context_dim,
        dynamic_feature_dim=dynamic_feature_dim,
        model_version=checkpoint.get("model_version", "tw_pointer_legacy_v1"),
    ).to(device)
    model.load_state_dict(state)
    print(f"[INIT] loaded checkpoint={checkpoint_path}")
    return model, checkpoint


def save_checkpoint(
    path: str | Path,
    *,
    model: TWPointerPolicy,
    args: argparse.Namespace,
    instances,
    split,
    history: list[dict],
    best: dict | None,
    supported_sizes: list[str],
    split_manifest: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    customer_sizes = sorted({int(instance.num_customers) for instance in instances})
    payload = {
        "model_state_dict": model.state_dict(),
        "embed_dim": model.embed_dim,
        "feature_dim": model.feature_dim,
        "context_dim": model.context_dim,
        "dynamic_feature_dim": model.dynamic_feature_dim,
        "model_version": model.model_version,
        "supported_sizes": [str(size) for size in supported_sizes],
        "num_customers": customer_sizes[0] if len(customer_sizes) == 1 else None,
        "mode": args.mode,
        "config": vars(args),
        "split": {key: [instance.name for instance in value] for key, value in split.items()},
        "split_indices": {
            key: [int(instance.index) for instance in value]
            for key, value in split.items()
        },
        "split_manifest": split_manifest or {},
        "history": history,
        "best": best,
    }
    torch.save(payload, path)
    metadata = dict(payload)
    metadata.pop("model_state_dict")
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    data = load_mixed_data(args)
    instances = data["instances"]
    if not instances:
        raise ValueError("No training instances loaded.")
    if not args.split_root and len({instance.num_customers for instance in instances}) != 1:
        raise ValueError("Train one customer scale at a time.")

    split = data["split"]
    train_by_size = data["train_by_size"]
    val_instances = data["val_instances"]
    supported_sizes = data["supported_sizes"]
    model, _ = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    expert_cache: dict[str, list[int]] = {}
    baseline_state: dict[str, dict[str, float | None]] = {}
    history: list[dict] = []
    best: dict | None = None
    best_checkpoint = args.best_checkpoint or str(Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_best.pt"))

    print(
        "[START] "
        f"mode={args.mode} instances={len(instances)} "
        f"train={sum(len(value) for value in train_by_size.values())} "
        f"val={len(val_instances)} sizes={supported_sizes} "
        f"device={device} epochs={args.epochs} batch_size={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_rewards = []
        epoch_metrics = []
        imitation_phase = epoch <= max(0, args.imitation_epochs)
        for step_idx, size in enumerate(epoch_size_schedule(train_by_size, args, rng)):
            batch_instances, batch = sample_batch(
                train_by_size[size],
                args.batch_size,
                rng=rng,
                device=device,
            )
            train_metrics = []
            if imitation_phase:
                expert_orders = expert_orders_tensor(batch_instances, args, device, expert_cache)
                log_probs = order_log_probs(model, batch, expert_orders)
                loss = -args.imitation_weight * log_probs.mean()
                rewards = []
                for row, (instance, order_tensor) in enumerate(zip(batch_instances, expert_orders)):
                    order = [int(node) for node in order_tensor.detach().cpu().tolist()]
                    matrix = matrix_for_training(
                        instance,
                        args,
                        epoch=epoch,
                        row=row + step_idx * args.batch_size,
                    )
                    reward, metrics = score_order(
                        instance,
                        order,
                        matrix,
                        decoder=args.decoder,
                        insert_top_k=args.insert_top_k,
                        post_opt=args.post_opt,
                        penalties=penalty_config(args),
                    )
                    rewards.append(reward)
                    train_metrics.append(metrics)
                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            else:
                out = rollout(model, batch, decode="sample")
                rewards = []
                for row, (instance, actions) in enumerate(zip(batch_instances, out.actions)):
                    order = actions_to_order(actions)
                    matrix = matrix_for_training(
                        instance,
                        args,
                        epoch=epoch,
                        row=row + step_idx * args.batch_size,
                    )
                    reward, metrics = score_order(
                        instance,
                        order,
                        matrix,
                        decoder=args.decoder,
                        insert_top_k=args.insert_top_k,
                        post_opt=args.post_opt,
                        penalties=penalty_config(args),
                    )
                    rewards.append(reward)
                    train_metrics.append(metrics)

                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                advantage = reinforce_advantage(
                    reward_tensor,
                    size=str(size),
                    baseline_state=baseline_state,
                    args=args,
                )
                loss = -(advantage.detach() * out.log_probs.sum(dim=1)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_rewards.append(float(reward_tensor.mean().detach().cpu().item()))
            epoch_metrics.extend(train_metrics)

        train_summary = aggregate_metrics(epoch_metrics)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "reward": float(np.mean(epoch_rewards)),
            "train_total_cost": train_summary.get("total_cost", 0.0),
            "train_cvr": train_summary.get("cvr", 0.0),
            "train_feasibility": train_summary.get("feasibility", 0.0),
            "train_vehicles_excess": train_summary.get("vehicles_excess", 0.0),
            "train_route_count": train_summary.get("route_count", 0.0),
            "train_route_overuse": train_summary.get("route_overuse", 0.0),
            "train_cost_per_customer": train_summary.get("cost_per_customer", 0.0),
            "train_late_per_customer": train_summary.get("late_per_customer", 0.0),
            "train_time_window_violations": train_summary.get("time_window_violations", 0.0),
            "train_capacity_violations": train_summary.get("capacity_violations", 0.0),
            "phase": "imitation" if imitation_phase else "reinforce",
            "steps": len(epoch_losses),
        }

        should_validate = bool(val_instances) and (
            epoch == 1 or epoch == args.epochs or epoch % max(1, args.val_every) == 0
        )
        if should_validate:
            val_summary = evaluate_policy(model, val_instances, args, device)
            row.update(
                {
                    "val_total_cost": val_summary.get("total_cost", 0.0),
                    "val_cvr": val_summary.get("cvr", 0.0),
                    "val_feasibility": val_summary.get("feasibility", 0.0),
                    "val_vehicles_excess": val_summary.get("vehicles_excess", 0.0),
                    "val_route_count": val_summary.get("route_count", 0.0),
                    "val_route_overuse": val_summary.get("route_overuse", 0.0),
                    "val_cost_per_customer": val_summary.get("cost_per_customer", 0.0),
                    "val_late_per_customer": val_summary.get("late_per_customer", 0.0),
                    "val_time_window_violations": val_summary.get("time_window_violations", 0.0),
                    "val_capacity_violations": val_summary.get("capacity_violations", 0.0),
                }
            )
            if best is None or best_key(row, args) > best_key(best, args):
                best = dict(row)
                save_checkpoint(
                    best_checkpoint,
                    model=model,
                    args=args,
                    instances=instances,
                    split=split,
                    history=history + [row],
                    best=best,
                    supported_sizes=supported_sizes,
                    split_manifest=data["manifest"],
                )

        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.print_every) == 0:
            message = (
                f"epoch={epoch:04d} loss={row['loss']:.4f} reward={row['reward']:.2f} "
                f"phase={row['phase']} train_cost={row['train_total_cost']:.2f} "
                f"train_cvr={row['train_cvr']:.2f} train_feas={row['train_feasibility']:.2f} "
                f"train_veh_excess={row['train_vehicles_excess']:.2f} "
                f"train_overuse={row['train_route_overuse']:.2f}"
            )
            if "val_total_cost" in row:
                message += (
                    f" val_cost={row['val_total_cost']:.2f} "
                    f"val_feas={row['val_feasibility']:.2f} "
                    f"val_cvr={row['val_cvr']:.2f} "
                    f"val_veh_excess={row['val_vehicles_excess']:.2f} "
                    f"val_overuse={row['val_route_overuse']:.2f} "
                    f"val_cost_pc={row['val_cost_per_customer']:.2f}"
                )
            print(message)

    save_checkpoint(
        args.checkpoint,
        model=model,
        args=args,
        instances=instances,
        split=split,
        history=history,
        best=best,
        supported_sizes=supported_sizes,
        split_manifest=data["manifest"],
    )
    print(f"[DONE] checkpoint={args.checkpoint} best_checkpoint={best_checkpoint if best else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
