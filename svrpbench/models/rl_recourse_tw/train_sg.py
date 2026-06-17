#!/usr/bin/env python
"""Train an event-driven recourse policy via static-guided behavioral cloning.

Loads pre-generated dynamic dataset snapshots (traffic matrix + drifted due
times + static-planner expert order), performs imitation learning against the
static expert, followed by lightweight REINFORCE with CVaR.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ensure_dir, SOLOMON_TW_DIR  # noqa: F401
from dataset import load_instances
from env import EventDrivenTWEnv
from policy import EventDrivenSTPolicy
from rollout import rollout_policy
from traffic import planning_matrix, sample_traffic_matrix, stable_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static-guided BC + REINFORCE for online recourse.")
    parser.add_argument("--dataset_dir", required=True, help="Directory with *_snapshots.npz files.")
    parser.add_argument("--split_root", required=True, help="Data split root for loading instances.")
    parser.add_argument("--sizes", nargs="+", required=True, help="Sizes to train on.")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--traffic_time_scale", default="depot_day", choices=["raw", "depot_day"])
    parser.add_argument("--mask_late", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc_epochs", type=int, default=50, help="Static-guided BC epochs.")
    parser.add_argument("--rl_epochs", type=int, default=15, help="Lightweight REINFORCE epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--risk_objective", default="cvar", choices=["mean", "cvar"])
    parser.add_argument("--cvar_alpha", type=float, default=0.20)
    parser.add_argument("--mc_samples_train", type=int, default=1)
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_profile", default="additive", choices=["additive", "proportional"])
    parser.add_argument("--traffic_strength", type=float, default=1.0)
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_limit", type=int, default=8)
    parser.add_argument("--checkpoint", default="checkpoints/recourse_sg.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--late_penalty", type=float, default=300.0)
    parser.add_argument("--time_window_penalty", type=float, default=10000.0)
    parser.add_argument("--capacity_penalty", type=float, default=5000.0)
    parser.add_argument("--missing_penalty", type=float, default=5000.0)
    parser.add_argument("--duplicate_penalty", type=float, default=5000.0)
    parser.add_argument("--vehicle_penalty", type=float, default=10000.0)
    parser.add_argument("--forced_penalty", type=float, default=1000.0)
    parser.add_argument("--cost_weight", type=float, default=1.0)
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


def load_sg_dataset(
    dataset_dir: str | Path,
    sizes: list[str],
    instance_map: dict[str, object],
) -> list[dict]:
    snapshots = []
    for size in sizes:
        size_dir = Path(dataset_dir) / size
        if not size_dir.exists():
            continue
        for npz_path in sorted(size_dir.glob("*_snapshots.npz")):
            data = np.load(npz_path, allow_pickle=False)
            inst_name = str(data["instance_name"])
            if inst_name not in instance_map:
                continue
            instance = instance_map[inst_name]
            num_snaps = int(data["num_snapshots"])
            for i in range(num_snaps):
                order = data["expert_orders"][i]
                order = [int(x) for x in order if int(x) >= 0]
                if not order:
                    continue
                snapshots.append({
                    "instance": instance,
                    "traffic_matrix": np.asarray(data["traffic_matrices"][i], dtype=np.float32),
                    "drifted_due_times": np.asarray(data["drifted_due_times"][i], dtype=np.float32),
                    "expert_order": order,
                })
    return snapshots


def load_instances_for_sizes(
    split_root: str | Path,
    sizes: list[str],
    split: str,
    source: str,
) -> dict[str, object]:
    instance_map = {}
    for size in sizes:
        size_dir = Path(split_root) / split / size
        if not size_dir.exists():
            continue
        instances = load_instances(data_root=size_dir, size=size, source=source)
        for inst in instances:
            instance_map[inst.name] = inst
    return instance_map


def reward_from_metrics(metrics: dict, args: argparse.Namespace) -> float:
    n = max(1, len(metrics.get("served_customers", [])) or 1)
    reward = -args.cost_weight * float(metrics.get("total_cost", 0.0)) / n
    reward -= args.late_penalty * float(metrics.get("late_minutes", 0.0)) / n
    reward -= args.time_window_penalty * float(metrics.get("time_window_violations", 0.0)) / n
    reward -= args.capacity_penalty * float(metrics.get("capacity_violations", 0.0)) / n
    reward -= args.missing_penalty * len(metrics.get("missing_customers", [])) / n
    reward -= args.duplicate_penalty * float(metrics.get("duplicate_visits", 0.0)) / n
    reward -= args.vehicle_penalty * float(metrics.get("vehicles_excess", 0.0)) / n
    reward -= args.forced_penalty * float(metrics.get("forced_late_actions", 0.0)) / n
    return reward


def risk_reduce(values: list[float], args: argparse.Namespace) -> float:
    if not values:
        return 0.0
    if args.risk_objective == "mean" or len(values) == 1:
        return float(np.mean(values))
    count = max(1, int(np.ceil(len(values) * min(max(args.cvar_alpha, 0.01), 1.0))))
    return float(np.mean(sorted(values)[:count]))


def sg_bc_rollout(policy, snapshot, device: torch.device) -> dict:
    instance = snapshot["instance"]
    matrix = snapshot["traffic_matrix"]
    drifted = snapshot["drifted_due_times"]
    expert_order = snapshot["expert_order"]

    env = EventDrivenTWEnv(
        instance,
        matrix,
        mask_late=True,
        traffic_time_scale="depot_day",
        drifted_due_times=drifted,
    )

    losses: list[torch.Tensor] = []
    correct = 0
    decisions = 0

    while not env.done:
        state = env.decision_state()
        if state is None:
            break

        expert_customer = _expert_from_order(expert_order, env, state)
        if expert_customer <= 0:
            break

        customer_features = torch.tensor(env.candidate_features(state), dtype=torch.float32, device=device)
        vehicle_features = torch.tensor(env.vehicle_features(state), dtype=torch.float32, device=device)
        legal_mask = torch.tensor(state.legal_mask[1:], dtype=torch.bool, device=device)

        policy.eval()
        with torch.no_grad():
            logits = policy(customer_features, vehicle_features, legal_mask)
        policy.train()

        target = torch.tensor([expert_customer - 1], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        predicted = int(torch.argmax(logits).cpu().item()) + 1
        correct += int(predicted == expert_customer)
        decisions += 1

        env.step(expert_customer, state)

    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss = torch.stack(losses).mean() if losses else zero
    metrics = env.metrics()
    metrics["imit_accuracy"] = float(correct / max(1, decisions))
    return {"loss": loss, "metrics": metrics}


def _expert_from_order(expert_order: list[int], env: EventDrivenTWEnv, state) -> int:
    for node in expert_order:
        if not env.served[node] and state.legal_mask[node]:
            return int(node)
    candidates = [idx for idx, allowed in enumerate(state.legal_mask) if idx > 0 and allowed]
    if not candidates:
        candidates = [idx for idx, allowed in enumerate(state.relaxed_mask) if idx > 0 and allowed]
    if not candidates:
        return 0
    return min(candidates, key=lambda node: (float(env.instance.due_times[node]), node))


def rl_rollout(policy, snapshot, args, device: torch.device, *, decode: str):
    instance = snapshot["instance"]
    matrix = snapshot["traffic_matrix"]
    drifted = snapshot["drifted_due_times"]

    env = EventDrivenTWEnv(
        instance,
        matrix,
        mask_late=True,
        traffic_time_scale=args.traffic_time_scale,
        drifted_due_times=drifted,
    )
    return rollout_policy(policy, env, decode=decode, device=device)


def evaluate_sg_policy(policy, val_snapshots, args, device: torch.device) -> dict:
    rows = []
    policy.eval()
    with torch.no_grad():
        for idx, snap in enumerate(val_snapshots[: max(0, args.val_limit)]):
            result = rl_rollout(policy, snap, args, device, decode="greedy")
            rows.append(result.metrics)
    return aggregate(rows)


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"instances": 0}
    keys = [
        "total_cost", "late_minutes", "time_window_violations",
        "capacity_violations", "vehicles_excess", "route_count",
        "forced_late_actions", "cvr", "imit_accuracy",
    ]
    summary = {"instances": len(rows)}
    for key in keys:
        summary[key] = float(np.mean([float(row.get(key, 0.0)) for row in rows]))
    summary["feasibility"] = float(np.mean([1.0 if row.get("feasible") else 0.0 for row in rows]))
    return summary


def save_checkpoint(path, policy, args, sizes, history, best) -> None:
    output = ensure_dir(path)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "model_config": {
                "customer_feature_dim": policy.customer_feature_dim,
                "vehicle_feature_dim": policy.vehicle_feature_dim,
                "embed_dim": policy.embed_dim,
            },
            "training_config": vars(args),
            "supported_sizes": sizes,
            "history": history,
            "best": best,
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_config": {
                    "customer_feature_dim": policy.customer_feature_dim,
                    "vehicle_feature_dim": policy.vehicle_feature_dim,
                    "embed_dim": policy.embed_dim,
                },
                "training_config": vars(args),
                "supported_sizes": sizes,
                "history": history,
                "best": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    instance_map = load_instances_for_sizes(args.split_root, args.sizes, args.split, args.source)
    val_instance_map = load_instances_for_sizes(args.split_root, args.sizes, args.val_split, args.source)
    print(f"[LOAD] train_instances={len(instance_map)} val_instances={len(val_instance_map)}")

    train_snapshots = load_sg_dataset(args.dataset_dir, args.sizes, instance_map)
    val_snapshots = load_sg_dataset(args.dataset_dir, args.sizes, val_instance_map)
    print(f"[SNAPSHOTS] train={len(train_snapshots)} val={len(val_snapshots)}")

    if not train_snapshots:
        raise ValueError("No training snapshots loaded.")

    policy = EventDrivenSTPolicy(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history: list[dict] = []
    best: dict | None = None
    best_path = str(Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_best.pt"))

    total_epochs = args.bc_epochs + args.rl_epochs

    for epoch in range(1, total_epochs + 1):
        policy.train()
        phase = "bc" if epoch <= args.bc_epochs else "rl"
        rows = []
        losses = []
        rewards_log = []

        for step in range(args.steps_per_epoch):
            indices = rng.integers(0, len(train_snapshots), size=args.batch_size)
            batch_snapshots = [train_snapshots[int(i)] for i in indices]

            if phase == "bc":
                batch_losses = []
                for snap in batch_snapshots:
                    result = sg_bc_rollout(policy, snap, device)
                    batch_losses.append(result["loss"])
                    rows.append(result["metrics"])
                loss = torch.stack(batch_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                continue

            batch_log_probs = []
            batch_rewards = []
            for snap in batch_snapshots:
                sample_rewards = []
                sample_log_probs = []
                for s_idx in range(max(1, args.mc_samples_train)):
                    result = rl_rollout(policy, snap, args, device, decode="sample")
                    sample_rewards.append(reward_from_metrics(result.metrics, args))
                    sample_log_probs.append(result.log_prob)
                reward_value = risk_reduce(sample_rewards, args)
                selected_idx = int(np.argmin(sample_rewards)) if args.risk_objective == "cvar" else 0
                batch_log_probs.append(sample_log_probs[selected_idx])
                batch_rewards.append(reward_value)
                rows.append(result.metrics)

            reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            advantage = reward_tensor - reward_tensor.mean()
            if advantage.numel() == 1:
                advantage = reward_tensor.detach() * 0.0 + (reward_tensor - float(np.mean(rewards_log[-20:] or [0.0])))
            log_prob_tensor = torch.stack(batch_log_probs)
            loss = -(advantage.detach() * log_prob_tensor).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            rewards_log.extend(batch_rewards)

        train_summary = aggregate(rows)
        row = {
            "epoch": epoch,
            "phase": phase,
            "loss": float(np.mean(losses)),
            "reward": float(np.mean(rewards_log)) if rewards_log else 0.0,
            "train_total_cost": train_summary.get("total_cost", 0.0),
            "train_cvr": train_summary.get("cvr", 0.0),
            "train_feasibility": train_summary.get("feasibility", 0.0),
            "train_imit_accuracy": train_summary.get("imit_accuracy", 0.0),
        }

        if epoch % max(1, args.val_every) == 0 or epoch == 1:
            val_summary = evaluate_sg_policy(policy, val_snapshots, args, device)
            row.update({
                "val_total_cost": val_summary.get("total_cost", 0.0),
                "val_cvr": val_summary.get("cvr", 0.0),
                "val_feasibility": val_summary.get("feasibility", 0.0),
            })
            key = (-row["val_cvr"], row["val_feasibility"], -row["val_total_cost"])
            if best is None or key > best["key"]:
                best = {"key": key, "row": dict(row)}
                save_checkpoint(best_path, policy, args, args.sizes, history + [row], best)

        history.append(row)
        print(
            f"epoch={epoch:04d} phase={row['phase']} loss={row['loss']:.4f} "
            f"train_cvr={row['train_cvr']:.4f} train_feas={row['train_feasibility']:.3f} "
            f"train_acc={row['train_imit_accuracy']:.3f}"
            + (f" val_cvr={row['val_cvr']:.4f}" if "val_cvr" in row else "")
        )

    save_checkpoint(args.checkpoint, policy, args, args.sizes, history, best)
    print(f"[DONE] checkpoint={args.checkpoint} best={best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
