#!/usr/bin/env python
"""Train an event-driven online recourse policy for TWCVRP."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ensure_dir, SOLOMON_TW_DIR  # noqa: F401
from dataset import load_instances, split_instances
from env import EventDrivenTWEnv
from policy import EventDrivenSTPolicy
from rollout import rollout_policy
from traffic import planning_matrix, sample_traffic_matrix, stable_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train event-driven online recourse RL for TWCVRP.")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--size", default=None)
    parser.add_argument("--sizes", nargs="+", default=None)
    parser.add_argument("--source", default="auto", choices=["auto", "solomon", "npz"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--mode", default="traffic", choices=["static", "traffic"])
    parser.add_argument("--traffic_sigma", type=float, default=0.20)
    parser.add_argument("--traffic_buffer", type=float, default=0.50)
    parser.add_argument("--traffic_profile", default="additive", choices=["additive", "proportional"])
    parser.add_argument("--traffic_strength", type=float, default=1.0)
    parser.add_argument("--traffic_time_scale", default="depot_day", choices=["raw", "depot_day"])
    parser.add_argument("--risk_objective", default="cvar", choices=["mean", "cvar"])
    parser.add_argument("--cvar_alpha", type=float, default=0.20)
    parser.add_argument("--mc_samples_train", type=int, default=1)
    parser.add_argument("--mask_late", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imitation_epochs", type=int, default=0)
    parser.add_argument("--expert_strategy", default="earliest_due", choices=["earliest_due", "min_late"])
    parser.add_argument("--imitation_weight", type=float, default=1.0)
    parser.add_argument("--bc_weight_after_imitation", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--traffic_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_limit", type=int, default=4)
    parser.add_argument("--checkpoint", default="checkpoints/recourse_tw.pt")
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


def load_data(args: argparse.Namespace) -> dict:
    if args.input:
        instances = load_instances(input_path=args.input)
        split = split_instances(instances, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
        return {"train": split["train"], "val": split["val"], "supported_sizes": sorted({str(i.num_customers) for i in instances})}

    sizes = [str(size) for size in (args.sizes or ([args.size] if args.size else ["100"]))]
    train: list = []
    val: list = []
    for size in sizes:
        if args.split_root:
            train.extend(load_instances(data_root=Path(args.split_root) / "train", size=size, source=args.source))
            val.extend(load_instances(data_root=Path(args.split_root) / "val", size=size, source=args.source))
        else:
            instances = load_instances(data_root=args.data_root, size=size, source=args.source)
            split = split_instances(instances, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
            train.extend(split["train"])
            val.extend(split["val"])
    return {"train": train, "val": val, "supported_sizes": sizes}


def matrix_for_instance(instance, args: argparse.Namespace, *, epoch: int, step: int, sample_idx: int) -> np.ndarray:
    if args.mode == "static":
        return planning_matrix(instance, mode="static")
    return sample_traffic_matrix(
        instance,
        seed=stable_seed(instance.name, epoch, step, sample_idx, base_seed=args.traffic_seed),
        traffic_sigma=args.traffic_sigma,
        traffic_profile=args.traffic_profile,
        traffic_strength=args.traffic_strength,
    )


def rollout_instance(policy, instance, matrix, args, device: torch.device, *, decode: str):
    env = EventDrivenTWEnv(
        instance,
        matrix,
        mask_late=args.mask_late,
        traffic_time_scale=args.traffic_time_scale,
    )
    return rollout_policy(policy, env, decode=decode, device=device)


def expert_action(env: EventDrivenTWEnv, state, strategy: str) -> int:
    candidates = [idx for idx, allowed in enumerate(state.legal_mask) if idx > 0 and allowed]
    if not candidates:
        candidates = [idx for idx, allowed in enumerate(state.relaxed_mask) if idx > 0 and allowed]
    if not candidates:
        candidates = [idx for idx in range(1, env.num_customers + 1) if not env.served[idx]]
    if not candidates:
        return 0

    if strategy == "earliest_due":
        return min(candidates, key=lambda node: (float(env.instance.due_times[node]), node))
    if strategy == "min_late":
        current = int(state.current_node)
        current_time = float(state.current_time)

        def key(node: int) -> tuple[float, float, float, int]:
            arrival = current_time + float(env.travel[current, node])
            late = max(0.0, arrival - float(env.instance.due_times[node]))
            wait = max(0.0, float(env.instance.ready_times[node]) - arrival)
            return late, wait, float(env.travel[current, node]), node

        return min(candidates, key=key)
    raise ValueError("expert_strategy must be 'earliest_due' or 'min_late'.")


def imitation_rollout(policy, instance, matrix, args, device: torch.device) -> dict:
    env = EventDrivenTWEnv(
        instance,
        matrix,
        mask_late=args.mask_late,
        traffic_time_scale=args.traffic_time_scale,
    )
    losses: list[torch.Tensor] = []
    correct = 0
    decisions = 0

    while not env.done:
        state = env.decision_state()
        if state is None:
            break
        customer = expert_action(env, state, args.expert_strategy)
        if customer <= 0:
            break

        customer_features = torch.tensor(env.candidate_features(state), dtype=torch.float32, device=device)
        vehicle_features = torch.tensor(env.vehicle_features(state), dtype=torch.float32, device=device)
        legal_mask = torch.tensor(state.legal_mask[1:], dtype=torch.bool, device=device)
        logits = policy(customer_features, vehicle_features, legal_mask)
        target = torch.tensor([customer - 1], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        predicted = int(torch.argmax(logits.detach()).cpu().item()) + 1
        correct += int(predicted == customer)
        decisions += 1

        env.step(customer, state)

    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss = torch.stack(losses).mean() if losses else zero
    metrics = env.metrics()
    metrics["imit_accuracy"] = float(correct / max(1, decisions))
    metrics["imit_decisions"] = int(decisions)
    return {"loss": loss, "metrics": metrics}


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


def evaluate_policy(policy, instances, args, device: torch.device) -> dict:
    rows = []
    policy.eval()
    with torch.no_grad():
        for idx, instance in enumerate(instances[: max(0, args.val_limit)]):
            matrix = matrix_for_instance(instance, args, epoch=0, step=idx, sample_idx=0)
            result = rollout_instance(policy, instance, matrix, args, device, decode="greedy")
            rows.append(result.metrics)
    return aggregate(rows)


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"instances": 0}
    keys = [
        "total_cost",
        "late_minutes",
        "time_window_violations",
        "capacity_violations",
        "vehicles_excess",
        "route_count",
        "forced_late_actions",
        "cvr",
        "imit_accuracy",
    ]
    summary = {"instances": len(rows)}
    for key in keys:
        summary[key] = float(np.mean([float(row.get(key, 0.0)) for row in rows]))
    summary["feasibility"] = float(np.mean([1.0 if row.get("feasible") else 0.0 for row in rows]))
    return summary


def sample_instances(instances: list, batch_size: int, rng: np.random.Generator) -> list:
    selected = rng.integers(0, len(instances), size=batch_size)
    return [instances[int(idx)] for idx in selected]


def save_checkpoint(path: str | Path, policy, args, supported_sizes, history, best) -> None:
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
            "supported_sizes": supported_sizes,
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
                "supported_sizes": supported_sizes,
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
    data = load_data(args)
    train_items = data["train"]
    if not train_items:
        raise ValueError("Training split is empty.")
    policy = EventDrivenSTPolicy(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history: list[dict] = []
    best: dict | None = None
    best_path = str(Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_best.pt"))

    print(
        f"[START] recourse mode={args.mode} train={len(train_items)} val={len(data['val'])} "
        f"sizes={data['supported_sizes']} device={device} epochs={args.epochs}"
    )

    for epoch in range(1, args.epochs + 1):
        policy.train()
        phase = "imitation" if epoch <= max(0, args.imitation_epochs) else "reinforce"
        rows = []
        losses = []
        imitation_losses = []
        bc_losses_for_log = []
        rewards_for_log = []
        for step in range(args.steps_per_epoch):
            batch_instances = sample_instances(train_items, args.batch_size, rng)
            if phase == "imitation":
                batch_losses = []
                for instance in batch_instances:
                    matrix = matrix_for_instance(instance, args, epoch=epoch, step=step, sample_idx=0)
                    result = imitation_rollout(policy, instance, matrix, args, device)
                    batch_losses.append(result["loss"])
                    rows.append(result["metrics"])
                    rewards_for_log.append(reward_from_metrics(result["metrics"], args))
                loss = torch.stack(batch_losses).mean() * float(args.imitation_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                imitation_losses.append(loss_value)
                continue

            batch_log_probs = []
            batch_rewards = []
            batch_metrics = []
            batch_bc_losses = []
            for instance in batch_instances:
                sample_rewards = []
                sample_log_probs = []
                sample_metrics = []
                for sample_idx in range(max(1, args.mc_samples_train)):
                    matrix = matrix_for_instance(instance, args, epoch=epoch, step=step, sample_idx=sample_idx)
                    result = rollout_instance(policy, instance, matrix, args, device, decode="sample")
                    sample_rewards.append(reward_from_metrics(result.metrics, args))
                    sample_log_probs.append(result.log_prob)
                    sample_metrics.append(result.metrics)
                reward_value = risk_reduce(sample_rewards, args)
                selected_idx = int(np.argmin(sample_rewards)) if args.risk_objective == "cvar" else 0
                batch_log_probs.append(sample_log_probs[selected_idx])
                batch_rewards.append(reward_value)
                batch_metrics.append(sample_metrics[selected_idx])
                if args.bc_weight_after_imitation > 0.0:
                    matrix = matrix_for_instance(instance, args, epoch=epoch, step=step, sample_idx=0)
                    batch_bc_losses.append(imitation_rollout(policy, instance, matrix, args, device)["loss"])

            reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            advantage = reward_tensor - reward_tensor.mean()
            if advantage.numel() == 1:
                advantage = reward_tensor.detach() * 0.0 + (reward_tensor - float(np.mean(rewards_for_log[-20:] or [0.0])))
            log_prob_tensor = torch.stack(batch_log_probs)
            rl_loss = -(advantage.detach() * log_prob_tensor).mean()
            if batch_bc_losses:
                bc_loss = torch.stack(batch_bc_losses).mean()
                loss = rl_loss + float(args.bc_weight_after_imitation) * bc_loss
                bc_losses_for_log.append(float(bc_loss.detach().cpu().item()))
            else:
                loss = rl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            rewards_for_log.extend(batch_rewards)
            rows.extend(batch_metrics)

        train_summary = aggregate(rows)
        row = {
            "epoch": epoch,
            "phase": phase,
            "loss": float(np.mean(losses)),
            "reward": float(np.mean(rewards_for_log)),
            "imit_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
            "bc_loss": float(np.mean(bc_losses_for_log)) if bc_losses_for_log else 0.0,
            "train_total_cost": train_summary.get("total_cost", 0.0),
            "train_cvr": train_summary.get("cvr", 0.0),
            "train_feasibility": train_summary.get("feasibility", 0.0),
            "train_late_minutes": train_summary.get("late_minutes", 0.0),
            "train_forced_late_actions": train_summary.get("forced_late_actions", 0.0),
            "train_imit_accuracy": train_summary.get("imit_accuracy", 0.0),
        }
        if epoch % max(1, args.val_every) == 0 or epoch == 1:
            val_summary = evaluate_policy(policy, data["val"], args, device)
            row.update(
                {
                    "val_total_cost": val_summary.get("total_cost", 0.0),
                    "val_cvr": val_summary.get("cvr", 0.0),
                    "val_feasibility": val_summary.get("feasibility", 0.0),
                    "val_late_minutes": val_summary.get("late_minutes", 0.0),
                    "val_forced_late_actions": val_summary.get("forced_late_actions", 0.0),
                }
            )
            key = (
                -row["val_cvr"],
                row["val_feasibility"],
                -row["val_late_minutes"],
                -row["val_total_cost"],
            )
            if best is None or key > best["key"]:
                best = {"key": key, "row": dict(row)}
                save_checkpoint(best_path, policy, args, data["supported_sizes"], history + [row], best)
        history.append(row)
        print(
            f"epoch={epoch:04d} phase={row['phase']} loss={row['loss']:.4f} reward={row['reward']:.2f} "
            f"train_cvr={row['train_cvr']:.2f} train_feas={row['train_feasibility']:.2f} "
            f"train_late={row['train_late_minutes']:.2f} train_acc={row['train_imit_accuracy']:.2f}"
            + (
                f" val_cvr={row['val_cvr']:.2f} val_feas={row['val_feasibility']:.2f} "
                f"val_late={row['val_late_minutes']:.2f}"
                if "val_cvr" in row
                else ""
            )
        )

    save_checkpoint(args.checkpoint, policy, args, data["supported_sizes"], history, best)
    print(f"[DONE] checkpoint={args.checkpoint} best_checkpoint={best_path if best else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
