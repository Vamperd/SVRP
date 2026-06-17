"""Rollout helpers for event-driven TWCVRP recourse policies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from env import EventDrivenTWEnv


@dataclass
class RecourseRollout:
    routes: list[list[int]]
    metrics: dict
    log_prob: torch.Tensor
    actions: list[tuple[int, int]]


def rollout_policy(
    policy,
    env: EventDrivenTWEnv,
    *,
    decode: str,
    device: torch.device,
) -> RecourseRollout:
    log_probs: list[torch.Tensor] = []
    actions: list[tuple[int, int]] = []
    zero = torch.zeros((), dtype=torch.float32, device=device)

    while not env.done:
        state = env.decision_state()
        if state is None:
            break
        customer_features = torch.tensor(env.candidate_features(state), dtype=torch.float32, device=device)
        vehicle_features = torch.tensor(env.vehicle_features(state), dtype=torch.float32, device=device)
        legal_mask = torch.tensor(state.legal_mask[1:], dtype=torch.bool, device=device)
        logits = policy(customer_features, vehicle_features, legal_mask)
        if not torch.isfinite(logits).any() or torch.all(logits <= -1.0e7):
            action_idx = _fallback_action(env, state)
            step_log_prob = zero
        elif decode == "greedy":
            action_idx = int(torch.argmax(logits).detach().cpu().item())
            step_log_prob = zero
        else:
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()
            action_idx = int(action_tensor.detach().cpu().item())
            step_log_prob = dist.log_prob(action_tensor)

        customer = action_idx + 1
        env.step(customer, state)
        actions.append((int(state.vehicle_idx), int(customer)))
        log_probs.append(step_log_prob)

    metrics = env.metrics()
    log_prob = torch.stack(log_probs).sum() if log_probs else zero
    return RecourseRollout(
        routes=metrics["routes"],
        metrics=metrics,
        log_prob=log_prob,
        actions=actions,
    )


def _fallback_action(env: EventDrivenTWEnv, state) -> int:
    candidates = np.where(state.relaxed_mask[1:])[0]
    if len(candidates) == 0:
        candidates = np.where(~env.served[1:])[0]
    if len(candidates) == 0:
        return 0
    current = int(state.current_node)
    current_time = float(state.current_time)
    best_idx = int(candidates[0])
    best_key = None
    for idx in candidates:
        node = int(idx) + 1
        arrival = current_time + float(env.travel[current, node])
        late = max(0.0, arrival - float(env.instance.due_times[node]))
        key = (late, float(env.travel[current, node]), node)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(idx)
    return best_idx

