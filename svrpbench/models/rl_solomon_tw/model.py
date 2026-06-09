"""Lightweight pointer policy for standalone Solomon TWCVRP RL."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class RolloutOutput:
    actions: torch.Tensor
    log_probs: torch.Tensor


class TWPointerPolicy(nn.Module):
    """A compact attention policy that emits a customer permutation."""

    def __init__(
        self,
        embed_dim: int = 128,
        feature_dim: int = 6,
        context_dim: int = 9,
        dynamic_feature_dim: int = 7,
        model_version: str = "tw_pointer_dynamic_v2",
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feature_dim = int(feature_dim)
        self.context_dim = int(context_dim)
        self.dynamic_feature_dim = int(dynamic_feature_dim)
        self.model_version = model_version
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dynamic_encoder = (
            nn.Sequential(
                nn.Linear(self.dynamic_feature_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            if self.dynamic_feature_dim > 0
            else None
        )
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        coords_norm: torch.Tensor,
        current_nodes: torch.Tensor,
        remaining_capacity_norm: torch.Tensor,
        current_time_norm: torch.Tensor,
        visited: torch.Tensor,
        dynamic_features: torch.Tensor | None = None,
        used_vehicles_norm: torch.Tensor | None = None,
        remaining_vehicles_norm: torch.Tensor | None = None,
        customers_remaining_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits over customer nodes 1..N."""
        batch_size = features.shape[0]
        customer_emb = self.node_encoder(features[:, 1:, :])
        if self.dynamic_encoder is not None:
            if dynamic_features is None:
                dynamic_features = torch.zeros(
                    (batch_size, customer_emb.shape[1], self.dynamic_feature_dim),
                    dtype=customer_emb.dtype,
                    device=customer_emb.device,
                )
            customer_emb = customer_emb + self.dynamic_encoder(dynamic_features)

        batch_idx = torch.arange(batch_size, device=features.device)
        current_xy = coords_norm[batch_idx, current_nodes]
        depot_xy = coords_norm[:, 0]
        context_parts = [
            current_xy,
            depot_xy,
            remaining_capacity_norm.unsqueeze(-1),
            current_time_norm.unsqueeze(-1),
        ]
        if self.context_dim > 6:
            zeros = torch.zeros_like(current_time_norm)
            context_parts.extend(
                [
                    (used_vehicles_norm if used_vehicles_norm is not None else zeros).unsqueeze(-1),
                    (
                        remaining_vehicles_norm
                        if remaining_vehicles_norm is not None
                        else zeros
                    ).unsqueeze(-1),
                    (
                        customers_remaining_norm
                        if customers_remaining_norm is not None
                        else zeros
                    ).unsqueeze(-1),
                ]
            )
        context = torch.cat(context_parts, dim=-1)
        context_emb = self.context_encoder(context).unsqueeze(1)
        logits = self.score(torch.tanh(self.key(customer_emb) + self.query(context_emb))).squeeze(-1)
        logits = logits.masked_fill(visited[:, 1:].clone(), float("-inf"))
        return logits


def rollout(policy: TWPointerPolicy, batch: dict[str, torch.Tensor], *, decode: str) -> RolloutOutput:
    """Sample or greedily decode one customer permutation per instance."""
    features = batch["features"]
    coords_norm = batch["coords_norm"]
    demands = batch["demands"]
    capacity = batch["capacity"]
    ready_norm = batch["ready_norm"]
    due_norm = batch["due_norm"]
    service_norm = batch["service_norm"]
    vehicle_count = torch.clamp(batch["vehicle_count"], min=1.0)
    batch_size, num_nodes, _ = features.shape
    num_customers = num_nodes - 1
    device = features.device

    visited = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)
    visited[:, 0] = True
    current_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)
    remaining_capacity = capacity.clone()
    current_time_norm = torch.zeros(batch_size, dtype=torch.float32, device=device)
    used_vehicles = torch.ones(batch_size, dtype=torch.float32, device=device)
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []

    for step in range(num_customers):
        remaining_norm = remaining_capacity / torch.clamp(capacity, min=1e-6)
        used_vehicles_norm = used_vehicles / vehicle_count
        remaining_vehicles_norm = torch.clamp(vehicle_count - used_vehicles, min=0.0) / vehicle_count
        customers_remaining_norm = torch.full(
            (batch_size,),
            float(num_customers - step) / max(1, num_customers),
            dtype=torch.float32,
            device=device,
        )
        dynamic_features = _candidate_dynamic_features(
            coords_norm,
            demands,
            capacity,
            current_nodes,
            remaining_capacity,
            current_time_norm,
            ready_norm,
            due_norm,
        )
        logits = policy(
            features,
            coords_norm,
            current_nodes,
            remaining_norm,
            current_time_norm,
            visited,
            dynamic_features=dynamic_features,
            used_vehicles_norm=used_vehicles_norm,
            remaining_vehicles_norm=remaining_vehicles_norm,
            customers_remaining_norm=customers_remaining_norm,
        )
        if decode == "greedy":
            action_customer = torch.argmax(logits, dim=-1)
            step_log_prob = torch.zeros(batch_size, dtype=torch.float32, device=device)
        else:
            distribution = Categorical(logits=logits)
            action_customer = distribution.sample()
            step_log_prob = distribution.log_prob(action_customer)

        next_nodes = action_customer + 1
        batch_idx = torch.arange(batch_size, device=device)
        demand = demands[batch_idx, next_nodes]
        needs_new_route = demand > remaining_capacity
        used_vehicles = torch.where(needs_new_route, used_vehicles + 1.0, used_vehicles)
        remaining_capacity = torch.where(
            needs_new_route,
            capacity - demand,
            remaining_capacity - demand,
        )

        travel_norm = torch.linalg.vector_norm(
            coords_norm[batch_idx, next_nodes] - coords_norm[batch_idx, current_nodes],
            dim=-1,
        )
        arrival_norm = torch.where(needs_new_route, travel_norm, current_time_norm + travel_norm)
        wait_norm = torch.relu(ready_norm[batch_idx, next_nodes] - arrival_norm)
        current_time_norm = arrival_norm + wait_norm + service_norm[batch_idx, next_nodes]

        visited[batch_idx, next_nodes] = True
        current_nodes = next_nodes
        actions.append(next_nodes)
        log_probs.append(step_log_prob)

    return RolloutOutput(actions=torch.stack(actions, dim=1), log_probs=torch.stack(log_probs, dim=1))


def actions_to_order(actions: torch.Tensor) -> list[int]:
    return [int(node) for node in actions.detach().cpu().tolist()]


def order_log_probs(
    policy: TWPointerPolicy,
    batch: dict[str, torch.Tensor],
    orders: torch.Tensor,
) -> torch.Tensor:
    """Return log-probability of teacher-forced customer orders."""
    features = batch["features"]
    coords_norm = batch["coords_norm"]
    demands = batch["demands"]
    capacity = batch["capacity"]
    ready_norm = batch["ready_norm"]
    due_norm = batch["due_norm"]
    service_norm = batch["service_norm"]
    vehicle_count = torch.clamp(batch["vehicle_count"], min=1.0)
    batch_size, num_nodes, _ = features.shape
    num_customers = num_nodes - 1
    device = features.device

    visited = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)
    visited[:, 0] = True
    current_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)
    remaining_capacity = capacity.clone()
    current_time_norm = torch.zeros(batch_size, dtype=torch.float32, device=device)
    used_vehicles = torch.ones(batch_size, dtype=torch.float32, device=device)
    log_probs: list[torch.Tensor] = []

    for step in range(num_customers):
        remaining_norm = remaining_capacity / torch.clamp(capacity, min=1e-6)
        used_vehicles_norm = used_vehicles / vehicle_count
        remaining_vehicles_norm = torch.clamp(vehicle_count - used_vehicles, min=0.0) / vehicle_count
        customers_remaining_norm = torch.full(
            (batch_size,),
            float(num_customers - step) / max(1, num_customers),
            dtype=torch.float32,
            device=device,
        )
        dynamic_features = _candidate_dynamic_features(
            coords_norm,
            demands,
            capacity,
            current_nodes,
            remaining_capacity,
            current_time_norm,
            ready_norm,
            due_norm,
        )
        logits = policy(
            features,
            coords_norm,
            current_nodes,
            remaining_norm,
            current_time_norm,
            visited,
            dynamic_features=dynamic_features,
            used_vehicles_norm=used_vehicles_norm,
            remaining_vehicles_norm=remaining_vehicles_norm,
            customers_remaining_norm=customers_remaining_norm,
        )
        next_nodes = orders[:, step]
        action_customer = next_nodes - 1
        step_log_probs = F.log_softmax(logits, dim=-1).gather(1, action_customer.unsqueeze(1)).squeeze(1)
        batch_idx = torch.arange(batch_size, device=device)
        demand = demands[batch_idx, next_nodes]
        needs_new_route = demand > remaining_capacity
        used_vehicles = torch.where(needs_new_route, used_vehicles + 1.0, used_vehicles)
        remaining_capacity = torch.where(
            needs_new_route,
            capacity - demand,
            remaining_capacity - demand,
        )
        travel_norm = torch.linalg.vector_norm(
            coords_norm[batch_idx, next_nodes] - coords_norm[batch_idx, current_nodes],
            dim=-1,
        )
        arrival_norm = torch.where(needs_new_route, travel_norm, current_time_norm + travel_norm)
        wait_norm = torch.relu(ready_norm[batch_idx, next_nodes] - arrival_norm)
        current_time_norm = arrival_norm + wait_norm + service_norm[batch_idx, next_nodes]
        visited[batch_idx, next_nodes] = True
        current_nodes = next_nodes
        log_probs.append(step_log_probs)

    return torch.stack(log_probs, dim=1).sum(dim=1)


def _candidate_dynamic_features(
    coords_norm: torch.Tensor,
    demands: torch.Tensor,
    capacity: torch.Tensor,
    current_nodes: torch.Tensor,
    remaining_capacity: torch.Tensor,
    current_time_norm: torch.Tensor,
    ready_norm: torch.Tensor,
    due_norm: torch.Tensor,
) -> torch.Tensor:
    batch_size = coords_norm.shape[0]
    batch_idx = torch.arange(batch_size, device=coords_norm.device)
    current_xy = coords_norm[batch_idx, current_nodes].unsqueeze(1)
    customer_xy = coords_norm[:, 1:, :]
    travel_norm = torch.linalg.vector_norm(customer_xy - current_xy, dim=-1)
    arrival_norm = current_time_norm.unsqueeze(-1) + travel_norm
    ready_customer = ready_norm[:, 1:]
    due_customer = due_norm[:, 1:]
    wait_norm = torch.relu(ready_customer - arrival_norm)
    service_start_norm = arrival_norm + wait_norm
    late_norm = torch.relu(service_start_norm - due_customer)
    slack_norm = torch.clamp(due_customer - service_start_norm, min=-1.0, max=1.0)
    demand_customer = demands[:, 1:]
    demand_feasible = (demand_customer <= remaining_capacity.unsqueeze(-1)).to(coords_norm.dtype)
    remaining_after_norm = (
        (remaining_capacity.unsqueeze(-1) - demand_customer)
        / torch.clamp(capacity.unsqueeze(-1), min=1e-6)
    ).clamp(min=-1.0, max=1.0)
    return torch.stack(
        [
            travel_norm,
            arrival_norm,
            wait_norm,
            late_norm,
            slack_norm,
            demand_feasible,
            remaining_after_norm,
        ],
        dim=-1,
    )
