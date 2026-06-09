"""Lightweight pointer-style policy for standalone CVRP RL."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class RolloutOutput:
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    distances: torch.Tensor
    capacity_violation: torch.Tensor


class PointerPolicy(nn.Module):
    """A compact attention policy that orders all customers once."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        locations_norm: torch.Tensor,
        demands_norm: torch.Tensor,
        current_nodes: torch.Tensor,
        remaining_capacity_norm: torch.Tensor,
        visited: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits over customer nodes 1..N."""
        batch_size, num_nodes, _ = locations_norm.shape
        customer_features = torch.cat(
            [locations_norm[:, 1:, :], demands_norm[:, 1:].unsqueeze(-1)],
            dim=-1,
        )
        node_emb = self.node_encoder(customer_features)

        batch_idx = torch.arange(batch_size, device=locations_norm.device)
        current_xy = locations_norm[batch_idx, current_nodes]
        depot_xy = locations_norm[:, 0]
        context = torch.cat(
            [current_xy, depot_xy, remaining_capacity_norm.unsqueeze(-1)],
            dim=-1,
        )
        context_emb = self.context_encoder(context).unsqueeze(1)

        logits = self.score(torch.tanh(self.key(node_emb) + self.query(context_emb))).squeeze(-1)
        # Clone the mask because rollout updates ``visited`` in later steps.
        # Autograd keeps the mask for backward, so sharing the same tensor would
        # look like an in-place modification of a saved variable.
        logits = logits.masked_fill(visited[:, 1:].clone(), float("-inf"))
        return logits


def rollout(
    policy: PointerPolicy,
    batch: dict[str, torch.Tensor],
    *,
    decode: str,
    capacity_penalty: float = 100.0,
) -> RolloutOutput:
    """Roll out one full route per instance."""
    locations = batch["locations"]
    locations_norm = batch["locations_norm"]
    demands = batch["demands"]
    demands_norm = batch["demands_norm"]
    capacity = batch["capacity"]

    batch_size, num_nodes, _ = locations.shape
    num_customers = num_nodes - 1
    device = locations.device

    visited = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)
    visited[:, 0] = True
    current_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)
    remaining_capacity = capacity.clone()
    route_distance = torch.zeros(batch_size, dtype=torch.float32, device=device)
    log_probs: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []

    for _ in range(num_customers):
        remaining_norm = remaining_capacity / torch.clamp(capacity, min=1e-6)
        logits = policy(locations_norm, demands_norm, current_nodes, remaining_norm, visited)
        if decode == "greedy":
            action_customer = torch.argmax(logits, dim=-1)
            step_log_prob = torch.zeros(batch_size, dtype=torch.float32, device=device)
        else:
            distribution = Categorical(logits=logits)
            action_customer = distribution.sample()
            step_log_prob = distribution.log_prob(action_customer)

        next_nodes = action_customer + 1
        batch_idx = torch.arange(batch_size, device=device)
        route_distance += torch.linalg.vector_norm(
            locations[batch_idx, next_nodes] - locations[batch_idx, current_nodes],
            dim=-1,
        )
        remaining_capacity = remaining_capacity - demands[batch_idx, next_nodes]
        visited[batch_idx, next_nodes] = True
        current_nodes = next_nodes
        log_probs.append(step_log_prob)
        actions.append(next_nodes)

    batch_idx = torch.arange(batch_size, device=device)
    route_distance += torch.linalg.vector_norm(
        locations[batch_idx, current_nodes] - locations[:, 0],
        dim=-1,
    )
    capacity_violation = torch.relu(-remaining_capacity)
    rewards = -route_distance - capacity_penalty * capacity_violation

    return RolloutOutput(
        actions=torch.stack(actions, dim=1),
        log_probs=torch.stack(log_probs, dim=1),
        rewards=rewards,
        distances=route_distance,
        capacity_violation=capacity_violation,
    )


def actions_to_route(actions: torch.Tensor) -> list[int]:
    return [0] + [int(node) for node in actions.detach().cpu().tolist()] + [0]
