"""Time-aware attention policy for event-driven recourse decisions."""
from __future__ import annotations

import torch
from torch import nn


class EventDrivenSTPolicy(nn.Module):
    """Score one available vehicle against all remaining customer candidates."""

    def __init__(
        self,
        *,
        customer_feature_dim: int = 14,
        vehicle_feature_dim: int = 9,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.customer_feature_dim = int(customer_feature_dim)
        self.vehicle_feature_dim = int(vehicle_feature_dim)
        self.embed_dim = int(embed_dim)
        self.customer_encoder = nn.Sequential(
            nn.Linear(self.customer_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(self.vehicle_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        customer_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits over customers 1..N.

        ``customer_features`` has shape ``[N, F]`` and ``legal_mask`` has shape
        ``[N]``. Masked logits are set to a large negative value for hard
        masking during sampling and greedy decoding.
        """
        customer_emb = self.customer_encoder(customer_features)
        vehicle_emb = self.vehicle_encoder(vehicle_features).unsqueeze(0)
        logits = self.score(torch.tanh(self.key(customer_emb) + self.query(vehicle_emb))).squeeze(-1)
        return logits.masked_fill(~legal_mask.bool(), -1.0e8)

