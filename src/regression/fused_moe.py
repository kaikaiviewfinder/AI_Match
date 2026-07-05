"""Fused MoE regressor: visual features + VLM hidden states → coordinates.

Extends MoERegressor with dual-input pathways:
  - concat: [visual | vlm_hidden] → projection → Router + Experts
  - gated:  vlm_hidden → Router, visual → Experts

This bypasses the text bottleneck — VLM's geographic reasoning enters the
MoE as a dense vector, without compression through text → parser → KB.
"""

import torch
import torch.nn as nn
from typing import Optional

from .moe import MoERegressor
from .coord_regressor import CoordRegressor
from .moe_config import (
    NUM_EXPERTS, DEFAULT_EXPERT_CONFIG,
    ROUTER_HIDDEN_DIMS, ROUTER_TEMPERATURE,
)


class FusedMoERegressor(MoERegressor):
    """MoE with VLM hidden state fusion.

    Two fusion modes:
      - "concat" (default): visual(1536) + vlm_hidden(1536) → Linear(3072→1536)
                            → Router + Experts. Both components see both signals.
      - "gated":  vlm_hidden → Router (4 weights), visual → 4 Experts → weighted sum.
                  Cleaner division: VLM decides where, experts decide where.

    Reuses all MoERegressor training utilities (freeze/unfreeze, load_expert_state).
    """

    def __init__(
        self,
        fusion_mode: str = "concat",
        input_dim: int = 1536,
        vlm_hidden_dim: int = 1536,
        expert_configs: Optional[list[dict]] = None,
        router_hidden_dims: list[int] = None,
        temperature: float = ROUTER_TEMPERATURE,
    ):
        self.fusion_mode = fusion_mode
        self.vlm_hidden_dim = vlm_hidden_dim

        if fusion_mode == "concat":
            fused_dim = input_dim + vlm_hidden_dim
            super().__init__(
                input_dim=input_dim,
                expert_configs=expert_configs,
                router_hidden_dims=router_hidden_dims,
                temperature=temperature,
            )
            self.fusion_proj = nn.Linear(fused_dim, input_dim)

        elif fusion_mode == "gated":
            super().__init__(
                input_dim=input_dim,
                expert_configs=expert_configs,
                router_hidden_dims=router_hidden_dims,
                temperature=temperature,
            )
            # Replace router to take vlm_hidden_dim instead of visual input_dim
            self.router = self._make_router(vlm_hidden_dim, router_hidden_dims, temperature)

        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def _make_router(self, input_dim, hidden_dims, temperature):
        """Create a RouterMLP with the given input dimension."""
        from .moe import RouterMLP
        return RouterMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims or ROUTER_HIDDEN_DIMS,
            num_experts=NUM_EXPERTS,
            temperature=temperature,
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        vlm_hidden: torch.Tensor,
        return_weights: bool = False,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass with dual inputs.

        Args:
            visual_features: (batch, 1536) from vision encoder
            vlm_hidden: (batch, vlm_hidden_dim) from VLM language model
            return_weights: also return router weights and expert preds
            return_logits: also return raw router logits

        Returns:
            (batch, 2) fused prediction, plus optional extras
        """
        if self.fusion_mode == "concat":
            fused = self.fusion_proj(
                torch.cat([visual_features, vlm_hidden], dim=-1)
            )
            return super().forward(fused, return_weights=return_weights,
                                   return_logits=return_logits)

        elif self.fusion_mode == "gated":
            if return_logits:
                weights, logits = self.router(vlm_hidden, return_logits=True)
            else:
                weights = self.router(vlm_hidden)

            expert_preds = torch.stack(
                [expert(visual_features) for expert in self.experts], dim=1
            )
            fused = (weights.unsqueeze(-1) * expert_preds).sum(dim=1)

            if return_logits:
                return fused, weights, expert_preds, logits
            if return_weights:
                return fused, weights, expert_preds
            return fused

    def predict(
        self,
        visual_features: torch.Tensor,
        vlm_hidden: torch.Tensor,
        return_expert_info: bool = False,
    ):
        """Predict actual lat/lng coordinates from dual inputs.

        Returns:
            If return_expert_info=False: (lat, lng) each (batch,) in degrees
            If return_expert_info=True: (lat, lng, weights, expert_lats, expert_lngs)
        """
        fused, weights, expert_preds = self.forward(
            visual_features, vlm_hidden, return_weights=True
        )

        lat = fused[:, 0] * 90.0
        lng = fused[:, 1] * 180.0

        if return_expert_info:
            expert_lats = expert_preds[:, :, 0] * 90.0
            expert_lngs = expert_preds[:, :, 1] * 180.0
            return lat, lng, weights, expert_lats, expert_lngs
        return lat, lng
