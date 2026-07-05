"""Mixture of Experts for coordinate regression.

RouterMLP: 1536-dim visual features → 4 expert weights (softmax)
MoERegressor: Router + 4 CoordRegressor experts → weighted coordinate prediction

Training modes:
  1. expert_pretrain: train each expert independently on its scene data
  2. router_train: freeze experts, train router (Haversine + entropy + load_balance)
  3. joint_finetune: unfreeze all, low-lr fine-tuning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .coord_regressor import CoordRegressor, HaversineLoss
from .moe_config import (
    SceneType, NUM_EXPERTS,
    SPARSE_EXPERT_CONFIG, DEFAULT_EXPERT_CONFIG, SPARSE_EXPERT_THRESHOLD,
    ROUTER_HIDDEN_DIMS, ROUTER_TEMPERATURE, ROUTER_DROPOUT,
    ROUTER_ENTROPY_WEIGHT, ROUTER_LOAD_BALANCE_WEIGHT,
)


class RouterMLP(nn.Module):
    """MLP router: visual features → expert selection weights + scene logits.

    Input: (batch, 1536) visual features
    Output: (batch, 4) softmax weights + (batch, 4) raw logits for classification
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: list[int] = None,
        num_experts: int = NUM_EXPERTS,
        temperature: float = ROUTER_TEMPERATURE,
        dropout: float = ROUTER_DROPOUT,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = ROUTER_HIDDEN_DIMS

        self.temperature = temperature
        self.num_experts = num_experts

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_experts))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor, return_logits: bool = False):
        """Return softmax weights. If return_logits=True, also return raw logits.

        Returns:
            weights: (batch, 4) softmax
            logits: (batch, 4) raw, only if return_logits=True
        """
        logits = self.mlp(features)
        weights = F.softmax(logits / self.temperature, dim=-1)
        if return_logits:
            return weights, logits
        return weights

    def get_hard_assignment(self, features: torch.Tensor) -> torch.Tensor:
        """Return hard (one-hot) expert assignment. Shape: (batch, 4)."""
        weights = self.forward(features)
        idx = weights.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.num_experts).float()


class MoERegressor(nn.Module):
    """Mixture of Experts regressor: router + 4 expert heads → weighted prediction.

    Architecture:
        features (1536) ─┬─ RouterMLP ─→ weights (4)
                          │
                          ├─ Expert 0 (Urban) ─→ (lat, lng)
                          ├─ Expert 1 (Karst/Granite) ─→ (lat, lng)
                          ├─ Expert 2 (Alpine) ─→ (lat, lng)
                          └─ Expert 3 (Other) ─→ (lat, lng)

        output = sum(weight_i * expert_i(features))
    """

    def __init__(
        self,
        input_dim: int = 1536,
        expert_configs: Optional[list[dict]] = None,
        router_hidden_dims: list[int] = None,
        temperature: float = ROUTER_TEMPERATURE,
    ):
        super().__init__()

        # Create 4 experts with their individual configs
        if expert_configs is None:
            expert_configs = [DEFAULT_EXPERT_CONFIG] * NUM_EXPERTS

        self.experts = nn.ModuleList([
            CoordRegressor(
                input_dim=input_dim,
                hidden_dims=cfg.get("hidden_dims", DEFAULT_EXPERT_CONFIG["hidden_dims"]),
                dropout=cfg.get("dropout", DEFAULT_EXPERT_CONFIG["dropout"]),
            )
            for cfg in expert_configs
        ])

        self.router = RouterMLP(
            input_dim=input_dim,
            hidden_dims=router_hidden_dims or ROUTER_HIDDEN_DIMS,
            num_experts=NUM_EXPERTS,
            temperature=temperature,
        )

        self.num_experts = NUM_EXPERTS

    def forward(
        self,
        features: torch.Tensor,
        return_weights: bool = False,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            features: (batch, input_dim)
            return_weights: if True, also return router weights and individual expert preds
            return_logits: if True, also return raw router logits (for classification loss)

        Returns:
            If both False: (batch, 2) fused prediction
            If return_weights: (fused, weights, expert_preds)
            If return_logits: (fused, weights, expert_preds, logits)
        """
        if return_logits:
            weights, logits = self.router(features, return_logits=True)
        else:
            weights = self.router(features)

        expert_preds = torch.stack([expert(features) for expert in self.experts], dim=1)
        fused = (weights.unsqueeze(-1) * expert_preds).sum(dim=1)

        if return_logits:
            return fused, weights, expert_preds, logits
        if return_weights:
            return fused, weights, expert_preds
        return fused

    def predict(
        self,
        features: torch.Tensor,
        return_expert_info: bool = False,
    ) -> tuple:
        """Predict actual lat/lng coordinates.

        Args:
            features: (batch, 1536)
            return_expert_info: if True, include routing analysis

        Returns:
            If return_expert_info=False:
                (lat, lng) each (batch,) in degrees
            If return_expert_info=True:
                (lat, lng, weights, expert_lats, expert_lngs)
        """
        fused, weights, expert_preds = self.forward(features, return_weights=True)

        lat = fused[:, 0] * 90.0
        lng = fused[:, 1] * 180.0

        if return_expert_info:
            expert_lats = expert_preds[:, :, 0] * 90.0
            expert_lngs = expert_preds[:, :, 1] * 180.0
            return lat, lng, weights, expert_lats, expert_lngs
        return lat, lng

    def get_top_expert(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the index and weight of the top-1 expert for each sample."""
        weights = self.router(features)
        top_weight, top_idx = weights.max(dim=-1)
        return top_idx, top_weight

    # ── Expert management ──────────────────────────────────────────────

    def freeze_experts(self):
        """Freeze all expert parameters (for router training phase)."""
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = False

    def unfreeze_experts(self):
        """Unfreeze all expert parameters (for joint fine-tuning)."""
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = True

    def freeze_router(self):
        """Freeze only the router."""
        for p in self.router.parameters():
            p.requires_grad = False

    def unfreeze_router(self):
        """Unfreeze the router."""
        for p in self.router.parameters():
            p.requires_grad = True

    def get_expert(self, idx: int) -> CoordRegressor:
        """Get a specific expert by index."""
        return self.experts[idx]

    def load_expert_state(self, idx: int, state_dict: dict):
        """Load pre-trained weights into a specific expert."""
        self.experts[idx].load_state_dict(state_dict)

    def predict_constrained(
        self,
        features: torch.Tensor,
        constraint_box=None,
        blend_strength: float = 0.15,
        return_expert_info: bool = False,
        has_named_location: bool = False,
    ):
        """Predict with spatial constraint from cascade pipeline.

        Applies soft correction ONLY as a safety net when:
          - GeoCoT identified a specific named location
          - MoE prediction falls far outside the constraint region
          - Router is not highly confident

        Args:
            features: (batch, 1536) visual features
            constraint_box: ConstraintBox from spatial_constraint module
            blend_strength: how much to pull toward constraint center (0-1)
            return_expert_info: if True, include routing analysis
            has_named_location: True if GeoCoT named a specific place

        Returns:
            If return_expert_info=False: (lat, lng) each (batch,) in degrees
            If return_expert_info=True: (lat, lng, weights, expert_lats, expert_lngs)
        """
        lat, lng, weights, expert_lats, expert_lngs = self.predict(
            features, return_expert_info=True
        )

        if (constraint_box is not None
                and not getattr(constraint_box, 'is_default', True)
                and has_named_location):
            from .spatial_constraint import constraint_weighted_fusion

            # Get router confidence for gating
            top_weight = weights.max(dim=-1).values  # (batch,)

            adjusted_lat = []
            adjusted_lng = []
            for i in range(features.shape[0]):
                adj_lat, adj_lng = constraint_weighted_fusion(
                    lat[i].item(), lng[i].item(),
                    constraint_box,
                    blend_strength=blend_strength,
                    router_confidence=top_weight[i].item(),
                    has_named_location=has_named_location,
                )
                adjusted_lat.append(adj_lat)
                adjusted_lng.append(adj_lng)

            lat = torch.tensor(adjusted_lat, device=lat.device, dtype=lat.dtype)
            lng = torch.tensor(adjusted_lng, device=lng.device, dtype=lng.dtype)

        if return_expert_info:
            return lat, lng, weights, expert_lats, expert_lngs
        return lat, lng


class MoEHaversineLoss(nn.Module):
    """Haversine loss for MoE training.

    L_total = L_haversine + alpha * L_entropy + beta * L_load_balance
              + gamma * L_scene_classification

    Where:
    - L_entropy: encourages router to not collapse to single expert
    - L_load_balance: encourages even usage across batch
    - L_scene_classification: cross-entropy with scene labels (semi-supervised routing)
    """

    def __init__(
        self,
        entropy_weight: float = ROUTER_ENTROPY_WEIGHT,
        load_balance_weight: float = ROUTER_LOAD_BALANCE_WEIGHT,
        scene_class_weight: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.haversine = HaversineLoss(reduction=reduction)
        self.entropy_weight = entropy_weight
        self.load_balance_weight = load_balance_weight
        self.scene_class_weight = scene_class_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        pred_normed: torch.Tensor,
        true_normed: torch.Tensor,
        router_weights: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
        scene_labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total MoE loss.

        Args:
            pred_normed: (batch, 2) fused prediction
            true_normed: (batch, 2) ground truth
            router_weights: (batch, 4) router softmax output
            router_logits: (batch, 4) raw logits for scene classification
            scene_labels: (batch,) SceneType int labels (only for labeled samples)
            label_mask: (batch,) bool mask, True = has scene label

        Returns:
            (total_loss, components_dict)
        """
        loss_hav = self.haversine(pred_normed, true_normed)
        components = {"haversine": loss_hav.item()}

        loss_total = loss_hav

        if router_weights is not None:
            # Entropy regularization: reward high entropy
            entropy = -(router_weights * torch.log(router_weights + 1e-8)).sum(dim=-1).mean()
            loss_entropy = -self.entropy_weight * entropy
            loss_total = loss_total + loss_entropy

            # Load balancing: penalize uneven expert usage
            mean_usage = router_weights.mean(dim=0)
            load_balance = (mean_usage.std() / (mean_usage.mean() + 1e-8))
            loss_balance = self.load_balance_weight * load_balance
            loss_total = loss_total + loss_balance

            components["entropy"] = entropy.item()
            components["entropy_penalty"] = loss_entropy.item()
            components["load_balance"] = loss_balance.item()

        # Semi-supervised scene classification loss
        if (self.scene_class_weight > 0 and router_logits is not None
                and scene_labels is not None and label_mask is not None):
            if label_mask.any():
                loss_scene = self.ce_loss(
                    router_logits[label_mask],
                    scene_labels[label_mask],
                )
                loss_total = loss_total + self.scene_class_weight * loss_scene
                components["scene_class"] = loss_scene.item()

                # Accuracy on labeled samples
                pred_scene = router_logits[label_mask].argmax(dim=-1)
                acc = (pred_scene == scene_labels[label_mask]).float().mean()
                components["scene_acc"] = acc.item()

        components["total"] = loss_total.item()
        return loss_total, components


def get_expert_configs(sample_counts: dict[int, int]) -> list[dict]:
    """Determine expert configs based on per-expert sample counts.

    Args:
        sample_counts: {expert_idx: num_samples}

    Returns:
        list of 4 config dicts for CoordRegressor construction
    """
    configs = []
    for i in range(NUM_EXPERTS):
        count = sample_counts.get(i, 0)
        if count < SPARSE_EXPERT_THRESHOLD:
            configs.append(SPARSE_EXPERT_CONFIG)
        else:
            configs.append(DEFAULT_EXPERT_CONFIG)
    return configs
