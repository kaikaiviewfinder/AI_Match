"""Coordinate regression head: visual features → (lat, lng).

MLP architecture:
    input (1280) → Linear(512) → ReLU → Dropout → Linear(128) → ReLU → Linear(2)

Loss: Haversine distance — the great-circle distance between predicted and
true coordinates. This directly optimizes for the physical distance error
rather than MSE in coordinate space.

Normalization: lat/lng are normalized to roughly [-1, 1] range for stable training.
    lat_norm = lat / 90.0
    lng_norm = lng / 180.0
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordRegressor(nn.Module):
    """MLP that maps visual features (1280-dim) to (lat, lng) coordinates."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: list[int] = [512, 256, 64, 16],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 2))  # lat, lng

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for ReLU layers, small init for output."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Small init for the last layer (output is normalized coordinates)
        last_linear = None
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, std=0.01)
            nn.init.constant_(last_linear.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (batch, 1280), Output: (batch, 2) [lat_norm, lng_norm]."""
        return self.mlp(features)

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict actual lat/lng from features.

        Returns:
            lat: (batch,) in [-90, 90]
            lng: (batch,) in [-180, 180]
        """
        normed = self.forward(features)
        lat = normed[:, 0] * 90.0
        lng = normed[:, 1] * 180.0
        return lat, lng


class HaversineLoss(nn.Module):
    """Haversine great-circle distance loss in kilometers.

    L = 2 * R * arcsin(sqrt(hav(Δlat) + cos(lat1)*cos(lat2)*hav(Δlng)))

    Where R = 6371 km (Earth radius).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self._radius = 6371.0

    def forward(self, pred_normed: torch.Tensor, true_normed: torch.Tensor) -> torch.Tensor:
        """Compute Haversine distance between predicted and true coordinates.

        Args:
            pred_normed: (batch, 2) predicted [lat/90, lng/180]
            true_normed: (batch, 2) true [lat/90, lng/180]

        Returns:
            Scalar loss in kilometers.
        """
        pred_lat = pred_normed[:, 0] * math.pi / 2.0   # rad
        pred_lng = pred_normed[:, 1] * math.pi
        true_lat = true_normed[:, 0] * math.pi / 2.0
        true_lng = true_normed[:, 1] * math.pi

        dlat = true_lat - pred_lat
        dlng = true_lng - pred_lng

        a = torch.sin(dlat / 2) ** 2 + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlng / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a.clamp(0, 1)))

        dist_km = self._radius * c

        if self.reduction == "mean":
            return dist_km.mean()
        elif self.reduction == "sum":
            return dist_km.sum()
        return dist_km


def create_expert(
    input_dim: int = 1536,
    config: dict = None,
    pretrained_path: str = None,
    device: str = "cuda",
) -> CoordRegressor:
    """Factory: create a CoordRegressor expert with optional pretrained weights.

    Args:
        input_dim: feature vector dimension
        config: dict with keys hidden_dims, dropout (from moe_config)
        pretrained_path: path to .pt state_dict for warm-start
        device: target device

    Returns:
        CoordRegressor ready for training or inference
    """
    if config is None:
        config = {"hidden_dims": [512, 256, 64, 16], "dropout": 0.1}

    model = CoordRegressor(
        input_dim=input_dim,
        hidden_dims=config.get("hidden_dims", [512, 256, 64, 16]),
        dropout=config.get("dropout", 0.1),
    )

    if pretrained_path and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        # If pretrained model has different hidden_dims, only load compatible layers
        model_state = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
        skipped = len(state) - len(compatible)
        if skipped > 0:
            print(f"  create_expert: warm-started {len(compatible)}/{len(state)} layers "
                  f"({skipped} incompatible layers skipped)")

    model.to(device)
    return model


def haversine_distance_km(lat1, lng1, lat2, lng2):
    """Compute Haversine distance (km) between two coordinate pairs (scalars or tensors)."""
    R = 6371.0
    lat1_r = math.radians(lat1) if not isinstance(lat1, torch.Tensor) else lat1 * math.pi / 180.0
    lng1_r = math.radians(lng1) if not isinstance(lng1, torch.Tensor) else lng1 * math.pi / 180.0
    lat2_r = math.radians(lat2) if not isinstance(lat2, torch.Tensor) else lat2 * math.pi / 180.0
    lng2_r = math.radians(lng2) if not isinstance(lng2, torch.Tensor) else lng2 * math.pi / 180.0

    dlat = lat2_r - lat1_r
    dlng = lng2_r - lng1_r

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlng / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a.clamp(0, 1)))
    return R * c
