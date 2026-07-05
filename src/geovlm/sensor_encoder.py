"""Sensor token encoder: physical measurements → learnable tokens."""
import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    """Encodes [elevation, temperature, humidity] into 3 sensor tokens.

    Input:  [B, 3] raw sensor floats (elevation_m, temperature_c, humidity_pct)
    Output: [B, 3, 512] sensor tokens for Q-Former cross-attention

    Design: 3 independent MLPs with shared first layer, separate projections.
    Small enough (~10M params) to add negligible overhead.
    """

    def __init__(self, hidden_dim: int = 512, noise_std: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std  # >0 during Stage 3 training

        self.shared = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
        )
        self.proj_elev = nn.Linear(128, hidden_dim)
        self.proj_temp = nn.Linear(128, hidden_dim)
        self.proj_humid = nn.Linear(128, hidden_dim)

        # Learnable sensor-type embedding so Q-Former knows which is which
        self.type_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)

    def forward(self, sensor_values: torch.Tensor) -> torch.Tensor:
        """Args: sensor_values [B, 3] — order: elevation, temperature, humidity
           Returns: [B, 3, hidden_dim] sensor tokens
        """
        B = sensor_values.shape[0]

        # Training-time noise injection for robustness
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(sensor_values) * self.noise_std
            sensor_values = sensor_values + noise

        # Normalize to reasonable ranges before MLP
        elev = sensor_values[:, 0:1] / 5000.0    # 0-5000m → 0-1
        temp = sensor_values[:, 1:2] / 50.0       # -25-50°C → -0.5-1.0
        humid = sensor_values[:, 2:3] / 100.0     # 0-100% → 0-1

        feats = [
            self.proj_elev(self.shared(elev)),
            self.proj_temp(self.shared(temp)),
            self.proj_humid(self.shared(humid)),
        ]
        tokens = torch.stack(feats, dim=1)  # [B, 3, hidden_dim]
        tokens = tokens + self.type_embed.unsqueeze(0)  # add type identity
        return tokens
