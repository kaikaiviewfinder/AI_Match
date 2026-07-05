"""Field-Aware Q-Former: 8 learnable queries extract per-field info from visual tokens.

Each geographic field (climate, terrain, vegetation, urbanization, elevation,
architecture, pavement, language) gets one dedicated learnable query vector.
Field-Type Embeddings give each query its "identity" before attention.

Architecture (per block):
  Cross-Attention: queries ← visual_tokens + sensor_tokens
  Self-Attention:  queries ← other_queries       (field interdependency)
  FFN: SwiGLU with hidden_dim=2048

3 blocks total. ~110M parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldAwareQFormer(nn.Module):
    """Field-Aware Q-Former with 16 learnable queries, 3 transformer blocks."""

    FIELD_NAMES = [
        # Original 8
        "climate_zone", "terrain_type", "vegetation_zone", "urbanization",
        "architecture_style", "pavement_type", "language_script", "visible_text",
        # New 8 for spatial narrowing
        "soil_color", "sky_quality", "mountain_rock_type", "tree_species",
        "building_height", "water_type", "landform_detail", "scene_type",
    ]

    def __init__(self, hidden_dim: int = 512, num_blocks: int = 3, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fields = len(self.FIELD_NAMES)

        # 8 learnable field queries
        self.queries = nn.Parameter(torch.randn(self.num_fields, hidden_dim) * 0.02)

        # Field-Type Embeddings — give each query an identity
        self.field_type_embed = nn.Parameter(torch.randn(self.num_fields, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(hidden_dim, num_heads) for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens: torch.Tensor, sensor_tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            visual_tokens: [B, N_vis, hidden_dim] from MultiScaleVisionEncoder
            sensor_tokens: [B, 3, hidden_dim] from SensorEncoder
           Returns:
            [B, 8, hidden_dim] refined field-aware query embeddings
        """
        B = visual_tokens.shape[0]

        # Initialize queries with field-type embeddings
        queries = self.queries.unsqueeze(0).expand(B, -1, -1) + self.field_type_embed.unsqueeze(0)

        # Concatenate visual + sensor tokens as cross-attention key/value
        kv_tokens = torch.cat([visual_tokens, sensor_tokens], dim=1)  # [B, N_vis+3, D]

        for block in self.blocks:
            queries = block(queries, kv_tokens)

        return self.final_norm(queries)


class QFormerBlock(nn.Module):
    """One Q-Former block: Cross-Attn -> Self-Attn -> FFN (SwiGLU)."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)

        # SwiGLU FFN
        self.ffn_w1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn_w2 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn_w3 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.15)

    def forward(self, queries: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        # Cross-Attention: queries attend to visual+sensor tokens
        attn_out, _ = self.cross_attn(queries, kv_tokens, kv_tokens)
        queries = self.cross_norm(queries + self.dropout(attn_out))

        # Self-Attention: queries attend to each other
        attn_out, _ = self.self_attn(queries, queries, queries)
        queries = self.self_norm(queries + self.dropout(attn_out))

        # SwiGLU FFN
        gate = self.ffn_w1(queries)
        proj = self.ffn_w2(queries)
        ffn_out = self.ffn_w3(F.silu(gate) * proj)
        queries = self.ffn_norm(queries + self.dropout(ffn_out))

        return queries
