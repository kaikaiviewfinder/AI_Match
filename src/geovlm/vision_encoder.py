"""Multi-scale vision encoder with Token Merging (ToMe).

Three resolutions (224^2, 448^2, 672^2) share a single ViT backbone.
ToMe compresses mid/fine scale tokens by 60-70%.
Scale Fusion combines all three via learnable scale embeddings + light self-attention.

Key design decisions:
  - Shared weights across scales (same eyes, different distances) — 300M not 900M
  - ToMe uses greedy similarity-based pair merging
  - Scale Fusion uses 1-layer self-attention for efficiency
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor


def _create_local_processor():
    """Create ViTImageProcessor without network access."""
    return ViTImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        resample=3,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


class MultiScaleVisionEncoder(nn.Module):
    """Multi-scale ViT with Token Merging and Scale Fusion.

    Scales:
      224^2 → 196 tokens (coarse: overall landscape, climate, urbanization)
      448^2 → 784→314 tokens via ToMe (mid: building structure, vegetation type)
      672^2 → 1764→530 tokens via ToMe (fine: leaf texture, rock type, pavement)

    Output: [B, ~1040, 512] unified visual features (before optional compression)
    """

    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        hidden_dim: int = 512,
        tome_r_coarse: int = 0,     # no merging at 224^2
        tome_r_mid: int = 13,        # 784→314 (60% reduction)
        tome_r_fine: int = 23,       # 1764→530 (70% reduction)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared ViT backbone
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden = self.vit.config.hidden_size  # typically 768

        # Project ViT features to hidden_dim
        self.vit_proj = nn.Linear(self.vit_hidden, hidden_dim)

        # ToMe parameters
        self.tome_r_coarse = tome_r_coarse
        self.tome_r_mid = tome_r_mid
        self.tome_r_fine = tome_r_fine

        # Scale embeddings — one per resolution
        self.scale_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)

        # Scale Fusion: 1-layer self-attention + FFN
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.fusion_norm2 = nn.LayerNorm(hidden_dim)

        # Processor for multi-scale inputs — hardcoded to avoid HF network access
        self.processor = _create_local_processor()

    def _vit_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run ViT and return patch tokens (exclude CLS). [B, N, hidden_dim]"""
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        tokens = outputs.last_hidden_state[:, 1:, :]  # [B, N, 768] — drop CLS
        return self.vit_proj(tokens)  # [B, N, hidden_dim]

    def _tome_merge(self, tokens: torch.Tensor, r: int) -> torch.Tensor:
        """Token Merging: greedily merge r most-similar token pairs via averaging.

        Selects non-overlapping pairs (each token used at most once) so output size
        is uniform across batch elements: N - r_actual tokens per sample.
        """
        if r <= 0:
            return tokens

        B, N, D = tokens.shape
        r_actual = min(r, N // 2)
        if r_actual == 0:
            return tokens

        with torch.no_grad():
            sim = tokens @ tokens.transpose(-1, -2)  # [B, N, N]

        results = []
        for b in range(B):
            sim_b = sim[b].clone()  # [N, N]
            # Mask diagonal and lower triangle
            mask = torch.triu(torch.ones(N, N, device=tokens.device), diagonal=1).bool()
            sim_b.masked_fill_(~mask, -float("inf"))

            used = set()
            pairs = []
            for _ in range(r_actual):
                max_idx = sim_b.view(-1).argmax().item()
                i, j = max_idx // N, max_idx % N
                if sim_b[i, j].item() <= -1e9:
                    break
                pairs.append((i, j))
                # Mask all pairs involving i or j so they can't be reused
                sim_b[i, :] = -float("inf")
                sim_b[:, i] = -float("inf")
                sim_b[j, :] = -float("inf")
                sim_b[:, j] = -float("inf")

            r_b = len(pairs)
            if r_b == 0:
                results.append(tokens[b:b+1])
                continue

            keep_idx = [i for i in range(N) if i not in used]
            keep_t = tokens[b:b+1, keep_idx, :]                               # [1, N-2r_b, D]
            merged_t = torch.stack([
                (tokens[b, i] + tokens[b, j]) / 2 for i, j in pairs
            ]).unsqueeze(0)                                                    # [1, r_b, D]
            out_b = torch.cat([keep_t, merged_t], dim=1)                       # [1, N-r_b, D]

            # Pad to target_len = N - r_actual (uniform across batch)
            target_len = N - r_actual
            if out_b.shape[1] < target_len:
                pad = torch.zeros(1, target_len - out_b.shape[1], D,
                                  device=tokens.device, dtype=tokens.dtype)
                out_b = torch.cat([out_b, pad], dim=1)
            results.append(out_b)

        return torch.cat(results, dim=0)  # [B, N - r_actual, D]

    def forward(self, images: list) -> torch.Tensor:
        """Args: images — list of 3 PIL Images at [224^2, 448^2, 672^2]
           Returns: [B, ~1040, hidden_dim] fused visual features
        """
        # Process each scale through ViTImageProcessor
        inputs_coarse = self.processor(images=images[0], return_tensors="pt")
        inputs_mid = self.processor(images=images[1], return_tensors="pt")
        inputs_fine = self.processor(images=images[2], return_tensors="pt")

        device = self.vit_proj.weight.device
        tok_coarse = self._vit_forward(inputs_coarse.pixel_values.to(device))
        tok_mid = self._vit_forward(inputs_mid.pixel_values.to(device))
        tok_fine = self._vit_forward(inputs_fine.pixel_values.to(device))

        # Token Merging at each scale
        tok_coarse = self._tome_merge(tok_coarse, self.tome_r_coarse)  # 196 tok
        tok_mid = self._tome_merge(tok_mid, self.tome_r_mid)            # ~314 tok
        tok_fine = self._tome_merge(tok_fine, self.tome_r_fine)         # ~530 tok

        # Add scale embeddings
        tok_coarse = tok_coarse + self.scale_embed[0]
        tok_mid = tok_mid + self.scale_embed[1]
        tok_fine = tok_fine + self.scale_embed[2]

        # Concatenate all scales
        all_tokens = torch.cat([tok_coarse, tok_mid, tok_fine], dim=1)

        # Scale Fusion via self-attention + FFN (with residual)
        attn_out, _ = self.fusion_attn(all_tokens, all_tokens, all_tokens)
        all_tokens = self.fusion_norm(all_tokens + attn_out)
        ffn_out = self.fusion_ffn(all_tokens)
        all_tokens = self.fusion_norm2(all_tokens + ffn_out)

        return all_tokens


def prepare_multi_scale_images(image: "PIL.Image.Image") -> list:
    """Convert a single PIL image into 3 scales for the vision encoder.

    Returns: [img_224, img_448, img_672] — all RGB PIL Images
    """
    img_224 = image.resize((224, 224))
    img_448 = image.resize((448, 448))
    img_672 = image.resize((672, 672))
    return [img_224, img_448, img_672]
