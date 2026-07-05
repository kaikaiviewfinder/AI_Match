"""Extract visual features from Qwen2-VL / Qwen3-VL vision encoder for regression head.

Qwen2-VL-2B: ViT → merger → (num_patches/4, 1536)
Qwen3-VL-2B: ViT + DeepStack → merger → (num_patches/4, 2048)

We mean-pool over patches to get a fixed-dim vector per image.
"""

import torch
from PIL import Image
from typing import Optional


class VisualFeatureExtractor:
    """Extract pooled visual features from Qwen2-VL's vision encoder + merger."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._device = model.device

    @torch.no_grad()
    def extract(self, image: Image.Image, pool: str = "mean") -> torch.Tensor:
        """Extract pooled visual feature vector for a single image.

        Uses post-merger features which are normalized (LayerNorm) and
        projected to language-model space — much better conditioned for
        MLP training than raw vision features.

        Returns tensor of shape (1536,) on the model's device.
        """
        pixel_values, grid_thw = self._preprocess_image(image)

        # Raw vision encoder output
        vis_out = self.model.model.visual(
            pixel_values,
            grid_thw=grid_thw,
            output_hidden_states=False,
        )

        # Pass through merger for normalized features
        merged = self.model.model.visual.merger(vis_out.last_hidden_state)

        if pool == "mean":
            return merged.mean(dim=0)
        elif pool == "max":
            return merged.max(dim=0).values
        elif pool == "cls":
            return merged[0]
        else:
            raise ValueError(f"Unknown pool method: {pool}")

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image], pool: str = "mean") -> torch.Tensor:
        """Extract features for a batch of images. Returns (batch, 1536)."""
        features = []
        for img in images:
            feat = self.extract(img, pool=pool)
            features.append(feat)
        return torch.stack(features)

    def _preprocess_image(self, image: Image.Image) -> tuple:
        """Preprocess a single image for the vision encoder."""
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "."},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        return inputs["pixel_values"], inputs["image_grid_thw"]

    @property
    def feature_dim(self) -> int:
        return self.model.config.text_config.hidden_size
