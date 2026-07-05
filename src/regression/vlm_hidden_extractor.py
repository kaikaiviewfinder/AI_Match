"""Extract VLM language-model hidden states during GeoCoT reasoning.

Instead of generating text with model.generate(), runs a single forward pass
and captures the last-layer hidden states. This gives us a dense vector
representation of the VLM's geographic reasoning — without the information
loss of compressing through text → parser → KB.

Architecture:
  Image + Prompt → model(**inputs, output_hidden_states=True)
    → hidden_states[-1] (last layer, all tokens)
    → mean-pool → (1536,) vector
"""

import torch
from PIL import Image
from typing import Optional


class VLMHiddenExtractor:
    """Extract pooled hidden states from VLM for a given image + prompt."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._device = model.device

    @torch.no_grad()
    def extract(self, image: Image.Image, prompt_text: str) -> torch.Tensor:
        """Run VLM forward pass and return mean-pooled last-layer hidden state.

        Returns tensor of shape (hidden_dim,) — (1536,) for Qwen2-VL-2B.
        """
        inputs = self._build_inputs(image, prompt_text)

        outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = last_hidden.mean(dim=1)

        return pooled.squeeze(0)  # (hidden_dim,)

    @torch.no_grad()
    def extract_3stage(
        self,
        image: Image.Image,
        macro_prompt: str,
        regional_prompt: str,
        local_prompt: str,
    ) -> dict[str, torch.Tensor]:
        """Extract hidden states for all 3 GeoCoT reasoning stages.

        Returns dict with keys: macro, regional, local, fused
        fused = mean of the three stage vectors.
        """
        h_macro = self.extract(image, macro_prompt)
        h_regional = self.extract(image, regional_prompt)
        h_local = self.extract(image, local_prompt)
        h_fused = (h_macro + h_regional + h_local) / 3.0

        return {
            "macro": h_macro,
            "regional": h_regional,
            "local": h_local,
            "fused": h_fused,
        }

    def _build_inputs(self, image: Image.Image, prompt_text: str) -> dict:
        """Build model inputs from image + prompt text.

        Mirrors preprocessing in create_qwen2vl_model_fn() but sets
        add_generation_prompt=False since we don't need the assistant prefix.
        """
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    @property
    def hidden_dim(self) -> int:
        return self.model.config.text_config.hidden_size
