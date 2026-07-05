"""GeoVLM: Complete edge-optimized geographic element extraction model.

Assembles: MultiScaleVisionEncoder -> FieldAwareQFormer -> ConstraintGraphLayer -> PredictionHeads
Output: elements dict compatible with src/regression/element_fusion.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from .vision_encoder import MultiScaleVisionEncoder, prepare_multi_scale_images
from .sensor_encoder import SensorEncoder
from .qformer import FieldAwareQFormer
from .constraint_graph import ConstraintGraphLayer


# Field vocabulary -- matches design spec Section 9
# Field order matches Q-Former query order and ConstraintGraph node indices:
#   0=climate, 1=terrain, 2=vegetation, 3=urbanization, 4=architecture,
#   5=pavement, 6=language, 7=visible_text,
#   8=soil_color, 9=sky_quality, 10=mountain_rock_type, 11=tree_species,
#   12=building_height, 13=water_type, 14=landform_detail, 15=scene_type
FIELD_VOCABS = {
    # ── Original 8 fields ──────────────────────────────────────────────
    "climate_zone": ["tropical", "subtropical", "temperate", "arid", "alpine", "boreal"],
    "terrain_type": ["urban_flat", "farmland_plain", "rolling_hills", "sharp_mountains",
                     "karst_peaks", "sandstone_pillars", "desert_dunes", "grassland_steppe", "plateau"],
    "vegetation_zone": ["tropical_rainforest", "broadleaf_evergreen", "broadleaf_deciduous",
                        "conifer_forest", "mixed_forest", "alpine_meadow", "desert_scrub",
                        "grassland", "bamboo_forest", "cropland", "sparse"],
    "urbanization": ["metropolis", "medium_city", "small_town", "village", "rural", "wilderness"],
    "architecture_style": ["modern_glass", "modern_residential", "old_residential", "hui_style",
                           "tibetan_stone", "courtyard", "arcade", "stilt_house", "tulou",
                           "shikumen", "traditional_official", "soviet_industrial", "none_visible"],
    "pavement_type": ["red_brick_tiles", "grey_concrete", "asphalt", "natural", "not_visible"],
    "language_script": ["simplified_chinese", "traditional_chinese", "tibetan", "uyghur_arabic",
                        "mongolian", "bilingual_cn_en", "none_visible"],
    "visible_text": ["local_government", "local_phone_code", "local_license_plate",
                     "local_street_sign", "cross_region_shop", "national_chain",
                     "national_slogan", "none_visible"],
    # ── New 8 fields for spatial narrowing ─────────────────────────────
    "soil_color": ["red", "yellow", "brown", "black", "grey", "loess_yellow", "white"],
    "sky_quality": ["clear_blue", "grey_hazy", "thick_fog", "dusty_yellow"],
    "mountain_rock_type": ["granite_spheroidal", "quartz_sandstone_pillars",
                           "limestone_karst", "red_sandstone_danxia",
                           "snow_peaks_glaciers", "alpine_lakes", "none_visible"],
    "tree_species": ["banyan", "camphor", "plane_tree", "gingko", "poplar",
                     "coconut_palm", "chinese_red_pine", "bamboo", "none_visible"],
    "building_height": ["super_tall", "high_rise", "mid_rise", "low_rise", "none_visible"],
    "water_type": ["ocean", "lake", "glacier", "waterfall", "river", "stream", "none_visible"],
    "landform_detail": ["limestone_tower_karst_scattered", "limestone_tower_karst_dense",
                        "granite_spheroidal_rounded", "granite_spheroidal_sharp",
                        "quartz_sandstone_pillars", "red_sandstone_danxia",
                        "snow_peaks_glaciers", "alpine_meadow", "plateau_plain",
                        "river_canyon", "granite_boulder_field", "coastal_shore",
                        "urban_flat", "none_visible"],
    "scene_type": ["desert", "grassland", "coastal", "river_valley", "city_street",
                   "mountain_trail_karst", "mountain_trail_granite", "mountain_trail_snow",
                   "rural_village", "tourist_scenic", "wilderness_forest", "none_visible"],
}


@dataclass
class GeoVLMConfig:
    """GeoVLM model configuration."""
    vit_model_name: str = "google/vit-base-patch16-224"
    hidden_dim: int = 512
    qformer_num_blocks: int = 3
    qformer_num_heads: int = 8
    tome_r_mid: int = 13       # 784->314
    tome_r_fine: int = 23      # 1764->530
    sensor_noise_std: float = 0.0  # >0 for Stage 3 training
    dropout: float = 0.1
    refinement_threshold: float = 0.5  # confidence below this -> progressive refinement


class PredictionHeads(nn.Module):
    """8 classification heads. Elevation comes from sensor data, not prediction."""

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Classification heads: one per categorical field
        self.heads = nn.ModuleDict()
        for field_name, vocab in FIELD_VOCABS.items():
            self.heads[field_name] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, len(vocab)),
            )

        # Field-type shortcut: additive connection preserved from input
        self.field_shortcut = nn.Parameter(torch.randn(len(FIELD_VOCABS), hidden_dim) * 0.02)

    def forward(self, field_features: torch.Tensor) -> dict:
        """Args: field_features [B, 8, hidden_dim] from ConstraintGraphLayer
           Returns: dict with 'logits', 'probs', 'confidence'
        """
        B = field_features.shape[0]

        # Add field-type shortcut connection
        field_features = field_features + self.field_shortcut.unsqueeze(0)

        logits = {}
        probs = {}
        confidence = {}

        for i, (field_name, head) in enumerate(self.heads.items()):
            feat = field_features[:, i, :]  # [B, hidden_dim]
            raw = head(feat)                # [B, vocab_size]
            p = F.softmax(raw, dim=-1)
            logits[field_name] = raw
            probs[field_name] = p
            # Confidence = 1 - normalized entropy
            entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
            max_entropy = -torch.log(torch.tensor(1.0 / p.shape[-1], device=p.device))
            confidence[field_name] = 1.0 - entropy / max_entropy

        return {
            "logits": logits,
            "probs": probs,
            "confidence": confidence,
        }


class GeoVLM(nn.Module):
    """GeoVLM: Edge-optimized geographic element extraction model.

    Total: ~430M parameters, <600ms inference on DK-2500.
    Output: elements dict (compatible with fuse_elements_v3).
    """

    def __init__(self, config: Optional[GeoVLMConfig] = None):
        super().__init__()
        self.config = config or GeoVLMConfig()

        self.vision_encoder = MultiScaleVisionEncoder(
            vit_model_name=self.config.vit_model_name,
            hidden_dim=self.config.hidden_dim,
            tome_r_mid=self.config.tome_r_mid,
            tome_r_fine=self.config.tome_r_fine,
        )
        self.sensor_encoder = SensorEncoder(
            hidden_dim=self.config.hidden_dim,
            noise_std=self.config.sensor_noise_std,
        )
        self.qformer = FieldAwareQFormer(
            hidden_dim=self.config.hidden_dim,
            num_blocks=self.config.qformer_num_blocks,
            num_heads=self.config.qformer_num_heads,
        )
        self.constraint_graph = ConstraintGraphLayer(
            hidden_dim=self.config.hidden_dim,
        )
        self.heads = PredictionHeads(
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(self, images: list, sensor_values: torch.Tensor) -> dict:
        """Full forward pass.

        Args:
            images: list of 3 PIL Images at [224^2, 448^2, 672^2]
            sensor_values: [B, 3] -- [elevation_m, temperature_c, humidity_pct]
        Returns:
            dict with 'elements', 'confidence', 'field_consistency', 'raw'
        """
        # Vision: multi-scale -> unified tokens
        visual_tokens = self.vision_encoder(images)  # [B, ~1040, 512]

        # Sensor: physical -> sensor tokens
        sensor_tokens = self.sensor_encoder(sensor_values)  # [B, 3, 512]

        # Q-Former: 8 queries extract field-specific info
        field_features = self.qformer(visual_tokens, sensor_tokens)  # [B, 8, 512]

        # Constraint GNN: enforce physical consistency
        field_features, field_consistency = self.constraint_graph(field_features)

        # Prediction heads: field features -> structured output
        head_outputs = self.heads(field_features)

        # Build elements dict from head outputs + sensor elevation
        elements = self._build_elements_dict(head_outputs, sensor_values)

        return {
            "elements": elements,
            "confidence": head_outputs["confidence"],
            "field_consistency": field_consistency,
            "raw": head_outputs,
        }

    def _build_elements_dict(self, head_outputs: dict, sensor_values: torch.Tensor) -> dict:
        """Convert head outputs to elements dict. Elevation from sensor, not prediction."""
        elements = {}
        probs = head_outputs["probs"]

        for field_name in FIELD_VOCABS:
            vocab = FIELD_VOCABS[field_name]
            p = probs[field_name][0]  # [vocab_size] — first batch item
            best_idx = p.argmax().item()
            best_value = vocab[best_idx]
            best_prob = p[best_idx].item()

            elements[f"{field_name}_pred"] = best_value
            elements[f"{field_name}_confidence"] = round(best_prob, 3)
            # Top-3 alternatives
            top3 = p.topk(min(3, len(vocab)))
            elements[f"{field_name}_top3"] = [
                (vocab[idx.item()], round(prob.item(), 3))
                for idx, prob in zip(top3.indices, top3.values)
            ]

        # Elevation: use sensor data directly (more accurate than model prediction)
        elev_m = sensor_values[0, 0].item()
        elements["elevation_estimate_m"] = [round(elev_m), round(elev_m)]

        # Ruled-out features (low-confidence predictions)
        confidence = head_outputs["confidence"]
        ruled_out = []
        for field_name in FIELD_VOCABS:
            if confidence[field_name][0].item() < 0.3:
                ruled_out.append(field_name)
        elements["ruled_out_features"] = ruled_out

        return elements

    @torch.no_grad()
    def predict(self, image: "PIL.Image.Image",
                sensor_elevation_m: Optional[float] = None,
                sensor_temperature_c: Optional[float] = None,
                sensor_humidity_pct: Optional[float] = None) -> dict:
        """Convenience method: single-image inference -> elements dict.

        Args:
            image: PIL RGB image (any resolution)
            sensor_*: optional sensor readings (defaults provided if None)
        Returns:
            elements dict ready for fuse_elements_v3()
        """
        self.eval()
        scales = prepare_multi_scale_images(image)
        sensor = torch.tensor([[
            sensor_elevation_m if sensor_elevation_m is not None else 500.0,
            sensor_temperature_c if sensor_temperature_c is not None else 20.0,
            sensor_humidity_pct if sensor_humidity_pct is not None else 60.0,
        ]], dtype=torch.float32)

        device = next(self.parameters()).device
        output = self.forward(scales, sensor.to(device))
        return output["elements"]

    def count_parameters(self) -> dict:
        """Count parameters per component."""
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "vision_encoder": _count(self.vision_encoder),
            "sensor_encoder": _count(self.sensor_encoder),
            "qformer": _count(self.qformer),
            "constraint_graph": _count(self.constraint_graph),
            "heads": _count(self.heads),
            "total": _count(self),
        }
