"""GeoVLM: Edge-optimized geographic element extraction model.

Architecture:
  Multi-Scale ViT (224/448/672) + ToMe → Scale Fusion → [520, 512]
  Sensor Encoder: MLP(3→512) → [3, 512]
  Field-Aware Q-Former: 8 queries × 3 blocks → [8, 512]
  Constraint GNN: 8-node graph, 2-round message passing → [8, 512]
  Prediction Heads: 8 heads → structured elements dict

Output is compatible with src/regression/element_fusion.py downstream.
"""

from .model import GeoVLM, GeoVLMConfig
from .vision_encoder import MultiScaleVisionEncoder
from .sensor_encoder import SensorEncoder
from .qformer import FieldAwareQFormer
from .constraint_graph import ConstraintGraphLayer

__all__ = [
    "GeoVLM",
    "GeoVLMConfig",
    "MultiScaleVisionEncoder",
    "SensorEncoder",
    "FieldAwareQFormer",
    "ConstraintGraphLayer",
]
