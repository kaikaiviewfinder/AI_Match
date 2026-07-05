"""Cascade Pipeline: GeoCoT spatial narrowing → constrained MoE regression.

Orchestrates the Plan B cascade architecture:
  1. GeoCoT 3-stage reasoning (macro → regional → local)
  2. StructuredParser → GeoEntities per stage
  3. SpatialKBEngine → ScoredRegion candidates per stage
  4. SpatialConstraintFuser → fused ConstraintBox
  5. Constrained MoE → spatially-aware coordinate regression
  6. GeoExplanation → structured reasoning report

This gives the best of both worlds:
  - GeoCoT does what it's good at: reading the scene, narrowing the search area
  - MoE does what it's good at: precise numeric regression within constrained region
"""

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Ensure src is on path
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from Geocot.Geocot import (
    GeoCoTPipeline, GeoCoTResult, GeoPrediction, extract_prediction,
    GeoCoTStage,
)
from Geocot.structured_parser import parse_geocot_output, GeoEntities
from Geocot.spatial_kb import entities_to_spatial_regions, region_to_constraint_box
from regression.spatial_constraint import (
    ConstraintBox, SpatialConstraintFuser, constraint_weighted_fusion,
)
from regression.moe_config import SceneType, SCENE_TYPE_LABELS_CN


@dataclass
class CascadeResult:
    """Full cascade pipeline output."""
    # GeoCoT output
    geocot: GeoCoTResult = field(default_factory=GeoCoTResult)

    # Parsed entities per stage
    macro_entities: Optional[GeoEntities] = None
    regional_entities: Optional[GeoEntities] = None
    local_entities: Optional[GeoEntities] = None

    # Spatial regions per stage
    macro_regions: list = field(default_factory=list)
    regional_regions: list = field(default_factory=list)
    local_regions: list = field(default_factory=list)

    # Fused constraint
    constraint: Optional[ConstraintBox] = None

    # Final coordinates
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # MoE details
    moe_expert_weights: Optional[torch.Tensor] = None
    moe_expert_lats: Optional[torch.Tensor] = None
    moe_expert_lngs: Optional[torch.Tensor] = None

    # Explanation
    explanation: object = None  # GeoExplanation

    # Metadata
    cascade_info: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def spatial_narrowing_ratio(self) -> float:
        """How much the constraint narrowed from China default (0-1)."""
        if self.constraint is None:
            return 0.0
        china_area = 9_600_000  # km²
        constraint_area = self.constraint.area_km2
        if constraint_area >= china_area:
            return 0.0
        return 1.0 - (constraint_area / china_area)

    @property
    def constraint_radius_km(self) -> float:
        if self.constraint is None:
            return float("inf")
        return self.constraint.radius_km


class CascadePipeline:
    """GeoCoT → Parser → SpatialKB → Constraint → MoE → Report.

    Usage:
        pipeline = CascadePipeline(geocot_pipeline, moe_model, feature_extractor)
        result = pipeline.run(image)
        print(result.explanation.to_text())
    """

    def __init__(
        self,
        geocot_pipeline: GeoCoTPipeline,
        moe_model,
        feature_extractor,
        fuser_config: Optional[dict] = None,
    ):
        self._geocot = geocot_pipeline
        self._moe = moe_model
        self._extractor = feature_extractor
        self._device = next(moe_model.parameters()).device

        fuser_config = fuser_config or {}
        self._fuser = SpatialConstraintFuser(
            macro_radius_km=fuser_config.get("macro_radius_km", 1500.0),
            regional_radius_km=fuser_config.get("regional_radius_km", 200.0),
            local_radius_km=fuser_config.get("local_radius_km", 50.0),
        )

    @torch.no_grad()
    def run(self, image: Image.Image) -> CascadeResult:
        """Run the full cascade pipeline on a single image.

        Args:
            image: PIL RGB image

        Returns:
            CascadeResult with coordinates, constraints, and explanation
        """
        result = CascadeResult()

        try:
            # ── Stage 1: GeoCoT 3-stage reasoning ─────────────────────
            result.geocot = self._geocot.run(image)

            # ── Stage 2: Parse each stage into structured entities ─────
            stage_outputs = result.geocot.stage_outputs

            # Parse MACRO stage
            if "macro" in stage_outputs:
                result.macro_entities = parse_geocot_output(
                    {"macro": stage_outputs["macro"]}
                )

            # Parse MACRO+REGIONAL for regional context
            regional_inputs = {}
            if "macro" in stage_outputs:
                regional_inputs["macro"] = stage_outputs["macro"]
            if "regional" in stage_outputs:
                regional_inputs["regional"] = stage_outputs["regional"]
            if regional_inputs:
                result.regional_entities = parse_geocot_output(regional_inputs)

            # Parse ALL stages for local precision
            local_inputs = {k: v for k, v in stage_outputs.items()
                           if k in ("macro", "regional", "local")}
            if local_inputs:
                result.local_entities = parse_geocot_output(local_inputs)

            # ── Stage 3: Entities → Spatial regions ───────────────────
            if result.macro_entities and not result.macro_entities.is_empty:
                result.macro_regions = entities_to_spatial_regions(
                    result.macro_entities, top_k=3, max_radius_km=1500.0,
                )

            if result.regional_entities and not result.regional_entities.is_empty:
                result.regional_regions = entities_to_spatial_regions(
                    result.regional_entities, top_k=3, max_radius_km=300.0,
                )

            if result.local_entities and not result.local_entities.is_empty:
                result.local_regions = entities_to_spatial_regions(
                    result.local_entities, top_k=3, max_radius_km=100.0,
                )

            # ── Stage 4: Fuse constraints ─────────────────────────────
            result.constraint = self._fuser.fuse_from_stages(
                macro_regions=result.macro_regions,
                regional_regions=result.regional_regions,
                local_regions=result.local_regions,
            )

            # ── Stage 5: Constrained MoE regression ───────────────────
            features = self._extractor.extract(image).float()

            # Only enable constraint if GeoCoT identified a specific named location
            has_named = (result.local_entities is not None
                         and len(result.local_entities.named_locations) > 0)

            lat, lng, weights, expert_lats, expert_lngs = self._moe.predict_constrained(
                features.unsqueeze(0),
                constraint_box=result.constraint,
                blend_strength=0.15,
                return_expert_info=True,
                has_named_location=has_named,
            )

            result.latitude = lat.item()
            result.longitude = lng.item()
            result.moe_expert_weights = weights.squeeze(0)
            result.moe_expert_lats = expert_lats.squeeze(0)
            result.moe_expert_lngs = expert_lngs.squeeze(0)

            # ── Stage 6: Build explanation report ─────────────────────
            self._build_explanation(result)

            # ── Record cascade metadata ───────────────────────────────
            result.cascade_info = {
                "narrowing_ratio": round(result.spatial_narrowing_ratio, 4),
                "constraint_radius_km": round(result.constraint_radius_km, 1),
                "constraint_confidence": round(result.constraint.confidence, 4) if result.constraint else 0,
                "constraint_source": result.constraint.source if result.constraint else "none",
                "macro_entities_count": len(result.macro_entities.named_locations) if result.macro_entities else 0,
                "regional_entities_count": len(result.regional_entities.named_locations) if result.regional_entities else 0,
                "local_entities_count": len(result.local_entities.named_locations) if result.local_entities else 0,
                "macro_regions_count": len(result.macro_regions),
                "regional_regions_count": len(result.regional_regions),
                "local_regions_count": len(result.local_regions),
            }

            # Also set coordinates on GeoCoT result for compatibility
            if result.latitude is not None:
                result.geocot.final_prediction.latitude = result.latitude
                result.geocot.final_prediction.longitude = result.longitude

        except Exception as e:
            result.error = str(e)
            import traceback
            result.cascade_info["traceback"] = traceback.format_exc()

        return result

    def _build_explanation(self, result: CascadeResult):
        """Build GeoExplanation from cascade results, with spatial narrowing info."""
        from regression.explanation import GeoExplanation, Confidence
        from Geocot.Geocot import _coords_to_nearest_city, _infer_continent

        explanation = GeoExplanation()

        # KB reverse lookup
        kb_city, kb_country = _coords_to_nearest_city(
            result.latitude, result.longitude
        )

        # GeoCoT text for summary
        geocot_text = "\n".join(
            v for k, v in result.geocot.stage_outputs.items()
            if k in ("macro", "regional", "local")
        )

        explanation.build_from_moe(
            router_weights=result.moe_expert_weights,
            expert_lats=result.moe_expert_lats,
            expert_lngs=result.moe_expert_lngs,
            fused_lat=result.latitude,
            fused_lng=result.longitude,
            geocot_city=kb_city or result.geocot.final_prediction.city,
            geocot_country=kb_country or result.geocot.final_prediction.country,
            geocot_continent=_infer_continent(kb_country or "China"),
            geocot_output=geocot_text,
            geocot_scene_type=result.geocot.scene_type,
        )

        # Augment reconciliation with cascade-specific dimensions
        if result.constraint is not None and not result.constraint.is_default:
            cascade_items = [
                {
                    "dimension": "空间收窄",
                    "detail": (
                        f"级联约束将候选区域从全中国(~960万km²)收窄至"
                        f"~{result.constraint.radius_km:.0f}km半径范围"
                        f"(置信度{result.constraint.confidence:.1%})"
                    ),
                    "agree": result.constraint.confidence > 0.3,
                },
                {
                    "dimension": "约束来源",
                    "detail": f"来源: {result.constraint.source}",
                    "agree": result.constraint.confidence > 0.3,
                },
            ]
            explanation.reconciliation["items"].extend(cascade_items)
            explanation.reconciliation["cascade"] = result.cascade_info

        # Override confidence with cascade-aware computation
        if result.constraint is not None and result.constraint.confidence > 0.5:
            # Boost confidence when spatial narrowing succeeded
            if explanation.confidence_stars < Confidence.HIGH:
                explanation.confidence_stars = Confidence.HIGH
                explanation.confidence_reason = (
                    f"级联空间收窄成功(constraint置信度{result.constraint.confidence:.1%})，"
                    f"提升可信度至{explanation.confidence_stars}/5"
                )

        result.explanation = explanation
