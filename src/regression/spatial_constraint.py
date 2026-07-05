"""Spatial constraint layer: fuse multi-level constraints for regression.

ConstraintBox: bounding box + center + radius with confidence.
SpatialConstraintFuser: merge MACRO/REGIONAL/LOCAL constraints into
  a single narrowed ConstraintBox for constrained MoE regression.

Prior loss: soft penalty for predictions outside the constraint region.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ConstraintBox:
    """A spatial constraint bounding box with confidence metadata.

    All coordinates in degrees. radius_km is the approximate 1-sigma radius.
    """
    lat_min: float = 18.0
    lat_max: float = 54.0       # China default
    lng_min: float = 73.0
    lng_max: float = 135.0      # China default
    center_lat: float = 35.0
    center_lng: float = 105.0
    radius_km: float = 2000.0   # default: all of China
    confidence: float = 0.0     # 0.0-1.0
    source: str = "default"
    stage: str = ""             # "macro" | "regional" | "local"

    @classmethod
    def china_default(cls) -> "ConstraintBox":
        """Bounding box covering mainland China."""
        return cls(
            lat_min=18.0, lat_max=54.0,
            lng_min=73.0, lng_max=135.0,
            center_lat=36.0, center_lng=104.0,
            radius_km=2200.0,
            confidence=0.0,
            source="china_default",
            stage="macro",
        )

    @classmethod
    def from_center_radius(
        cls,
        center_lat: float,
        center_lng: float,
        radius_km: float,
        confidence: float = 0.5,
        source: str = "",
        stage: str = "",
    ) -> "ConstraintBox":
        """Create from center + radius in km."""
        deg_lat = radius_km / 111.0
        deg_lng = radius_km / (111.0 * math.cos(math.radians(center_lat)))
        return cls(
            lat_min=center_lat - deg_lat,
            lat_max=center_lat + deg_lat,
            lng_min=center_lng - deg_lng,
            lng_max=center_lng + deg_lng,
            center_lat=center_lat,
            center_lng=center_lng,
            radius_km=radius_km,
            confidence=confidence,
            source=source,
            stage=stage,
        )

    @classmethod
    def from_scored_regions(
        cls,
        regions: list,
        stage: str = "",
    ) -> Optional["ConstraintBox"]:
        """Create a ConstraintBox from a list of ScoredRegion objects.

        If multiple regions, computes weighted average center and
        uses the best region's radius. Returns None if no regions.
        """
        if not regions:
            return None

        if len(regions) == 1:
            r = regions[0]
            return cls.from_center_radius(
                center_lat=r.center_lat,
                center_lng=r.center_lng,
                radius_km=r.radius_km,
                confidence=r.confidence,
                source=r.source,
                stage=stage,
            )

        # Weighted average by confidence
        total_conf = sum(r.confidence for r in regions)
        if total_conf == 0:
            total_conf = len(regions)

        weights = [r.confidence / total_conf for r in regions]
        avg_lat = sum(w * r.center_lat for w, r in zip(weights, regions))
        avg_lng = sum(w * r.center_lng for w, r in zip(weights, regions))

        # Best confidence region's radius (most specific)
        best = max(regions, key=lambda r: r.confidence)
        avg_conf = sum(r.confidence for r in regions) / len(regions)
        # Boost confidence when multiple regions agree
        if len(regions) >= 2:
            avg_conf = min(1.0, avg_conf + 0.15)

        return cls.from_center_radius(
            center_lat=avg_lat,
            center_lng=avg_lng,
            radius_km=best.radius_km,
            confidence=avg_conf,
            source=" + ".join(r.source for r in regions[:3]),
            stage=stage,
        )

    @property
    def is_default(self) -> bool:
        return self.source == "default" or self.source == "china_default"

    @property
    def lat_span(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def lng_span(self) -> float:
        return self.lng_max - self.lng_min

    @property
    def area_km2(self) -> float:
        """Approximate area in km²."""
        avg_lat_rad = math.radians(self.center_lat)
        dlat_km = self.lat_span * 111.0
        dlng_km = self.lng_span * 111.0 * math.cos(avg_lat_rad)
        return dlat_km * dlng_km

    def to_dict(self) -> dict:
        return {
            "lat_min": self.lat_min, "lat_max": self.lat_max,
            "lng_min": self.lng_min, "lng_max": self.lng_max,
            "center_lat": self.center_lat, "center_lng": self.center_lng,
            "radius_km": self.radius_km, "confidence": self.confidence,
            "source": self.source, "stage": self.stage,
        }


class SpatialConstraintFuser:
    """Fuse multi-stage spatial constraints (MACRO → REGIONAL → LOCAL).

    Strategy:
      - MACRO: broad continent-scale (~500km) — default to China bounds
      - REGIONAL: province/cluster scale (~50km)
      - LOCAL: specific landmark scale (~5km)
      - Layers are merged via intersection when they agree,
        union when they disagree (weighted by confidence).
    """

    def __init__(
        self,
        macro_radius_km: float = 1500.0,
        regional_radius_km: float = 200.0,
        local_radius_km: float = 50.0,
    ):
        self.macro_radius = macro_radius_km
        self.regional_radius = regional_radius_km
        self.local_radius = local_radius_km

    def fuse(
        self,
        macro: Optional[ConstraintBox] = None,
        regional: Optional[ConstraintBox] = None,
        local: Optional[ConstraintBox] = None,
    ) -> ConstraintBox:
        """Fuse MACRO/REGIONAL/LOCAL constraints into a single box.

        Priority: LOCAL > REGIONAL > MACRO > default China.
        When higher-resolution constraint exists, narrow to it.
        When layers disagree, the more confident layer wins.
        """
        # Start from the most specific constraint available
        if local is not None and local.confidence > 0.3:
            base = local
        elif regional is not None and regional.confidence > 0.3:
            base = regional
        else:
            base = macro or ConstraintBox.china_default()

        # Intersect with coarser constraints to validate
        result = base
        result.stage = "fused"

        # Intersection with regional (if both exist and base is local)
        if local is not None and regional is not None:
            result = self._intersect_or_fallback(base, regional, "local", "regional")

        # Intersection with macro
        if macro is not None:
            result = self._intersect_or_fallback(result, macro, result.stage, "macro")

        # Clamp to China bounds
        result.lat_min = max(18.0, result.lat_min)
        result.lat_max = min(54.0, result.lat_max)
        result.lng_min = max(73.0, result.lng_min)
        result.lng_max = min(135.0, result.lng_max)

        # Recompute center and radius
        result.center_lat = (result.lat_min + result.lat_max) / 2
        result.center_lng = (result.lng_min + result.lng_max) / 2
        avg_lat_rad = math.radians(result.center_lat)
        result.radius_km = (
            (result.lat_span * 111.0) ** 2 +
            (result.lng_span * 111.0 * math.cos(avg_lat_rad)) ** 2
        ) ** 0.5 / 2

        return result

    def fuse_from_stages(
        self,
        macro_regions: Optional[list] = None,
        regional_regions: Optional[list] = None,
        local_regions: Optional[list] = None,
    ) -> ConstraintBox:
        """Convenience: fuse from ScoredRegion lists per stage.

        Args:
            macro_regions: ScoredRegion list from MACRO stage parse
            regional_regions: ScoredRegion list from REGIONAL stage parse
            local_regions: ScoredRegion list from LOCAL stage parse
        """
        macro = ConstraintBox.from_scored_regions(macro_regions or [], "macro")
        regional = ConstraintBox.from_scored_regions(regional_regions or [], "regional")
        local = ConstraintBox.from_scored_regions(local_regions or [], "local")

        # Apply default radii scaling per stage
        if macro and macro.source != "china_default":
            macro.radius_km = min(self.macro_radius, macro.radius_km)
        if regional:
            regional.radius_km = min(self.regional_radius, regional.radius_km)
        if local:
            local.radius_km = min(self.local_radius, local.radius_km)

        return self.fuse(macro, regional, local)

    @staticmethod
    def _intersect_or_fallback(
        primary: ConstraintBox,
        secondary: ConstraintBox,
        primary_name: str,
        secondary_name: str,
    ) -> ConstraintBox:
        """Intersect two constraint boxes. If no overlap, prefer the more confident one."""
        # Check for overlap
        lat_overlap = (primary.lat_min < secondary.lat_max and
                       primary.lat_max > secondary.lat_min)
        lng_overlap = (primary.lng_min < secondary.lng_max and
                       primary.lng_max > secondary.lng_min)

        if lat_overlap and lng_overlap:
            # Intersection
            intersected = ConstraintBox(
                lat_min=max(primary.lat_min, secondary.lat_min),
                lat_max=min(primary.lat_max, secondary.lat_max),
                lng_min=max(primary.lng_min, secondary.lng_min),
                lng_max=min(primary.lng_max, secondary.lng_max),
                center_lat=0, center_lng=0, radius_km=0, confidence=0,
                source=f"{primary_name} ∩ {secondary_name}",
                stage="fused",
            )
            intersected.center_lat = (intersected.lat_min + intersected.lat_max) / 2
            intersected.center_lng = (intersected.lng_min + intersected.lng_max) / 2
            avg_lat_rad = math.radians(intersected.center_lat)
            intersected.radius_km = (
                (intersected.lat_span * 111.0) ** 2 +
                (intersected.lng_span * 111.0 * math.cos(avg_lat_rad)) ** 2
            ) ** 0.5 / 2
            intersected.confidence = max(primary.confidence, secondary.confidence)
            return intersected

        # No overlap — prefer the more confident layer
        if primary.confidence >= secondary.confidence:
            return primary
        return secondary


def prior_loss(
    pred_normed: torch.Tensor,       # (batch, 2) — normalized [lat/90, lng/180]
    constraint: ConstraintBox,
    penalty_weight: float = 1.0,
    margin_km: float = 50.0,         # no penalty within this margin
) -> torch.Tensor:
    """Compute soft spatial prior loss: penalize predictions outside constraint box.

    Uses a squared-penalty (soft) approach — predictions far outside the box
    get heavier penalty, but predictions just outside get light penalty.

    Args:
        pred_normed: (batch, 2) tensor, normalized as [lat/90, lng/180]
        constraint: ConstraintBox defining the allowed region
        penalty_weight: overall weight multiplier for this loss
        margin_km: no penalty if within this distance of the constraint boundary

    Returns:
        scalar loss tensor (mean over batch)
    """
    # Denormalize to degrees
    lat = pred_normed[:, 0] * 90.0   # (batch,)
    lng = pred_normed[:, 1] * 180.0  # (batch,)

    # Normalize constraint bounds to the same scale as pred_normed
    lat_min_norm = constraint.lat_min / 90.0
    lat_max_norm = constraint.lat_max / 90.0
    lng_min_norm = constraint.lng_min / 180.0
    lng_max_norm = constraint.lng_max / 180.0

    # Margin in normalized units
    margin_lat = margin_km / 111.0 / 90.0
    margin_lng = margin_km / (111.0 * math.cos(math.radians(constraint.center_lat))) / 180.0

    # Soft boundary penalty: quadratic outside, zero inside
    lat_low_violation = torch.relu(lat_min_norm - margin_lat - pred_normed[:, 0])
    lat_high_violation = torch.relu(pred_normed[:, 0] - lat_max_norm - margin_lat)
    lng_low_violation = torch.relu(lng_min_norm - margin_lng - pred_normed[:, 1])
    lng_high_violation = torch.relu(pred_normed[:, 1] - lng_max_norm - margin_lng)

    # Quadratic penalty (smooth, differentiable)
    lat_penalty = (lat_low_violation ** 2 + lat_high_violation ** 2).mean()
    lng_penalty = (lng_low_violation ** 2 + lng_high_violation ** 2).mean()

    return penalty_weight * (lat_penalty + lng_penalty)


def constraint_weighted_fusion(
    moe_fused_lat: float,
    moe_fused_lng: float,
    constraint: ConstraintBox,
    blend_strength: float = 0.15,
    margin_km: float = 100.0,
    router_confidence: float = 1.0,
    has_named_location: bool = False,
) -> tuple[float, float]:
    """Safety-net correction: only blend MoE toward constraint when far outside
    AND GeoCoT identified a specific named location.

    Gating conditions (ALL must be met to apply correction):
      1. GeoCoT named a specific location (not just generic "China")
      2. Constraint has high confidence (>0.6)
      3. MoE is far outside constraint box (>margin_km)
      4. Router is NOT highly confident (<0.8)

    Without named locations, GeoCoT spatial constraints are unreliable
    (KB keyword matching produces random matches for generic descriptions).

    Args:
        moe_fused_lat/lng: MoE fused prediction in degrees
        constraint: ConstraintBox defining the allowed region
        blend_strength: how much to pull toward constraint (0-1), default weak
        margin_km: only apply correction if MoE is this far outside the constraint
        router_confidence: MoE router max weight (0-1), only correct when low
        has_named_location: True if GeoCoT identified a specific place name

    Returns:
        (adjusted_lat, adjusted_lng)
    """
    # CRITICAL: don't apply constraint unless GeoCoT actually named a place
    if not has_named_location:
        return moe_fused_lat, moe_fused_lng

    # Don't override a confident MoE router
    if router_confidence > 0.8:
        return moe_fused_lat, moe_fused_lng

    # Don't apply if constraint itself is weak
    if constraint.confidence < 0.6:
        return moe_fused_lat, moe_fused_lng

    # Convert margin to degrees
    margin_lat = margin_km / 111.0
    avg_lat = (constraint.lat_min + constraint.lat_max) / 2
    margin_lng = margin_km / (111.0 * math.cos(math.radians(avg_lat)))

    # Check how far MoE is from constraint (0 if inside)
    lat_below = max(0, constraint.lat_min - moe_fused_lat)
    lat_above = max(0, moe_fused_lat - constraint.lat_max)
    lng_below = max(0, constraint.lng_min - moe_fused_lng)
    lng_above = max(0, moe_fused_lng - constraint.lng_max)

    # If inside constraint, return unchanged
    if lat_below == 0 and lat_above == 0 and lng_below == 0 and lng_above == 0:
        return moe_fused_lat, moe_fused_lng

    # Only correct if significantly outside (>margin_km)
    if lat_below < margin_lat and lat_above < margin_lat and lng_below < margin_lng and lng_above < margin_lng:
        return moe_fused_lat, moe_fused_lng

    # Safety net: gently pull toward nearest edge of constraint box
    adjusted_lat = moe_fused_lat
    adjusted_lng = moe_fused_lng

    if lat_below >= margin_lat:
        adjusted_lat = moe_fused_lat + blend_strength * (constraint.lat_min - moe_fused_lat)
    elif lat_above >= margin_lat:
        adjusted_lat = moe_fused_lat + blend_strength * (constraint.lat_max - moe_fused_lat)

    if lng_below >= margin_lng:
        adjusted_lng = moe_fused_lng + blend_strength * (constraint.lng_min - moe_fused_lng)
    elif lng_above >= margin_lng:
        adjusted_lng = moe_fused_lng + blend_strength * (constraint.lng_max - moe_fused_lng)

    return adjusted_lat, adjusted_lng
