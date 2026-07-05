"""Sensor-first constraint satisfaction engine.

ConstraintSolver maintains an ordered list of spatial constraints and
applies them via intersection. When constraints conflict (intersection empty),
soft constraints are relaxed/dropped in LIFO order before touching hard ones.

Core principle: sensors + geographic elements = spatial constraints.
Intersection of all constraints = answer. No coordinate regression needed.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .geo_kb import BBox
from .constraint_utils import (
    intersect_bbox_lists,
    bbox_total_area_km2,
    compute_constraint_power,
    compute_centroid_and_radius,
    ConstraintStep,
    explain_chain,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConstraintSource:
    """A source of spatial constraints."""
    name: str                          # "elevation_4500m"
    source_type: str                   # "sensor_elevation"|"sensor_climate"|"element"|"compound"
    bboxes: list[BBox]                 # spatial constraint regions
    confidence: float = 0.5            # 0.0-1.0
    is_hard: bool = True               # False = can be relaxed/dropped on conflict
    metadata: dict = field(default_factory=dict)


@dataclass
class ConstraintResult:
    """Output of constraint solving."""
    center_lat: float
    center_lng: float
    uncertainty_km: float
    confidence: float
    remaining_bboxes: list[BBox]
    steps: list[ConstraintStep]        # each constraint application recorded
    dropped: list[str]                 # constraints that were relaxed/dropped
    initial_area_km2: float = 0.0

    @property
    def explanation(self) -> str:
        return explain_chain(
            self.steps, self.center_lat, self.center_lng,
            self.uncertainty_km, self.dropped,
        )

    def to_dict(self) -> dict:
        return {
            "center_lat": self.center_lat,
            "center_lng": self.center_lng,
            "uncertainty_km": self.uncertainty_km,
            "confidence": self.confidence,
            "n_remaining_bboxes": len(self.remaining_bboxes),
            "n_steps": len(self.steps),
            "dropped": self.dropped,
            "explanation": self.explanation,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Constraint satisfaction engine
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintSolver:
    """Sensor-first constraint satisfaction engine.

    Constraints are applied in the order they are added (caller controls
    priority). Hard constraints (is_hard=True) cannot be dropped; soft
    constraints can be relaxed or dropped when they cause empty intersection.

    Usage:
        solver = ConstraintSolver()
        solver.add_constraint(ConstraintSource("elev", "sensor_elevation",
                                elev_bboxes, confidence=0.9, is_hard=True))
        solver.add_constraint(ConstraintSource("karst", "element",
                                karst_bboxes, confidence=0.7, is_hard=False))
        result = solver.solve()
        print(result.explanation)
    """

    # Full China bounding box
    _CHINA_LAT_MIN = 18.0
    _CHINA_LAT_MAX = 54.0
    _CHINA_LNG_MIN = 73.0
    _CHINA_LNG_MAX = 135.5

    def __init__(self):
        self._constraints: list[ConstraintSource] = []
        self._china_bbox = BBox(
            self._CHINA_LAT_MIN, self._CHINA_LAT_MAX,
            self._CHINA_LNG_MIN, self._CHINA_LNG_MAX,
            "China",
        )

    def add_constraint(self, source: ConstraintSource):
        """Add a constraint. Order matters — first added = highest priority."""
        self._constraints.append(source)

    def add_constraints(self, sources: list[ConstraintSource]):
        """Add multiple constraints at once."""
        for s in sources:
            self._constraints.append(s)

    @property
    def n_constraints(self) -> int:
        return len(self._constraints)

    def solve(self) -> ConstraintResult:
        """Execute the constraint satisfaction pipeline.

        1. Start with full China bbox
        2. Apply hard constraints first (in order)
        3. Apply soft constraints next
        4. On empty intersection: relax/drop soft constraints LIFO
        5. If still empty after all soft dropped: keep hard-only result
        6. Record every step for explainability
        """
        remaining = [self._china_bbox]
        initial_area = bbox_total_area_km2(remaining)
        steps: list[ConstraintStep] = []
        dropped: list[str] = []

        hard = [c for c in self._constraints if c.is_hard]
        soft = [c for c in self._constraints if not c.is_hard]

        # ── Phase 1: Apply hard constraints ────────────────────────────────
        for c in hard:
            if not remaining:
                break
            before_area = bbox_total_area_km2(remaining)
            before_count = len(remaining)

            after = intersect_bbox_lists([remaining, c.bboxes])

            if not after:
                # Hard constraint conflict — relax by expanding constraint bboxes
                expanded = [_expand_bbox(b, 1.0) for b in c.bboxes]
                after = intersect_bbox_lists([remaining, expanded])
                if after:
                    steps.append(ConstraintStep(
                        name=c.name, source_type=c.source_type,
                        area_before_km2=before_area,
                        area_after_km2=bbox_total_area_km2(after),
                        prune_ratio=compute_constraint_power(remaining, after),
                        bbox_count_before=before_count,
                        bbox_count_after=len(after),
                        was_relaxed=True,
                    ))
                    remaining = after
                else:
                    # Even relaxed doesn't help — mark conflict but keep going
                    steps.append(ConstraintStep(
                        name=c.name, source_type=c.source_type,
                        area_before_km2=before_area,
                        area_after_km2=before_area,
                        prune_ratio=0.0,
                        bbox_count_before=before_count,
                        bbox_count_after=before_count,
                        was_dropped=True,
                    ))
                    dropped.append(c.name)
            else:
                after_area = bbox_total_area_km2(after)
                steps.append(ConstraintStep(
                    name=c.name, source_type=c.source_type,
                    area_before_km2=before_area,
                    area_after_km2=after_area,
                    prune_ratio=compute_constraint_power(remaining, after),
                    bbox_count_before=before_count,
                    bbox_count_after=len(after),
                ))
                remaining = after

        # ── Phase 2: Apply soft constraints ─────────────────────────────────
        for c in soft:
            if not remaining:
                break
            before_area = bbox_total_area_km2(remaining)
            before_count = len(remaining)

            after = intersect_bbox_lists([remaining, c.bboxes])

            if not after:
                # Soft constraint conflict — try relaxed, then drop if still empty
                expanded = [_expand_bbox(b, 2.0) for b in c.bboxes]
                after = intersect_bbox_lists([remaining, expanded])
                if after:
                    steps.append(ConstraintStep(
                        name=c.name, source_type=c.source_type,
                        area_before_km2=before_area,
                        area_after_km2=bbox_total_area_km2(after),
                        prune_ratio=compute_constraint_power(remaining, after),
                        bbox_count_before=before_count,
                        bbox_count_after=len(after),
                        was_relaxed=True,
                    ))
                    remaining = after
                else:
                    # Drop this constraint entirely
                    steps.append(ConstraintStep(
                        name=c.name, source_type=c.source_type,
                        area_before_km2=before_area,
                        area_after_km2=before_area,
                        prune_ratio=compute_constraint_power(remaining, remaining),
                        bbox_count_before=before_count,
                        bbox_count_after=before_count,
                        was_dropped=True,
                    ))
                    dropped.append(c.name)
            else:
                after_area = bbox_total_area_km2(after)
                steps.append(ConstraintStep(
                    name=c.name, source_type=c.source_type,
                    area_before_km2=before_area,
                    area_after_km2=after_area,
                    prune_ratio=compute_constraint_power(remaining, after),
                    bbox_count_before=before_count,
                    bbox_count_after=len(after),
                ))
                remaining = after

        # ── Phase 3: Compute result ─────────────────────────────────────────
        if remaining:
            # Pick the LARGEST bbox for centroid when multiple disconnected regions
            if len(remaining) > 1:
                remaining.sort(key=lambda b: b.area_km2, reverse=True)
                largest = remaining[0]
                # Only use largest if it's significantly bigger (>2x) than the runner-up
                if len(remaining) == 1 or largest.area_km2 > remaining[1].area_km2 * 2.0:
                    center_lat, center_lng = largest.center_lat, largest.center_lng
                    dlat = (largest.lat_max - largest.lat_min) * 111.0 / 2
                    dlng = (largest.lng_max - largest.lng_min) * math.cos(math.radians(largest.center_lat)) * 111.0 / 2
                    radius = max(math.sqrt(dlat**2 + dlng**2), 10.0)
                else:
                    center_lat, center_lng, radius = compute_centroid_and_radius(remaining)
            else:
                center_lat, center_lng, radius = compute_centroid_and_radius(remaining)
            radius = min(radius, 2500.0)
            remaining_area = bbox_total_area_km2(remaining)
            area_based_radius = math.sqrt(remaining_area / math.pi) * 0.3
            radius = max(radius, area_based_radius, 10.0)
        else:
            center_lat, center_lng, radius = 35.0, 105.0, 2500.0

        # Confidence: proportional to constraint effectiveness
        n_effective = sum(1 for s in steps if not s.was_dropped)
        total_prune = 1.0 - (bbox_total_area_km2(remaining) / initial_area) if initial_area > 0 else 0.0
        # Cap at 0.75 — sensors+GeoKB alone can't give >75% without cross-validation
        confidence = min(0.20 + 0.05 * n_effective + 0.25 * total_prune, 0.75)

        return ConstraintResult(
            center_lat=center_lat,
            center_lng=center_lng,
            uncertainty_km=radius,
            confidence=confidence,
            remaining_bboxes=remaining,
            steps=steps,
            dropped=dropped,
            initial_area_km2=initial_area,
        )


def _expand_bbox(bbox: BBox, deg: float) -> BBox:
    """Expand a bbox by deg degrees in all directions, clamped to China bounds."""
    return BBox(
        lat_min=max(18.0, bbox.lat_min - deg),
        lat_max=min(54.0, bbox.lat_max + deg),
        lng_min=max(73.0, bbox.lng_min - deg),
        lng_max=min(135.5, bbox.lng_max + deg),
        label=f"{bbox.label}+{deg}deg",
    )
