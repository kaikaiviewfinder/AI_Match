"""Grid-scoring engine for sensor-first geolocation.

Applies weighted log-likelihood scoring over constraint area cells,
selecting the MAP (maximum a posteriori) location.

Design:
  - Continuous sensors (elevation, temperature, humidity): Gaussian kernel
  - Discrete elements (VLM): self-information weighted compatibility
  - Compound scenes: confidence-weighted bonus
  - Output: argmax cell + uncertainty from top-K spread
"""

import math
from typing import Optional

import numpy as np

from .geo_kb import BBox, get_bboxes_for_element
from .constraint_utils import bbox_total_area_km2, haversine_km


# ═══════════════════════════════════════════════════════════════════════════════
# Per-element sensor-consistency knowledge base (Step 1 of fusion strategy)
# ═══════════════════════════════════════════════════════════════════════════════

# Climate zones → typical (temp_lo, temp_hi, humid_lo, humid_hi) in July
_CLIMATE_SENSOR_RANGES = {
    "tropical":       (25, 35, 70, 90),
    "subtropical":    (25, 32, 65, 85),
    "temperate":      (15, 28, 50, 75),
    "arid":           (20, 38, 15, 50),
    "alpine":         (5,  18, 35, 65),
    "boreal":         (10, 22, 45, 70),
}

# Terrain types → typical (elev_lo, elev_hi) in metres
_TERRAIN_ELEV_RANGES = {
    "urban_flat":       (0,    1500),
    "farmland_plain":   (0,    1200),
    "rolling_hills":    (50,   2500),
    "sharp_mountains":  (800,  5500),
    "karst_peaks":      (0,    2500),
    "sandstone_pillars":(200,  2500),
    "desert_dunes":     (-100, 2000),
    "grassland_steppe": (0,    3500),
    "plateau":          (2000, 5000),
}

# Vegetation zones → typical (elev_lo, elev_hi, temp_lo, temp_hi)
_VEGETATION_SENSOR_RANGES = {
    "tropical_rainforest":  (0,   1500, 22, 35),
    "broadleaf_evergreen":  (0,   2500, 15, 32),
    "broadleaf_deciduous":  (0,   3000,  5, 28),
    "conifer_forest":       (500, 4000,  0, 22),
    "mixed_forest":         (0,   3000,  5, 28),
    "alpine_meadow":        (2500,5500, -5, 18),
    "desert_scrub":         (-100,3500,  5, 38),
    "grassland":            (0,   4000,  0, 28),
    "bamboo_forest":        (0,   2000, 12, 32),
    "cropland":             (0,   2500,  5, 32),
    "sparse":               (0,   6000,-10, 35),
}


def _gaussian_consistency(value: float, lo: float, hi: float,
                           sigma_fraction: float = 0.25) -> float:
    """How well *value* fits inside [lo, hi].

    Returns 1.0 if value is inside the band; decays as a Gaussian outside.
    sigma = sigma_fraction * (hi - lo), clamped to ≥ 2.0 to avoid spike.
    """
    sigma = max(sigma_fraction * (hi - lo), 2.0)
    if lo <= value <= hi:
        return 1.0
    dist = min(abs(value - lo), abs(value - hi))
    return math.exp(-0.5 * (dist / sigma) ** 2)


def compute_element_weights(
    elements: dict,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
) -> dict:
    """Compute per-element sensor-consistency weights ∈ [0.05, 1.0].

    Returns dict keyed by (category, value) — same key format as
    ``precompute_element_infos()`` so the two dicts can be aligned
    inside ``score_location()``.

    Principles
    ----------
    - climate  → cross-check with temperature + humidity
    - terrain  → cross-check with elevation
    - vegetation → cross-check with elevation + temperature
    - elevation_estimate_m → compare VLM range against sensor elevation
    - urbanization, language, architecture (etc.) → default 0.70 (no
      direct sensor constraint, but VLM is generally reliable here)
    """
    weights = {}

    for cat, val in elements.items():
        if val is None or val == "":
            continue

        # Handle elevation_estimate_m before the scalar filter (it's a list)
        if cat in ("elevation_estimate_m",):
            # Use tuple version as dict key (list is unhashable)
            key = (cat, tuple(val) if isinstance(val, list) else val)
            if sensor_elevation_m is not None:
                if isinstance(val, list) and len(val) == 2:
                    v_lo, v_hi = float(val[0]), float(val[1])
                    w = _gaussian_consistency(sensor_elevation_m, v_lo, v_hi,
                                               sigma_fraction=0.35)
                    weights[key] = round(w, 2)
                else:
                    weights[key] = 0.50
            else:
                weights[key] = 0.70
            continue

        if isinstance(val, (dict, list)):
            continue

        key = (cat, val)

        val_lower = str(val).lower().replace(" ", "_")

        # ── climate ──────────────────────────────────────────────────────
        if cat in ("climate_zone",):
            ranges = _CLIMATE_SENSOR_RANGES.get(val_lower)
            if ranges and sensor_temperature_c is not None and sensor_humidity_pct is not None:
                t_lo, t_hi, h_lo, h_hi = ranges
                w_t = _gaussian_consistency(sensor_temperature_c, t_lo, t_hi)
                w_h = _gaussian_consistency(sensor_humidity_pct, h_lo, h_hi)
                weights[key] = round(w_t * w_h, 2)
            elif ranges:
                weights[key] = 0.80
            else:
                weights[key] = 0.70

        # ── terrain ──────────────────────────────────────────────────────
        elif cat in ("terrain_type",):
            ranges = _TERRAIN_ELEV_RANGES.get(val_lower)
            if ranges and sensor_elevation_m is not None:
                e_lo, e_hi = ranges
                weights[key] = round(_gaussian_consistency(sensor_elevation_m, e_lo, e_hi), 2)
            elif ranges:
                weights[key] = 0.80
            else:
                weights[key] = 0.70

        # ── vegetation ───────────────────────────────────────────────────
        elif cat in ("vegetation_zone",):
            ranges = _VEGETATION_SENSOR_RANGES.get(val_lower)
            if ranges:
                e_lo, e_hi, t_lo, t_hi = ranges
                w_e = (1.0 if sensor_elevation_m is None
                       else _gaussian_consistency(sensor_elevation_m, e_lo, e_hi))
                w_t = (1.0 if sensor_temperature_c is None
                       else _gaussian_consistency(sensor_temperature_c, t_lo, t_hi))
                weights[key] = round(w_e * w_t, 2)
            else:
                weights[key] = 0.70

        # ── everything else ──────────────────────────────────────────────
        else:
            weights[key] = 0.70

    return weights


def score_location(
    lat: float,
    lng: float,
    elements: dict,
    element_infos: dict,
    compound_matches: list,
    sensor_elevation_m: Optional[float],
    sensor_temperature_c: Optional[float],
    sensor_humidity_pct: Optional[float],
    dem,
    climate,
    geocot_prediction: Optional[tuple[float, float]] = None,
    geo_weight: float = 0.0,
    element_weights: Optional[dict] = None,
    province_bboxes: Optional[list[BBox]] = None,
    province_bonus_weight: float = 0.0,
    cluster_bonus_zones: Optional[list[tuple[float, "BBox"]]] = None,
    compound_dispersed: bool = False,
) -> float:
    """Weighted log-likelihood score for a single (lat, lng) cell.

    Higher is better. Score = 0 is the neutral baseline.

    Components (default weights → dynamic if compound anchors):
      elevation:  Gaussian kernel on DEM match          (0.50 → 0.30)
      temperature: Gaussian kernel on climate match     (0.15)
      humidity:    Gaussian kernel on humidity match    (0.05)
      elements:    self-information bonus per match     (0.25)
      compound:    confidence bonus if inside scene     (0.15 → 0.30)
      geocot:      Gaussian proximity to GeoCoT pred    (geo_weight)

    When a high-confidence compound scene (ratio ≥ 0.75) covers the cell,
    elevation is downweighted and compound is boosted: the compound scene
    represents a tighter spatial constraint than a single DEM grid cell.
    """
    # ── Check for high-confidence compound-scene anchor ──────────────────
    compound_anchor_ratio = 0.0
    if compound_matches:
        for scene_name, scene_bboxes, ratio in compound_matches:
            if ratio >= 0.75 and any(
                b.lat_min <= lat <= b.lat_max and b.lng_min <= lng <= b.lng_max
                for b in scene_bboxes
            ):
                compound_anchor_ratio = max(compound_anchor_ratio, ratio)
                break  # first high-conf match is enough

    # Dynamic weights: compound anchors override noisy sensor elevation
    if compound_anchor_ratio > 0:
        w_elev = 0.30
        w_compound = 0.30
    else:
        w_elev = 0.50
        w_compound = 0.15

    # Elevation-adaptive gating (V5): low elevation can't prune China.
    # At 63m, ±35% covers 9.6M km² (102% of China) — DEM matching is pure
    # noise. At 3500m, the same tolerance covers a thin altitude band that
    # strongly constrains location.
    # EXCEPTION: in DISPERSE mode VLM spatial signals are already scattered
    # and conflicting — elevation sensor is one of the few reliable anchors,
    # so we keep full weight regardless of altitude.
    if sensor_elevation_m is not None and not compound_dispersed:
        if sensor_elevation_m < 200:
            w_elev *= 0.3   # true lowland: DEM noise (~102% of China at 60m)
        elif sensor_elevation_m < 500:
            w_elev *= 0.6   # mid-elevation: partial signal (excludes coasts & plateaus)
        elif sensor_elevation_m < 2000:
            w_elev *= 0.6
        # >2000m: keep full weight (strong constraint)

    score = 0.0

    # ── Elevation (Gaussian kernel) ──────────────────────────────────────
    if sensor_elevation_m is not None and dem is not None:
        dem_elev = dem.query(lat, lng)
        if dem_elev is not None:
            sigma = max(abs(sensor_elevation_m) * 0.35, 200.0)
            diff = dem_elev - sensor_elevation_m
            score -= w_elev * (diff / sigma) ** 2
        else:
            score -= w_elev * 4.0  # no DEM data penalty (~2-sigma)

    # ── Temperature (Gaussian kernel) ────────────────────────────────────
    if sensor_temperature_c is not None and climate is not None:
        dem_elev = dem.query(lat, lng) if dem else 500
        elev_for_temp = dem_elev if dem_elev is not None else 500
        t_lo, t_hi = climate.estimate_temperature_range(lat, lng, elev_for_temp, month=7)
        t_mid = (t_lo + t_hi) / 2
        sigma = max((t_hi - t_lo) * 0.5, 5.0)
        diff = sensor_temperature_c - t_mid
        score -= 0.15 * (diff / sigma) ** 2

    # ── Humidity (Gaussian kernel) ───────────────────────────────────────
    if sensor_humidity_pct is not None and climate is not None:
        h_lo, h_hi = climate.estimate_humidity_range(lat, lng, month=7)
        h_mid = (h_lo + h_hi) / 2
        sigma = max((h_hi - h_lo) * 0.5, 10.0)
        diff = sensor_humidity_pct - h_mid
        score -= 0.05 * (diff / sigma) ** 2

    # ── Geographic elements (self-information weighted × sensor consistency) ──
    n_elem = len(element_infos)
    if n_elem > 0:
        elem_score = 0.0
        for (cat, val), info_weight in element_infos.items():
            bboxes = get_bboxes_for_element(cat, val)
            if not bboxes:
                continue
            # Sensor-consistency weight: how well this element agrees with
            # physical sensor readings (0.05 = strong conflict, 1.0 = perfect match)
            sw = (element_weights or {}).get((cat, val), 0.70)
            if any(b.lat_min <= lat <= b.lat_max and b.lng_min <= lng <= b.lng_max
                   for b in bboxes):
                elem_score += info_weight * sw
            else:
                elem_score -= 0.5  # small penalty for mismatch
        score += 0.25 * (elem_score / n_elem)

    # ── Compound scenes (confidence-weighted bonus, dynamic weight) ──────
    if compound_matches:
        for scene_name, scene_bboxes, ratio in compound_matches:
            if any(b.lat_min <= lat <= b.lat_max and b.lng_min <= lng <= b.lng_max
                   for b in scene_bboxes):
                score += w_compound * ratio
                break

    # ── GeoCoT spatial prior (Gaussian proximity bonus) ──────────────────
    if geocot_prediction is not None and geo_weight > 0.001:
        glat, glng = geocot_prediction
        dist = haversine_km(lat, lng, glat, glng)
        sigma = 300.0
        score += geo_weight * math.exp(-0.5 * (dist / sigma) ** 2)

    # ── Province-level prior (soft bonus, only when compound is ambiguous) ─
    if province_bboxes and province_bonus_weight > 0.001:
        for b in province_bboxes:
            if b.lat_min <= lat <= b.lat_max and b.lng_min <= lng <= b.lng_max:
                # Gaussian proximity from bbox center, sigma = half-width
                dx = (lng - b.center_lng) * 111.0
                dy = (lat - b.center_lat) * 111.0
                dist = math.sqrt(dx * dx + dy * dy)
                half_w = max((b.lat_max - b.lat_min) * 111.0 / 2, 100.0)
                proximity = math.exp(-0.5 * (dist / half_w) ** 2)
                score += province_bonus_weight * proximity
                break

    # ── Cluster bonus (E+B hybrid: DBSCAN cluster soft scoring) ──────────
    # Only active in DISPERSE mode. Dense clusters (many bboxes) give strong
    # bonus; isolated bboxes give weak bonus. No area is hard-excluded.
    if cluster_bonus_zones:
        for cw, zone in cluster_bonus_zones:
            if zone.lat_min <= lat <= zone.lat_max and zone.lng_min <= lng <= zone.lng_max:
                # Bonus ∝ cluster_weight × compound_anchor_ratio
                # Main clusters: cw ≥ 0.5 → bonus up to 0.15
                # Minor clusters: cw ≤ 0.3 → bonus ~0.03-0.06
                score += w_compound * cw
                break

    return score


def precompute_element_infos(
    elements: dict,
    constraint_area_km2: float,
    max_info: float = 5.0,
) -> dict:
    """Pre-compute self-information weight for each element.

    info = -log(bbox_area / constraint_area), capped at max_info.
    Small, specific elements get high weight; broad elements get low weight.

    Returns dict mapping (category, value) -> info_weight.
    """
    infos = {}
    for cat, val in elements.items():
        if val is None or val == "" or val == [] or isinstance(val, (dict, list)):
            continue
        bboxes = get_bboxes_for_element(cat, val)
        if not bboxes:
            continue
        # Use smallest bbox area (most specific region)
        min_area = min(b.area_km2 for b in bboxes)
        if min_area <= 0 or constraint_area_km2 <= 0:
            infos[(cat, val)] = 1.0
            continue
        p = min_area / constraint_area_km2
        # When element bbox >= constraint area, P >= 1, info <= 0.
        # These elements have zero discriminative power — skip them.
        if p >= 1.0:
            continue
        info = -math.log(max(p, 1e-8))
        # Skip near-zero info elements (< 0.01 nats ≈ no discriminative power)
        if info < 0.01:
            continue
        infos[(cat, val)] = min(info, max_info)
    return infos


def grid_search(
    constraint_bboxes: list[BBox],
    elements: dict,
    element_infos: dict,
    compound_matches: list,
    sensor_elevation_m: Optional[float],
    sensor_temperature_c: Optional[float],
    sensor_humidity_pct: Optional[float],
    dem,
    climate,
    target_points: int = 4000,
    local_refine_top_k: int = 5,
    geocot_prediction: Optional[tuple[float, float]] = None,
    geo_weight: float = 0.0,
    element_weights: Optional[dict] = None,
    province_bboxes: Optional[list[BBox]] = None,
    province_bonus_weight: float = 0.0,
    cluster_bonus_zones: Optional[list[tuple[float, "BBox"]]] = None,
    compound_dispersed: bool = False,
) -> tuple[float, float, float, list[tuple[float, float, float]]]:
    """Grid-search within constraint area for best-scoring location.

    1. Adaptive step size based on constraint area (~target_points total)
    2. Score every cell with score_location()
    3. Local refinement around top-K cells at 0.005 deg resolution
    4. Return (best_lat, best_lng, uncertainty_km, all_points)

    Uncertainty = standard deviation of top-10% scoring points from best.
    """
    if not constraint_bboxes:
        return 35.0, 105.0, 2500.0, []

    total_area = bbox_total_area_km2(constraint_bboxes)

    # Adaptive step: aim for ~target_points across total area
    # Area per cell ~ (step * 111)^2 km^2
    # n_points = total_area / cell_area = total_area / (step * 111)^2
    # step = sqrt(total_area / target_points) / 111
    step_deg = math.sqrt(total_area / target_points) / 111.0
    step_deg = max(0.008, min(step_deg, 0.10))  # clamp [0.008, 0.10] deg

    # ── Coarse grid search ───────────────────────────────────────────────
    all_points = []
    for b in constraint_bboxes:
        lat = b.lat_min
        while lat <= b.lat_max:
            lng = b.lng_min
            while lng <= b.lng_max:
                s = score_location(
                    lat, lng, elements, element_infos, compound_matches,
                    sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                    dem, climate, geocot_prediction, geo_weight,
                    element_weights=element_weights,
                    province_bboxes=province_bboxes,
                    province_bonus_weight=province_bonus_weight,
                    cluster_bonus_zones=cluster_bonus_zones,
                    compound_dispersed=compound_dispersed,
                )
                all_points.append((lat, lng, s))
                lng += step_deg
            lat += step_deg

    if not all_points:
        # Fallback: centroid
        from .constraint_utils import compute_centroid_and_radius
        clat, clng, rad = compute_centroid_and_radius(constraint_bboxes)
        return clat, clng, rad, []

    # ── Local refinement around top-K ────────────────────────────────────
    all_points.sort(key=lambda p: p[2], reverse=True)
    top_k = all_points[:local_refine_top_k]

    fine_step = 0.005
    refined = []
    for lat0, lng0, _ in top_k:
        for dlat in [-0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015]:
            for dlng in [-0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015]:
                rlat = lat0 + dlat
                rlng = lng0 + dlng
                s = score_location(
                    rlat, rlng, elements, element_infos, compound_matches,
                    sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                    dem, climate, geocot_prediction, geo_weight,
                    element_weights=element_weights,
                    province_bboxes=province_bboxes,
                    province_bonus_weight=province_bonus_weight,
                    cluster_bonus_zones=cluster_bonus_zones,
                    compound_dispersed=compound_dispersed,
                )
                refined.append((rlat, rlng, s))

    if refined:
        all_points.extend(refined)
        all_points.sort(key=lambda p: p[2], reverse=True)

    best = all_points[0]
    best_lat, best_lng, best_score = best

    # ── Uncertainty from top-10% spread ──────────────────────────────────
    top_10pct = [p for p in all_points if p[2] >= best_score - max(abs(best_score) * 0.1, 0.5)]
    if len(top_10pct) < 5:
        top_10pct = all_points[:max(10, len(all_points) // 10)]

    dists = [haversine_km(best_lat, best_lng, p[0], p[1]) for p in top_10pct]
    uncertainty = float(np.mean(dists) + np.std(dists)) if dists else 50.0
    uncertainty = max(min(uncertainty, 2000.0), 5.0)

    return best_lat, best_lng, uncertainty, all_points
