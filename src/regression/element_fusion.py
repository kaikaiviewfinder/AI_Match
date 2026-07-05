"""Multi-element geographic constraint fusion engine.

Tiered constraint algorithm:
  1. STRONG categories (terrain, vegetation, rock, soil, climate, sky):
     → hard bbox intersection → candidate region
  2. BROAD categories (language, urbanization, architecture):
     → confidence-only, don't participate in hard filtering
  3. Elevation sensor: hard grid filter on candidate bboxes
  4. If intersection empty → relax by removing broadest element
  5. Output: centroid + uncertainty radius + confidence
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geo_kb import (
    BBox, get_bboxes_for_element,
    match_compound_scenes, match_compound_scenes_soft, COMPOUND_SCENES,
)
from .dem_lookup import get_dem
from .climate_lookup import get_climate, filter_by_climate
from .constraint_solver import ConstraintSolver, ConstraintSource, ConstraintResult
from .constraint_utils import bbox_total_area_km2, intersect_bbox_lists, \
    cluster_compound_bboxes, cluster_zones_to_bonus_zones, subtract_bboxes
from .scoring import (
    score_location, precompute_element_infos, grid_search,
    compute_element_weights,
)

# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateRegion:
    center_lat: float
    center_lng: float
    radius_km: float
    confidence: float
    bboxes: list = field(default_factory=list)
    active_elements: list = field(default_factory=list)
    dropped_elements: list = field(default_factory=list)
    explanation: str = ""


@dataclass
class FusionResult:
    latitude: float
    longitude: float
    uncertainty_km: float
    confidence: float
    candidate_region: Optional[CandidateRegion] = None
    explanation_parts: list = field(default_factory=list)

    def explanation_text(self) -> str:
        return "\n".join(self.explanation_parts)


# ═══════════════════════════════════════════════════════════════════════════
# Category classification
# ═══════════════════════════════════════════════════════════════════════════

# These categories have WEAK spatial constraint — they cover most of China.
# We use them for confidence only, never for hard filtering.
BROAD_CATEGORIES = {
    "language_script",       # "simplified_chinese" = all of China
    "architecture_style",    # "modern_glass" = nationwide
    "urbanization",          # "metropolis"/"medium_city" = too broad (13-24 regions)
    "sky_quality",           # "clear_blue"/"grey_hazy" = half of China
}

# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def bbox_area(bbox: BBox) -> float:
    return bbox.area_km2


def intersect_bboxes(a_bboxes: list[BBox], b_bboxes: list[BBox]) -> list[BBox]:
    """Pairwise intersection of two bbox lists."""
    result = []
    for a in a_bboxes:
        for b in b_bboxes:
            lat_min = max(a.lat_min, b.lat_min)
            lat_max = min(a.lat_max, b.lat_max)
            lng_min = max(a.lng_min, b.lng_min)
            lng_max = min(a.lng_max, b.lng_max)
            if lat_min < lat_max and lng_min < lng_max:
                result.append(BBox(
                    lat_min=lat_min, lat_max=lat_max,
                    lng_min=lng_min, lng_max=lng_max,
                    label=f"{a.label} ∩ {b.label}",
                ))
    return result


def compute_bbox_center_radius(bboxes: list[BBox]) -> tuple[float, float, float]:
    """Centroid and bounding radius of bboxes. Returns (lat, lng, radius_km)."""
    if not bboxes:
        return 35.0, 105.0, 2500.0

    if len(bboxes) == 1:
        b = bboxes[0]
        dlat = (b.lat_max - b.lat_min) * 111.0 / 2
        dlng = (b.lng_max - b.lng_min) * math.cos(math.radians(b.center_lat)) * 111.0 / 2
        return b.center_lat, b.center_lng, max(math.sqrt(dlat**2 + dlng**2), 10.0)

    # Multiple bboxes: area-weighted centroid
    total_lat = sum(b.center_lat * bbox_area(b) for b in bboxes)
    total_lng = sum(b.center_lng * bbox_area(b) for b in bboxes)
    total_area = sum(bbox_area(b) for b in bboxes)
    center_lat = total_lat / total_area if total_area > 0 else 35.0
    center_lng = total_lng / total_area if total_area > 0 else 105.0

    max_dist = 0
    for b in bboxes:
        for lat, lng in [(b.lat_min, b.lng_min), (b.lat_min, b.lng_max),
                          (b.lat_max, b.lng_min), (b.lat_max, b.lng_max)]:
            d = haversine_km(center_lat, center_lng, lat, lng)
            max_dist = max(max_dist, d)
    return center_lat, center_lng, max(max_dist, 50.0)


# ═══════════════════════════════════════════════════════════════════════════
# Elevation heuristics
# ═══════════════════════════════════════════════════════════════════════════

def _is_high_elevation_region(lat: float, lng: float) -> bool:
    """True if this lat/lng is in a generally high-elevation region (>2000m avg)."""
    if 26.5 <= lat <= 36.5 and 78.0 <= lng <= 99.5:   # Tibet Plateau
        return True
    if 31.0 <= lat <= 39.5 and 89.0 <= lng <= 103.5:  # Qinghai
        return True
    if 26.0 <= lat <= 32.0 and 97.0 <= lng <= 104.0:  # W Sichuan / NW Yunnan
        return True
    if 41.0 <= lat <= 45.0 and 80.0 <= lng <= 96.5:   # Tianshan
        return True
    return False


def _is_low_elevation_region(lat: float, lng: float) -> bool:
    """True if generally low elevation (<500m avg)."""
    if 30.0 <= lat <= 41.0 and 113.5 <= lng <= 122.5:  # N China Plain
        return True
    if 41.0 <= lat <= 48.5 and 121.5 <= lng <= 135.5:  # NE China
        return True
    if 29.0 <= lat <= 33.0 and 116.0 <= lng <= 122.0:  # Yangtze Delta
        return True
    if 22.0 <= lat <= 30.5 and 110.0 <= lng <= 122.5:  # SE China
        return True
    if 21.0 <= lat <= 24.0 and 109.0 <= lng <= 117.5:  # Pearl River Delta
        return True
    if 29.0 <= lat <= 32.5 and 103.0 <= lng <= 108.0:  # Sichuan Basin
        return True
    return False


def filter_by_elevation(bboxes: list[BBox], elevation_m: float,
                         strict: bool = True) -> list[BBox]:
    """Remove bboxes incompatible with the given elevation.

    Primary: DEM grid lookup with ±35% tolerance (strict) / ±50% (relaxed).
    Fallback: heuristic region checks (if DEM grid unavailable).

    Two-pass strategy:
      1. Strict pass: DEM-based filtering with tight tolerance
      2. Relaxed pass (if strict yields nothing): wider tolerance

    Returns empty list only if NO bbox is plausible at any tolerance.
    Caller should fall back to unfiltered bboxes on empty return.
    """
    if elevation_m is None:
        return bboxes

    dem = get_dem()

    if dem is not None:
        # ── DEM-based filtering ──────────────────────────────────────────
        def _dem_passes(b: BBox, tolerance: float) -> bool:
            return dem.bbox_matches_elevation(
                b.lat_min, b.lat_max, b.lng_min, b.lng_max,
                elevation_m, tolerance_pct=tolerance,
            )

        # Strict pass: ±35% tolerance
        filtered = [b for b in bboxes if _dem_passes(b, 0.35)]
        if filtered:
            return filtered

        # Relaxed pass: ±50% tolerance
        relaxed = [b for b in bboxes if _dem_passes(b, 0.50)]
        return relaxed  # may be empty — caller should handle

    else:
        # ── Heuristic fallback (no DEM grid available) ───────────────────
        def _passes(b: BBox, relaxed: bool = False) -> bool:
            clat, clng = b.center_lat, b.center_lng
            is_high = _is_high_elevation_region(clat, clng)
            is_low = _is_low_elevation_region(clat, clng)

            if elevation_m > 3500:
                if relaxed:
                    return not is_low
                return is_high
            if elevation_m > 2500:
                return not is_low
            if elevation_m < 100:
                if relaxed:
                    return not is_high
                return is_low
            return True

        filtered = [b for b in bboxes if _passes(b, relaxed=False)]
        if filtered:
            return filtered

        relaxed = [b for b in bboxes if _passes(b, relaxed=True)]
        return relaxed


# ═══════════════════════════════════════════════════════════════════════════
# JSON parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_geocot_json(text: str) -> dict:
    """Extract structured JSON from GeoCoT output text (CoT-safe).

    Handles both code-fenced JSON and raw JSON at end of Thinking chain-of-thought.
    """
    # 1) Markdown code block — greedy to capture full multi-line JSON
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) Last JSON-like block in text (CoT puts answer JSON at the end)
    matches = list(re.finditer(r'\{[^{}]*\}', text))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
            if any(k in obj for k in ['climate_zone', 'language_script', 'scene_type',
                                        'terrain_type', 'architecture_style']):
                return obj
        except json.JSONDecodeError:
            continue

    # 3) Fallback: try tail of long text (CoT reasoning, answer at the end)
    if len(text) > 400:
        tail = text[-800:]
        for m in re.finditer(r'\{[^{}]*\}', tail):
            try:
                obj = json.loads(m.group(0))
                if any(k in obj for k in ['climate_zone', 'language_script',
                                            'terrain_type', 'vegetation_zone']):
                    return obj
            except json.JSONDecodeError:
                continue

    return {}


def parse_simple_visual_json(text: str) -> dict:
    """Extract simple visual answers from VLM output."""
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            obj = json.loads(m.group(0))
            if any(k in obj for k in ['scene', 'soil_color', 'vegetation', 'terrain', 'sky']):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


# ── Simple visual answer → GeoKB element mapping ─────────────────────────

# Each dict maps answer letter → (element_category, element_value)
# None value means "can't determine" — skip this element

_SCENE_MAP = {
    'A': ('scene_type', 'city_street'),
    'B': ('scene_type', 'mountain_road'),
    'C': ('scene_type', 'mountain_trail_karst'),  # generic mountain trail
    'D': ('scene_type', 'desert'),
    'E': ('scene_type', 'grassland'),
    'F': ('scene_type', 'coastal'),
    'G': ('scene_type', 'river_valley'),
    'H': ('scene_type', 'wilderness_forest'),
    'I': ('scene_type', 'farmland'),
    'J': ('scene_type', 'rural_village'),
    'K': ('scene_type', 'snow_mountain'),  # added in COMPOUND_SCENES
    'L': None,
}

_SOIL_COLOR_MAP = {
    'A': ('soil_color', 'red'),
    'B': ('soil_color', 'yellow'),
    'C': ('soil_color', 'brown'),
    'D': ('soil_color', 'black'),
    'E': ('soil_color', 'grey'),
    'F': ('soil_color', 'white'),
    'G': None,
}

_BUILDING_MAP = {
    'A': None,  # no buildings
    'B': ('architecture_style', 'modern_glass'),
    'C': ('architecture_style', 'modern_residential'),
    'D': ('architecture_style', 'modern_residential'),  # old residential → generic
    'E': ('architecture_style', 'hui_style'),  # traditional Chinese → hui_style
    'F': ('architecture_style', 'tibetan_stone'),
    'G': None,
}

# Urbanization inferred from building presence
def _building_to_urbanization(answer: str) -> tuple:
    if answer == 'A':
        return ('urbanization', 'wilderness')
    elif answer in ('B', 'C'):
        return ('urbanization', 'metropolis')
    elif answer in ('D', 'E'):
        return ('urbanization', 'rural')
    elif answer == 'F':
        return ('urbanization', 'rural')
    return None

_VEGETATION_MAP = {
    'A': ('vegetation_zone', 'broadleaf_evergreen'),  # dense forest → broadleaf
    'B': ('vegetation_zone', 'mixed_forest'),  # sparse trees → mixed
    'C': ('vegetation_zone', 'grassland'),
    'D': ('vegetation_zone', 'desert_scrub'),
    'E': ('vegetation_zone', 'broadleaf_evergreen'),  # farmland → cropland-like
    'F': ('vegetation_zone', 'tropical_rainforest'),
    'G': ('vegetation_zone', 'conifer_forest'),
    'H': ('vegetation_zone', 'broadleaf_evergreen'),
}

_SKY_MAP = {
    'A': ('sky_quality', 'clear_blue'),
    'B': ('sky_quality', 'grey_hazy'),  # cloudy → general
    'C': ('sky_quality', 'grey_hazy'),
    'D': ('sky_quality', 'grey_hazy'),
    'E': ('sky_quality', 'dusty_yellow'),
    'F': ('sky_quality', 'thick_fog'),
}

_WATER_MAP = {
    'A': None,
    'B': ('water_type', 'river'),
    'C': ('water_type', 'lake'),
    'D': ('water_type', 'ocean'),
    'E': ('water_type', 'waterfall'),
    'F': ('water_type', 'glacier'),
}

_TEXT_MAP = {
    'A': None,
    'B': ('language_script', 'simplified_chinese'),
    'C': ('language_script', 'tibetan'),
    'D': ('language_script', 'uyghur_arabic'),
    'E': ('language_script', 'simplified_chinese'),  # bilingual → CN dominant
}

_TERRAIN_MAP = {
    'A': ('terrain_type', 'urban_flat'),
    'B': ('terrain_type', 'rolling_hills'),
    'C': ('terrain_type', 'sharp_mountains'),
    'D': ('terrain_type', 'plateau'),
    'E': ('terrain_type', 'sharp_mountains'),  # canyon → mountain-like
}

_ROCK_MAP = {
    'A': None,
    'B': ('mountain_rock_type', 'limestone_karst'),  # grey-white rock → karst
    'C': ('mountain_rock_type', 'red_sandstone_danxia'),
    'D': ('mountain_rock_type', 'granite_spheroidal'),  # dark grey → granite
    'E': ('mountain_rock_type', 'granite_spheroidal'),  # yellow-brown → generic
}

_ELEVATION_MAP = {
    'A': ('elevation_estimate_m', 100),
    'B': ('elevation_estimate_m', 600),
    'C': ('elevation_estimate_m', 1750),
    'D': ('elevation_estimate_m', 3250),
    'E': ('elevation_estimate_m', 5000),
}


def simple_answers_to_elements(answers: dict) -> dict:
    """Convert simple visual answers (letter codes) to GeoKB-element format.

    Input: {"scene": "C", "soil_color": "A", "building": "A", ...}
    Output: {"scene_type": "mountain_trail_karst", "soil_color": "red", ...}
    """
    elements = {}

    mappings = [
        (answers.get('scene'), _SCENE_MAP),
        (answers.get('soil_color'), _SOIL_COLOR_MAP),
        (answers.get('vegetation'), _VEGETATION_MAP),
        (answers.get('sky'), _SKY_MAP),
        (answers.get('water'), _WATER_MAP),
        (answers.get('text'), _TEXT_MAP),
        (answers.get('terrain'), _TERRAIN_MAP),
        (answers.get('rock'), _ROCK_MAP),
        (answers.get('elevation_perception'), _ELEVATION_MAP),
    ]

    for answer, mapping in mappings:
        if answer and answer in mapping:
            result = mapping[answer]
            if result is not None:
                key, val = result
                elements[key] = val

    # Building → architecture + urbanization
    building_answer = answers.get('building', '')
    if building_answer and building_answer in _BUILDING_MAP:
        result = _BUILDING_MAP[building_answer]
        if result is not None:
            key, val = result
            elements[key] = val

    urb = _building_to_urbanization(building_answer)
    if urb is not None:
        key, val = urb
        # Only set urbanization if scene type doesn't already imply it
        if key not in elements:
            elements[key] = val

    # Water visibility derived from water answer
    water_answer = answers.get('water', '')
    if water_answer and water_answer != 'A':
        elements['water_visible'] = True
    elif water_answer == 'A':
        elements['water_visible'] = False

    return elements


def _is_valid_element_value(val) -> bool:
    """Filter out sentinel values that GeoKB can't map."""
    if val is None:
        return False
    if isinstance(val, str) and val.lower() in ("none_visible", "null", "n/a", "none", ""):
        return False
    if isinstance(val, list) and len(val) == 0:
        return False
    return True


def extract_all_elements(macro_json: dict, regional_json: dict, local_json: dict) -> dict:
    """Merge structured output from all three GeoCoT stages.

    Filters out sentinel values ('none_visible', 'null', empty list, etc.)
    that GeoKB has no mapping for.
    """
    elements = {}
    for key in ['climate_zone', 'terrain_type', 'vegetation_zone', 'urbanization',
                 'building_height', 'pavement_type', 'ruled_out_features']:
        val = macro_json.get(key)
        if _is_valid_element_value(val):
            elements[key] = val
    for key in ['language_script', 'architecture_style', 'tree_species',
                 'mountain_rock_type', 'infrastructure_tags', 'sky_quality',
                 'likely_provinces']:
        val = regional_json.get(key)
        if _is_valid_element_value(val):
            elements[key] = val
    # visible_text: none_visible is a valid answer (no text in image)
    vt = regional_json.get('visible_text')
    if vt is not None and vt != '' and vt != 'null' and vt != 'none' and vt != 'n/a':
        elements['visible_text'] = vt
    if _is_valid_element_value(regional_json.get('rock_type')):
        elements['mountain_rock_type'] = regional_json['rock_type']
    for key in ['landform_detail', 'rock_color', 'soil_color',
                 'water_visible', 'water_type', 'distinctive_features',
                 'scene_type']:
        val = local_json.get(key)
        if _is_valid_element_value(val):
            elements[key] = val
    for j in [macro_json, regional_json, local_json]:
        for k in ['elevation_estimate_m', 'altitude_estimate_m']:
            val = j.get(k)
            if _is_valid_element_value(val):
                elements['elevation_estimate_m'] = val
                break
    return elements


# ═══════════════════════════════════════════════════════════════════════════
# Sensor anchor pre-filter
# ═══════════════════════════════════════════════════════════════════════════

def _sensor_anchor_from_elevation(
    dem, elevation_m: float, res_deg: float = 0.25
) -> list[BBox]:
    """Find all regions in China compatible with a sensor elevation reading.

    Samples the DEM grid at ~res_deg spacing, finds cells within ±50% of
    the sensor reading, and clusters adjacent compatible cells into bboxes.

    This acts as a PHYSICAL TRUTH FILTER — VLM elements that point to
    incompatible elevation regions are discarded.
    """
    # Adaptive tolerance: relative for high elevations, absolute for low
    # ±50% at 1000m = 500-1500m (good), but ±50% at 10m = 5-15m (too narrow)
    # Use a floor of ±200m so coastal/lowland readings don't over-constrain
    rel_tolerance = 0.50
    abs_floor = 200.0  # meters
    elev_lo = min(elevation_m * (1.0 - rel_tolerance), elevation_m - abs_floor)
    elev_hi = max(elevation_m * (1.0 + rel_tolerance), elevation_m + abs_floor)

    # Sample the grid at ~res_deg resolution
    lat_step = int(res_deg / dem.res_deg)
    lng_step = int(res_deg / dem.res_deg)
    lat_step = max(lat_step, 1)
    lng_step = max(lng_step, 1)

    subsampled = dem.grid[::lat_step, ::lng_step]

    # Compatible cell mask
    mask = (subsampled >= elev_lo) & (subsampled <= elev_hi)

    if not mask.any():
        return []  # No compatible region — sensor reading is very unusual

    # Find contiguous compatible regions via simple connected-components
    # (8-connected on the subsampled grid)
    from collections import deque
    visited = set()
    regions = []

    n_rows, n_cols = mask.shape
    for r0 in range(n_rows):
        for c0 in range(n_cols):
            if not mask[r0, c0] or (r0, c0) in visited:
                continue
            # BFS flood fill
            q = deque([(r0, c0)])
            visited.add((r0, c0))
            r_min, r_max = r0, r0
            c_min, c_max = c0, c0
            while q:
                r, c = q.popleft()
                r_min, r_max = min(r_min, r), max(r_max, r)
                c_min, c_max = min(c_min, c), max(c_max, c)
                for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols:
                        if mask[nr, nc] and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))

            # Convert grid indices back to lat/lng
            lat_min = dem.lat_min + r_min * lat_step * dem.res_deg
            lat_max = dem.lat_min + (r_max + 1) * lat_step * dem.res_deg
            lng_min = dem.lng_min + c_min * lng_step * dem.res_deg
            lng_max = dem.lng_min + (c_max + 1) * lng_step * dem.res_deg

            # Only keep regions of meaningful size
            region_cells = (r_max - r_min + 1) * (c_max - c_min + 1)
            regions.append(BBox(
                    lat_min=max(lat_min, 17.0), lat_max=min(lat_max, 55.0),
                    lng_min=max(lng_min, 72.0), lng_max=min(lng_max, 136.0),
                    label=f"elevation_{elevation_m:.0f}m",
                ))

    if not regions:
        # Fallback: single bbox covering all compatible cells
        rows, cols = mask.nonzero()
        if len(rows) > 0:
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            lat_min = dem.lat_min + r_min * lat_step * dem.res_deg
            lat_max = dem.lat_min + (r_max + 1) * lat_step * dem.res_deg
            lng_min = dem.lng_min + c_min * lng_step * dem.res_deg
            lng_max = dem.lng_min + (c_max + 1) * lng_step * dem.res_deg
            regions.append(BBox(
                lat_min=lat_min, lat_max=lat_max,
                lng_min=lng_min, lng_max=lng_max,
                label=f"elevation_{elevation_m:.0f}m",
            ))

    return regions


def _sensor_anchor_from_climate(
    sensor_temperature_c: float, sensor_humidity_pct: float
) -> list[BBox]:
    """Find regions in China compatible with temperature + humidity readings.

    Uses the ClimateValidator physical model to check each province.
    Returns list of compatible province bboxes.
    """
    from .geo_kb import PROVINCE_BBOXES

    climate = get_climate()
    dem = get_dem()
    if not climate:
        return []

    compatible = []
    for name, bbox_list in PROVINCE_BBOXES.items():
        bbox = bbox_list[0] if isinstance(bbox_list, list) else bbox_list
        # Sample center and corners of province
        clat, clng = bbox.center_lat, bbox.center_lng
        elev = dem.query(clat, clng) if dem else 500
        if elev is None:
            elev = 500
        month = 7  # default summer

        # Check temperature
        temp_ok = True
        if sensor_temperature_c is not None:
            t_lo, t_hi = climate.estimate_temperature_range(clat, clng, elev, month)
            # Tight tolerance for pre-filter: ±8°C (was ±15°C, too loose)
            if not (t_lo - 8 <= sensor_temperature_c <= t_hi + 8):
                temp_ok = False

        # Check humidity
        humid_ok = True
        if sensor_humidity_pct is not None:
            h_lo, h_hi = climate.estimate_humidity_range(clat, clng, month)
            # Tight tolerance: ±10% (was ±20%, too loose)
            if not (h_lo - 10 <= sensor_humidity_pct <= h_hi + 10):
                humid_ok = False

        if temp_ok and humid_ok:
            compatible.append(bbox)

    return compatible


# ═══════════════════════════════════════════════════════════════════════════
# Core fusion algorithm (GeoCoT-first refinement architecture)
# ═══════════════════════════════════════════════════════════════════════════

def _score_element_consistency(
    geocot_prediction: tuple[float, float],
    elements: dict,
) -> tuple[list[str], list[str], float]:
    """Check which elements' GeoKB bboxes contain the GeoCoT prediction.

    Returns:
      confirmed: element categories whose bbox contains GeoCoT point
      questioned: element categories whose bbox does NOT contain GeoCoT point
      consist_score: (confirmed - questioned) / total, range [-1, 1]
    """
    glat, glng = geocot_prediction
    confirmed = []
    questioned = []

    for cat, val in elements.items():
        if val is None or val == "" or val == []:
            continue
        if isinstance(val, dict):
            continue
        bboxes = get_bboxes_for_element(cat, val)
        if not bboxes:
            continue
        inside = any(
            b.lat_min <= glat <= b.lat_max and b.lng_min <= glng <= b.lng_max
            for b in bboxes
        )
        if inside:
            confirmed.append(cat)
        else:
            questioned.append(cat)

    total = len(confirmed) + len(questioned)
    if total == 0:
        return [], [], 0.0
    consist_score = (len(confirmed) - len(questioned)) / total
    return confirmed, questioned, consist_score


def _check_sensor_compatibility_full(
    geocot_prediction: tuple[float, float],
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
) -> tuple[bool, str, float]:
    """Full sensor verification at GeoCoT coordinate.

    Checks elevation (DEM), temperature, and humidity against sensor readings.
    Returns (any_ok, message, penalty) where penalty ∈ [0, -0.45].
    """
    glat, glng = geocot_prediction
    dem = get_dem()
    climate = get_climate()
    sensor_results = []
    penalty = 0.0

    # ── Elevation check ───────────────────────────────────────────────
    if sensor_elevation_m is not None and dem is not None:
        dem_elev = dem.query(glat, glng)
        if dem_elev is not None:
            if dem_elev > 0 and sensor_elevation_m > 0:
                ratio = dem_elev / sensor_elevation_m
                if 0.5 <= ratio <= 2.0:
                    sensor_results.append(f"DEM={dem_elev:.0f}m vs {sensor_elevation_m:.0f}m OK")
                else:
                    sensor_results.append(f"DEM={dem_elev:.0f}m vs {sensor_elevation_m:.0f}m MISMATCH")
                    penalty -= 0.15
            else:
                if abs(dem_elev - sensor_elevation_m) <= 300:
                    sensor_results.append(f"DEM={dem_elev:.0f}m vs {sensor_elevation_m:.0f}m OK")
                else:
                    sensor_results.append(f"DEM={dem_elev:.0f}m vs {sensor_elevation_m:.0f}m MISMATCH")
                    penalty -= 0.15
        else:
            sensor_results.append("DEM no data at point")
    elif sensor_elevation_m is not None and dem is None:
        sensor_results.append("DEM unavailable")

    # ── Temperature check ─────────────────────────────────────────────
    if sensor_temperature_c is not None and climate is not None:
        dem_elev = dem.query(glat, glng) if dem else 500
        elev_for_temp = dem_elev if dem_elev else 500
        t_lo, t_hi = climate.estimate_temperature_range(glat, glng, elev_for_temp, month=7)
        if t_lo - 10 <= sensor_temperature_c <= t_hi + 10:
            sensor_results.append(f"T={sensor_temperature_c:.0f}C vs [{t_lo:.0f},{t_hi:.0f}]C OK")
        else:
            sensor_results.append(f"T={sensor_temperature_c:.0f}C vs [{t_lo:.0f},{t_hi:.0f}]C MISMATCH")
            penalty -= 0.15
    elif sensor_temperature_c is not None and climate is None:
        sensor_results.append("Climate model unavailable")

    # ── Humidity check ────────────────────────────────────────────────
    if sensor_humidity_pct is not None and climate is not None:
        h_lo, h_hi = climate.estimate_humidity_range(glat, glng, month=7)
        if h_lo - 15 <= sensor_humidity_pct <= h_hi + 15:
            sensor_results.append(f"RH={sensor_humidity_pct:.0f}% vs [{h_lo:.0f},{h_hi:.0f}]% OK")
        else:
            sensor_results.append(f"RH={sensor_humidity_pct:.0f}% vs [{h_lo:.0f},{h_hi:.0f}]% MISMATCH")
            penalty -= 0.15
    elif sensor_humidity_pct is not None and climate is None:
        sensor_results.append("Climate model unavailable")

    if not sensor_results:
        return True, "no sensor data", 0.0

    msg = " | ".join(sensor_results)
    return penalty > -0.15, msg, penalty


def _refine_near_geocot(
    geocot_lat: float,
    geocot_lng: float,
    search_radius_km: float,
    refine_categories: list[str],
    elements: dict,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
    compound_bboxes: Optional[list[BBox]] = None,
) -> tuple[float, float, float]:
    """Search near GeoCoT anchor for best point matching elements + sensors.

    1. Intersect element bboxes (non-broad) + compound scene bboxes within search window
    2. Multi-objective DEM + climate sampling at 0.05 grid → best match
    3. Return (refined_lat, refined_lng, uncertainty_km)
    """
    deg_radius = search_radius_km / 111.0

    # Collect bboxes for refinement (skip BROAD_CATEGORIES)
    all_bboxes = []
    for cat in refine_categories:
        if cat in BROAD_CATEGORIES:
            continue
        val = elements.get(cat)
        if val is None or isinstance(val, dict):
            continue
        bboxes = get_bboxes_for_element(cat, val)
        all_bboxes.extend(bboxes)

    # Add compound scene bboxes (high-precision constraints)
    if compound_bboxes:
        all_bboxes.extend(compound_bboxes)

    # Search window around GeoCoT prediction
    search_bbox = BBox(
        lat_min=max(17.0, geocot_lat - deg_radius),
        lat_max=min(55.0, geocot_lat + deg_radius),
        lng_min=max(72.0, geocot_lng - deg_radius),
        lng_max=min(136.0, geocot_lng + deg_radius),
        label="search_window",
    )

    if all_bboxes:
        candidate = intersect_bboxes([search_bbox], all_bboxes)
    else:
        candidate = [search_bbox]

    if not candidate:
        candidate = [search_bbox]

    dem = get_dem()
    climate = get_climate()

    # No sensor data → centroid of candidate
    if sensor_elevation_m is None and sensor_temperature_c is None and sensor_humidity_pct is None:
        clat, clng, radius = compute_bbox_center_radius(candidate)
        return clat, clng, min(radius, search_radius_km)

    # Multi-objective sampling
    best_point = None
    best_score = float('inf')
    points = []

    step = 0.05
    for b in candidate:
        lat = b.lat_min
        while lat <= b.lat_max:
            lng = b.lng_min
            while lng <= b.lng_max:
                score = 0.0
                n_components = 0

                # Elevation component
                if sensor_elevation_m is not None and dem is not None:
                    elev = dem.query(lat, lng)
                    if elev is not None:
                        if sensor_elevation_m > 0:
                            ratio = max(elev, sensor_elevation_m) / max(min(elev, sensor_elevation_m), 1)
                            score += (ratio - 1.0) * 3  # normalize: ratio 1.0=0, 2.0=3
                        else:
                            score += abs(elev - sensor_elevation_m) / 300.0
                        n_components += 1
                    else:
                        score += 2.0  # penalty for no data
                        n_components += 1

                # Temperature component
                if sensor_temperature_c is not None and climate is not None:
                    elev_for_temp = dem.query(lat, lng) if dem else 500
                    if elev_for_temp is None:
                        elev_for_temp = 500
                    t_lo, t_hi = climate.estimate_temperature_range(lat, lng, elev_for_temp, month=7)
                    t_mid = (t_lo + t_hi) / 2
                    t_range = max(t_hi - t_lo, 1)
                    score += abs(sensor_temperature_c - t_mid) / t_range
                    n_components += 1

                # Humidity component
                if sensor_humidity_pct is not None and climate is not None:
                    h_lo, h_hi = climate.estimate_humidity_range(lat, lng, month=7)
                    if h_lo <= sensor_humidity_pct <= h_hi:
                        pass  # perfect, score += 0
                    else:
                        dist = min(abs(sensor_humidity_pct - h_lo), abs(sensor_humidity_pct - h_hi))
                        score += dist / 30.0
                    n_components += 1

                if n_components > 0:
                    score /= n_components
                    points.append((lat, lng, score))
                    if score < best_score:
                        best_score = score
                        best_point = (lat, lng)
                lng += step
            lat += step

    if best_point and points:
        # Inverse-score weighted centroid (better points get higher weight)
        max_score = max(p[2] for p in points) if points else 1.0
        weights = [1.0 / max(p[2] + 0.1, 0.01) for p in points]
        total_w = sum(weights)
        weighted_lat = sum(p[0] * w for p, w in zip(points, weights)) / total_w
        weighted_lng = sum(p[1] * w for p, w in zip(points, weights)) / total_w

        dists = [haversine_km(weighted_lat, weighted_lng, p[0], p[1]) for p in points]
        uncertainty = float(np.mean(dists) + np.std(dists)) if len(dists) > 1 else search_radius_km * 0.3
        uncertainty = max(min(uncertainty, search_radius_km), 10.0)

        return weighted_lat, weighted_lng, uncertainty

    clat, clng, radius = compute_bbox_center_radius(candidate)
    return clat, clng, min(radius, search_radius_km)


def _fallback_bbox_centroid(
    elements: dict,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
) -> FusionResult:
    """Legacy fallback: element bbox intersection → centroid.

    Used when GeoCoT prediction is unavailable or strongly contradicted.
    Applies elevation + climate filtering when sensor data is available.
    """
    ex = []

    strong = {}
    for cat, val in elements.items():
        if val is None or val == "" or val == []:
            continue
        if isinstance(val, dict):
            continue
        bboxes = get_bboxes_for_element(cat, val)
        if not bboxes:
            continue
        if cat in BROAD_CATEGORIES:
            continue
        strong[cat] = bboxes
        ex.append(f"[{cat}] {val} -> {len(bboxes)} region(s)")

    if not strong:
        return FusionResult(35.0, 105.0, 2500.0, 0.0,
                            explanation_parts=["[FALLBACK] No geographic elements identified."])

    sorted_items = sorted(strong.items(), key=lambda x: len(x[1]))
    active_bboxes = list(sorted_items[0][1])
    active_cats = sorted_items[0][0]

    for cat, bboxes in sorted_items[1:]:
        inter = intersect_bboxes(active_bboxes, bboxes)
        if inter:
            active_bboxes = inter
            active_cats += f" + {cat}"

    # ── Sensor filtering ───────────────────────────────────────────────
    if sensor_elevation_m is not None:
        pre = len(active_bboxes)
        filtered = filter_by_elevation(active_bboxes, sensor_elevation_m)
        if filtered:
            active_bboxes = filtered
            ex.append(f"[FALLBACK-ELEV] {pre}→{len(active_bboxes)} regions after elevation filter")
        else:
            ex.append(f"[FALLBACK-ELEV] ALL {pre} regions excluded by elevation, keeping unfiltered")

    if (sensor_temperature_c is not None or sensor_humidity_pct is not None) and active_bboxes:
        dem = get_dem()
        pre = len(active_bboxes)
        filtered = filter_by_climate(
            active_bboxes, sensor_temperature_c, sensor_humidity_pct, dem)
        if filtered:
            active_bboxes = filtered
            ex.append(f"[FALLBACK-CLIMATE] {pre}→{len(active_bboxes)} regions after climate filter")
        else:
            ex.append(f"[FALLBACK-CLIMATE] ALL {pre} regions excluded by climate, keeping unfiltered")

    clat, clng, radius = compute_bbox_center_radius(active_bboxes)
    area = sum(bbox_area(b) for b in active_bboxes)
    confidence = max(0.05, min(0.50, 0.40 * len(strong) / max(1, len(elements))))
    if sensor_elevation_m is not None:
        confidence += 0.05

    ex.append(f"[FALLBACK] Bbox centroid: ({clat:.2f}, {clng:.2f}), radius={radius:.0f}km, area={area:,.0f}km2")

    candidate = CandidateRegion(clat, clng, radius, confidence,
                                active_bboxes, list(strong.keys()), [])
    return FusionResult(clat, clng, radius, confidence, candidate, ex)


def fuse_elements(
    elements: dict,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
    geocot_prediction: Optional[tuple[float, float]] = None,
) -> FusionResult:
    """Fuse geographic elements with GeoCoT-first refinement architecture.

    GeoCoT coordinate = initial anchor. Elements + sensors = refinement signals.

    Three-case decision:
      Case A (consist_score >= 0.0 AND sensor compatible):
        Anchor on GeoCoT, tighten uncertainty via element bbox intersection.
      Case B (consist_score >= -0.3):
        Refine near GeoCoT within 150-300km using confirmed elements + DEM.
      Case C (else):
        Fall back to element bbox centroid (legacy logic).
    """
    ex = []

    # ── Step 0: No GeoCoT prediction → fallback ─────────────────────────
    if geocot_prediction is None:
        ex.append("[GEOCO-ABSENT] No GeoCoT prediction, using fallback")
        result = _fallback_bbox_centroid(elements, sensor_elevation_m,
                                          sensor_temperature_c, sensor_humidity_pct)
        result.explanation_parts = ex + result.explanation_parts
        return result

    glat, glng = geocot_prediction
    ex.append(f"[GEOCO-ANCHOR] Prediction: ({glat:.4f}, {glng:.4f})")

    # ── Step 1: Element consistency scoring ─────────────────────────────
    confirmed, questioned, consist_score = _score_element_consistency(
        geocot_prediction, elements)
    ex.append(f"[CONSISTENCY] score={consist_score:+.2f} "
              f"confirmed={confirmed} questioned={questioned}")

    # ── Step 2: Compound scene matching (fuzzy) ────────────────────────
    compound_matches = match_compound_scenes_soft(elements, min_ratio=0.5)
    if compound_matches:
        scene_names = [f"{m[0]}({m[2]:.0%})" for m in compound_matches[:3]]
        ex.append(f"[COMPOUND] {len(compound_matches)} partial match(es): {', '.join(scene_names)}")

    # ── Step 3: Sensor verification (full: elevation + temperature + humidity) ─
    sensor_ok, sensor_msg, sensor_penalty = _check_sensor_compatibility_full(
        geocot_prediction, sensor_elevation_m,
        sensor_temperature_c, sensor_humidity_pct)
    ex.append(f"[SENSOR] {sensor_msg}")

    # Adjust consistency with sensor signal
    adjusted_score = consist_score + sensor_penalty
    if sensor_penalty < 0:
        ex.append(f"[SENSOR-PENALTY] penalty={sensor_penalty:+.2f}, adjusted score: {adjusted_score:+.2f}")

    # ── Step 4: Decision ─────────────────────────────────────────────────

    if adjusted_score >= 0.0:
        # ═══ CASE A: GeoCoT Anchoring ════════════════════════════════════
        ex.append("[CASE-A] GeoCoT anchoring — elements confirm prediction")

        # Collect non-broad confirmed element bboxes
        confirmed_bboxes = []
        for cat in confirmed:
            if cat in BROAD_CATEGORIES:
                continue
            val = elements.get(cat)
            if val is not None and not isinstance(val, dict):
                bboxes = get_bboxes_for_element(cat, val)
                if bboxes:
                    confirmed_bboxes.append(bboxes)

        # ── Integrate compound scenes for uncertainty tightening ────────
        compound_used = False
        if compound_matches:
            # Use high-confidence compound scenes (ratio ≥ 0.75)
            high_conf_scenes = [(n, b, r) for n, b, r in compound_matches if r >= 0.75]
            if high_conf_scenes:
                compound_used = True
                for scene_name, scene_bboxes, ratio in high_conf_scenes:
                    ex.append(f"[COMPOUND-A] {scene_name} ({ratio:.0%}) → {len(scene_bboxes)} tight bbox(es)")
                    confirmed_bboxes.append(list(scene_bboxes))

        if confirmed_bboxes:
            active = confirmed_bboxes[0]
            for bboxes in confirmed_bboxes[1:]:
                inter = intersect_bboxes(active, bboxes)
                if inter:
                    active = inter
            area = sum(bbox_area(b) for b in active)
            radius = math.sqrt(area / math.pi) * 0.5
            # Tighter cap when compound scenes are active
            cap = 200.0 if compound_used else 500.0
            radius = max(min(radius, cap), 15.0)
        else:
            active = []
            area = 0
            radius = 150.0

        confidence = 0.55 + 0.05 * min(len(confirmed), 6) + 0.10 * (1 if sensor_ok else 0)
        if sensor_elevation_m is not None:
            confidence += 0.05
        if compound_used:
            confidence += 0.10  # bonus for compound scene match
        confidence = max(0.05, min(confidence, 0.90))

        ex.append(f"[CASE-A] Anchored at ({glat:.4f}, {glng:.4f}), "
                  f"uncertainty={radius:.0f}km, confidence={confidence:.1%}"
                  + (" [compound]" if compound_used else ""))

        candidate = CandidateRegion(glat, glng, radius, confidence,
                                    active, confirmed, questioned)
        return FusionResult(glat, glng, radius, confidence, candidate, ex)

    elif adjusted_score >= -0.3:
        # ═══ CASE B: Near-Refinement ═════════════════════════════════════
        ex.append("[CASE-B] Near-refinement — searching around GeoCoT anchor")

        search_radius = 150.0 if adjusted_score >= -0.15 else 300.0

        # Use confirmed + non-broad questioned elements
        refine_cats = list(confirmed) + [q for q in questioned
                                          if q not in BROAD_CATEGORIES]
        if not refine_cats:
            refine_cats = list(confirmed)

        # Collect compound scene bboxes (lower threshold for Case B: ratio ≥ 0.4)
        compound_bboxes = []
        if compound_matches:
            for scene_name, scene_bboxes, ratio in compound_matches:
                if ratio >= 0.4:
                    ex.append(f"[COMPOUND-B] {scene_name} ({ratio:.0%}) bboxes added to search")
                    compound_bboxes.extend(scene_bboxes)

        rlat, rlng, uncertainty = _refine_near_geocot(
            glat, glng, search_radius, refine_cats, elements,
            sensor_elevation_m=sensor_elevation_m,
            sensor_temperature_c=sensor_temperature_c,
            sensor_humidity_pct=sensor_humidity_pct,
            compound_bboxes=compound_bboxes if compound_bboxes else None,
        )

        shift = haversine_km(glat, glng, rlat, rlng)
        ex.append(f"[CASE-B] Refined to ({rlat:.4f}, {rlng:.4f}), "
                  f"shift={shift:.0f}km, uncertainty={uncertainty:.0f}km")

        confidence = 0.40 + 0.04 * min(len(confirmed), 5) + 0.10 * (1 if sensor_ok else 0)
        if sensor_elevation_m is not None:
            confidence += 0.05
        confidence = max(0.05, min(confidence, 0.80))

        # Collect element bboxes for candidate
        element_bboxes = []
        for cat in refine_cats:
            if cat in BROAD_CATEGORIES:
                continue
            val = elements.get(cat)
            if val is not None and not isinstance(val, dict):
                bboxes = get_bboxes_for_element(cat, val)
                element_bboxes.extend(bboxes)
        if compound_bboxes:
            element_bboxes.extend(compound_bboxes)

        candidate = CandidateRegion(rlat, rlng, uncertainty, confidence,
                                    element_bboxes, refine_cats, questioned)
        return FusionResult(rlat, rlng, uncertainty, confidence, candidate, ex)

    else:
        # ═══ CASE C: Strong Contradiction → Fallback ═════════════════════
        ex.append("[CASE-C] Strong contradiction — falling back to bbox centroid")
        result = _fallback_bbox_centroid(elements, sensor_elevation_m,
                                          sensor_temperature_c, sensor_humidity_pct)
        result.explanation_parts = ex + result.explanation_parts
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Climate zone validation via physical sensor constraints
# ═══════════════════════════════════════════════════════════════════════════

# Köppen-Geiger informed: each climate zone's physically possible bounds.
# Ranges are deliberately WIDE to accommodate seasonal variation.
# Only physically IMPOSSIBLE combinations trigger hard veto.
CLIMATE_PHYSICS = {
    "tropical": {       # e.g. Hainan, S Taiwan
        "temp_range": (18, 38),      # year-round warm
        "humid_range": (55, 100),
        "elev_max": 2000,            # lapse rate makes high tropical impossible
    },
    "subtropical": {    # e.g. Yangtze basin, SE China
        "temp_range": (8, 38),
        "humid_range": (50, 100),
        "elev_max": 3000,
    },
    "temperate": {      # e.g. N China Plain, NE China
        "temp_range": (-15, 35),
        "humid_range": (25, 85),
        "elev_max": 3500,
    },
    "arid": {           # e.g. Xinjiang, Gansu, Inner Mongolia
        "temp_range": (-20, 42),
        "humid_range": (5, 55),      # arid = low humidity is defining trait
        "elev_max": 5000,
    },
    "alpine": {         # e.g. Tibet Plateau, Tianshan
        "temp_range": (-25, 22),     # even summer is cool at altitude
        "humid_range": (15, 75),
        "elev_min": 2000,            # alpine needs high elevation
    },
    "boreal": {         # e.g. N Heilongjiang, N Inner Mongolia
        "temp_range": (-30, 25),
        "humid_range": (30, 75),
        "elev_min": 300,
        "elev_max": 3000,
    },
}

# Standard atmospheric lapse rate: temperature drops ~6.5°C per 1000m elevation gain
LAPSE_RATE = 6.5  # °C / 1000m


def _sensor_climate_zone(
    temperature_c: float,
    humidity_pct: float,
    elevation_m: float,
) -> str:
    """Infer most likely climate zone from physical sensor readings alone.

    Uses temperature-elevation-humidity constraints. Conservative: only
    returns a zone when the sensor signal is unambiguous.
    """
    # ── Elevation-first rules ──────────────────────────────────────────
    if elevation_m > 2500:
        return "alpine"
    if elevation_m > 1800 and temperature_c < 18:
        return "alpine"

    # ── Humidity-first rules ───────────────────────────────────────────
    if humidity_pct < 40 and temperature_c > 15:
        return "arid"

    # ── Temperature-first rules ────────────────────────────────────────
    if temperature_c > 25:
        if humidity_pct > 65:
            if elevation_m < 2000:
                return "subtropical"
            return "temperate"  # high elev + warm → temperate highland
        if humidity_pct < 45:
            return "arid"
        return "subtropical"  # warm + moderate humidity → subtropical by default

    if temperature_c < 5:
        if elevation_m > 2000:
            return "alpine"
        if elevation_m > 500:
            return "boreal"
        return "temperate"

    # ── Ambiguous zone ─────────────────────────────────────────────────
    return ""  # can't determine, don't override VLM


def validate_climate_zone(
    vlm_climate: str,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
) -> tuple[str, float, str]:
    """Cross-validate VLM climate_zone against physical sensor readings.

    Two-tier approach:
      Tier 1 (hard veto): VLM output is physically IMPOSSIBLE → replace.
      Tier 2 (soft down-weight): VLM output is suspicious → reduce weight.

    Returns:
      corrected_zone: climate zone to use (may be same as vlm_climate)
      weight_mul: multiplier for this element's scoring weight (0.0–1.0)
      reason: explanation string (empty if no issue)
    """
    if not vlm_climate or vlm_climate not in CLIMATE_PHYSICS:
        return vlm_climate, 1.0, ""

    physics = CLIMATE_PHYSICS[vlm_climate]
    reasons = []

    # ── Tier 1: Hard impossibility checks ──────────────────────────────

    # Check elevation bounds
    if sensor_elevation_m is not None:
        elev_min = physics.get("elev_min", 0)
        elev_max = physics.get("elev_max", 9000)
        if sensor_elevation_m < elev_min:
            reasons.append(f"elev {sensor_elevation_m:.0f}m < min {elev_min}m")
        if sensor_elevation_m > elev_max:
            reasons.append(f"elev {sensor_elevation_m:.0f}m > max {elev_max}m")

    # Check temperature bounds
    if sensor_temperature_c is not None:
        t_lo, t_hi = physics["temp_range"]
        if sensor_temperature_c < t_lo - 5:  # 5°C slack for weather anomalies
            reasons.append(f"temp {sensor_temperature_c:.0f}°C << range [{t_lo},{t_hi}]")
        if sensor_temperature_c > t_hi + 5:
            reasons.append(f"temp {sensor_temperature_c:.0f}°C >> range [{t_lo},{t_hi}]")

    # Check humidity bounds (arid's defining trait is low humidity)
    if sensor_humidity_pct is not None:
        h_lo, h_hi = physics["humid_range"]
        if sensor_humidity_pct < h_lo - 10:
            reasons.append(f"humid {sensor_humidity_pct:.0f}% << range [{h_lo},{h_hi}]")
        if vlm_climate == "arid" and sensor_humidity_pct > h_hi + 10:
            reasons.append(f"arid but humid={sensor_humidity_pct:.0f}% >> {h_hi}%")

    # Temperature-elevation combined check (lapse rate physics)
    if sensor_temperature_c is not None and sensor_elevation_m is not None:
        # At sea level, 29°C. At 4000m, lapse rate says ~29 - 4*6.5 = 3°C.
        sea_level_equiv = sensor_temperature_c + (sensor_elevation_m / 1000) * LAPSE_RATE
        if vlm_climate == "alpine" and sea_level_equiv > 35:
            reasons.append(f"sea-level-equiv {sea_level_equiv:.0f}°C at {sensor_elevation_m:.0f}m impossible for alpine")
        if vlm_climate == "tropical" and sea_level_equiv < 15:
            reasons.append(f"sea-level-equiv {sea_level_equiv:.0f}°C too cold for tropical")

    # Check extreme severity: elevation > 2× bound or humidity > 2× for arid
    extreme = False
    if sensor_elevation_m is not None:
        elev_min = physics.get("elev_min", 0)
        elev_max = physics.get("elev_max", 9000)
        if sensor_elevation_m > elev_max * 1.5 and elev_max > 0:
            extreme = True
        if sensor_elevation_m < elev_min * 0.5 and elev_min > 0:
            extreme = True
    if vlm_climate == "arid" and sensor_humidity_pct is not None:
        h_hi = physics["humid_range"][1]
        if sensor_humidity_pct > h_hi * 1.4:
            extreme = True

    if len(reasons) >= 2 or extreme:
        # Multiple or extreme violations → override with sensor-inferred zone
        inferred = _sensor_climate_zone(
            sensor_temperature_c or 20, sensor_humidity_pct or 50,
            sensor_elevation_m or 500)
        if inferred and inferred != vlm_climate:
            reason_str = "; ".join(reasons)
            tag = "[CLIMATE-VETO]" if len(reasons) >= 2 else "[CLIMATE-VETO-EXTREME]"
            return inferred, 0.1, f"{tag} {vlm_climate} impossible ({reason_str}) -> {inferred}"

    # ── Tier 2: Soft inconsistency → down-weight ──────────────────────

    if len(reasons) == 1:
        return vlm_climate, 0.3, f"[CLIMATE-SOFT] {vlm_climate} suspicious ({reasons[0]}) -> weight reduced"

    # Check for "borderline" cases: temperate but sensor suggests subtropical
    if (vlm_climate == "temperate" and sensor_temperature_c is not None
            and sensor_humidity_pct is not None):
        # Combined temp+humid threshold: warm AND humid → subtropical
        if sensor_temperature_c >= 26 and sensor_humidity_pct > 72:
            inferred = _sensor_climate_zone(
                sensor_temperature_c, sensor_humidity_pct, sensor_elevation_m or 500)
            if inferred in ("subtropical", "tropical"):
                return inferred, 0.5, f"[CLIMATE-SOFT] temperate unlikely at {sensor_temperature_c:.0f}C + {sensor_humidity_pct:.0f}%RH -> {inferred}"

    if (vlm_climate == "subtropical" and sensor_temperature_c is not None
            and sensor_temperature_c < 10):
        return vlm_climate, 0.3, f"[CLIMATE-SOFT] subtropical but {sensor_temperature_c:.0f}°C, possible winter"

    return vlm_climate, 1.0, ""


# ═══════════════════════════════════════════════════════════════════════════
# Sensor-First Fusion (v2.2)
# ═══════════════════════════════════════════════════════════════════════════

# Categories with strong physical-geography spatial constraint
HARD_ELEMENT_CATEGORIES = {
    "climate_zone",      # alpine vs subtropical = completely different latitudes
    "terrain_type",      # karst_peaks vs desert_dunes = different regions
    "vegetation_zone",   # tropical_rainforest vs conifer_forest = different biomes
}


def score_sensor_consistency(
    elements: dict,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
) -> float:
    """Aggregate sensor-consistency score ∈ [0,1] for Best-of-N candidate selection."""
    weights = compute_element_weights(
        elements,
        sensor_elevation_m=sensor_elevation_m,
        sensor_temperature_c=sensor_temperature_c,
        sensor_humidity_pct=sensor_humidity_pct,
    )
    if not weights:
        return 0.0
    return sum(weights.values()) / len(weights)


# ═══════════════════════════════════════════════════════════════════════════
# Element-level agreement check (for Adaptive N=2/N=3 stability)
# ═══════════════════════════════════════════════════════════════════════════

_KEY_ELEMENTS = [
    "climate_zone",
    "terrain_type",
    "vegetation_zone",
    "urbanization",
    "elevation_estimate_m",
    "likely_provinces",
]


def compute_element_agreement(elements1: dict, elements2: dict) -> dict:
    """Compare key localization elements between two VLM runs.

    Returns dict with:
      - agreement_score: float [0, 1] — overall agreement across all key elements
      - per_element: dict[str, bool] — whether each element agrees
      - disagree_count: int — number of key elements that differ
    """
    per_element = {}
    agree_count = 0
    total = 0

    for key in _KEY_ELEMENTS:
        v1 = elements1.get(key)
        v2 = elements2.get(key)
        total += 1

        if key == "elevation_estimate_m":
            # Compare midpoint within 50% tolerance of the average
            try:
                if isinstance(v1, (list, tuple)) and len(v1) == 2:
                    m1 = (float(v1[0]) + float(v1[1])) / 2
                else:
                    m1 = float(str(v1).replace("[", "").replace("]", "").split(",")[0])
                if isinstance(v2, (list, tuple)) and len(v2) == 2:
                    m2 = (float(v2[0]) + float(v2[1])) / 2
                else:
                    m2 = float(str(v2).replace("[", "").replace("]", "").split(",")[0])
                avg_m = (m1 + m2) / 2
                tol = max(100, avg_m * 0.5)  # at least 100m tolerance
                per_element[key] = abs(m1 - m2) <= tol
            except (ValueError, TypeError, IndexError):
                per_element[key] = False
        elif key == "likely_provinces":
            # Jaccard similarity >= 0.3 → agree
            s1 = set(v1) if isinstance(v1, list) else set()
            s2 = set(v2) if isinstance(v2, list) else set()
            if not s1 and not s2:
                per_element[key] = True  # both empty → agree
            elif not s1 or not s2:
                per_element[key] = False
            else:
                jaccard = len(s1 & s2) / len(s1 | s2)
                per_element[key] = jaccard >= 0.3
        else:
            # String comparison (case-insensitive, normalize underscores)
            s1 = str(v1).lower().replace(" ", "_").strip("[]'\"")
            s2 = str(v2).lower().replace(" ", "_").strip("[]'\"")
            per_element[key] = s1 == s2

        if per_element[key]:
            agree_count += 1

    return {
        "agreement_score": agree_count / total if total else 0.0,
        "per_element": per_element,
        "disagree_count": total - agree_count,
    }


def select_best_elements(candidates: list[tuple[dict, float, dict]]) -> tuple[dict, float]:
    """Select the best element set from N candidates.

    candidates: list of (elements, sensor_score, agreement_dict)
    Returns: (best_elements, best_score)

    Priority: prefer candidates whose elements agree with at least one other.
    Among agreeing candidates, pick highest sensor_score.
    If none agree, pick highest sensor_score.
    """
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1]

    # Find candidates that agree with at least one other (agreement_score >= 0.5)
    consensus = [(e, s, a) for e, s, a in candidates
                 if a.get("agreement_score", 0) >= 0.5]
    pool = consensus if consensus else candidates
    best = max(pool, key=lambda x: x[1])
    return best[0], best[1]


def fuse_elements_v3(
    elements: dict,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
    geocot_prediction: Optional[tuple[float, float]] = None,
) -> FusionResult:
    """Scoring-first spatial reasoning: all evidence votes, no hard pruning.

    ARCHITECTURE (v2.3):
      1. VLM element extraction → GeoKB spatial bbox lookup
      2. Elements + compound scenes → coarse search space (geom intersect only)
      3. Sensors (elevation, temp, humidity) → scoring ONLY (Gaussian kernel)
      4. Grid-search over search space → MAP location

    No ConstraintSolver. No hard pruning. Every piece of evidence contributes
    through the scoring function — sensors and elements are equal voters.
    """
    ex = []
    dem = get_dem()
    climate = get_climate()

    # ── Step 1: Log sensor data ────────────────────────────────────────────
    if sensor_elevation_m is not None:
        ex.append(f"[SENSOR] Elevation: {sensor_elevation_m:.0f}m")
    if sensor_temperature_c is not None:
        ex.append(f"[SENSOR] Temperature: {sensor_temperature_c:.1f}°C")
    if sensor_humidity_pct is not None:
        ex.append(f"[SENSOR] Humidity: {sensor_humidity_pct:.0f}%")

    # ── Step 1.25: Climate zone validation ─────────────────────────────
    # Cross-check VLM climate_zone against physical sensor readings.
    # Two-tier: hard veto (physically impossible → replace) or soft down-weight.
    climate_weight_mul = 1.0
    if sensor_temperature_c is not None or sensor_elevation_m is not None:
        vlm_climate = str(elements.get("climate_zone", ""))
        corrected, climate_weight_mul, climate_reason = validate_climate_zone(
            vlm_climate,
            sensor_elevation_m=sensor_elevation_m,
            sensor_temperature_c=sensor_temperature_c,
            sensor_humidity_pct=sensor_humidity_pct,
        )
        if corrected != vlm_climate and corrected:
            elements["climate_zone"] = corrected
            ex.append(climate_reason)
        elif climate_weight_mul < 1.0:
            ex.append(climate_reason)

    # ── Step 1.5: Elevation pre-pruning (sensor-first spatial narrowing) ────
    # Use DEM reverse lookup to find regions compatible with the elevation
    # reading. These form the FOUNDATION of the search space — for distinctive
    # elevations (high mountains, deep basins) this drastically shrinks the
    # search area and yields finer grid resolution.
    # Element bboxes are UNIONED on top, so GeoKB gaps never exclude truth.
    elevation_bboxes = []
    if sensor_elevation_m is not None and dem is not None:
        elevation_bboxes = dem.find_elevation_range(
            sensor_elevation_m, tolerance_pct=0.35)
        if elevation_bboxes:
            elev_area = bbox_total_area_km2(elevation_bboxes)
            pct_of_china = elev_area / 9_600_000 * 100
            ex.append(f"[ELEV-PRUNE] {sensor_elevation_m:.0f}m ±35% → "
                      f"{len(elevation_bboxes)} region(s), {elev_area:,.0f} km² "
                      f"({pct_of_china:.1f}% of China)")
        else:
            ex.append("[ELEV-PRUNE] No compatible elevation regions found")

    # ── Step 2: Geographic elements → bboxes ──────────────────────────────
    all_element_bboxes = []
    if not elements:
        ex.append("[ELEMENTS] No elements")
        search_bboxes = list(elevation_bboxes) if elevation_bboxes else [BBox(18.0, 54.0, 73.0, 135.5, "China")]
    else:
        all_element_bboxes = []
        n_found = 0
        n_no_bbox = 0
        for cat, val in elements.items():
            if cat in BROAD_CATEGORIES:
                continue
            if val is None or isinstance(val, (dict, list)):
                continue
            bboxes = get_bboxes_for_element(cat, val)
            if not bboxes:
                n_no_bbox += 1
                continue
            all_element_bboxes.append(bboxes)
            n_found += 1

        ex.append(f"[ELEMENTS] {n_found} elements with GeoKB bboxes"
                  + (f" (+{n_no_bbox} no bbox)" if n_no_bbox else ""))

        # Search space: intersect element bboxes with elevation-compatible
        # regions when available. This prevents low-elevation false positives
        # (e.g. Taklamakan at 1000m) from entering the search space when
        # sensor elevation is high (e.g. 4300m at Qaidam).
        search_bboxes = list(elevation_bboxes) if elevation_bboxes else []
        if elevation_bboxes:
            for bboxes in all_element_bboxes:
                trimmed = intersect_bbox_lists([bboxes, elevation_bboxes])
                if trimmed:
                    search_bboxes.extend(trimmed)
                else:
                    search_bboxes.extend(bboxes)
        else:
            for bboxes in all_element_bboxes:
                search_bboxes.extend(bboxes)
        if not search_bboxes:
            search_bboxes = [BBox(18.0, 54.0, 73.0, 135.5, "China")]

    # ── Step 3: Compound scenes ───────────────────────────────────────────
    compound_dispersed = False
    compound_xval_failed = False
    compound_matches = match_compound_scenes_soft(elements, min_ratio=0.5)
    if compound_matches:
        top3 = [f"{m[0]}({m[2]:.0%})" for m in compound_matches[:3]]
        ex.append(f"[COMPOUND] {len(compound_matches)} match(es): {', '.join(top3)}")

        # High-confidence compound scenes (≥75%) provide tight spatial anchors.
        # When element bboxes are broad (urban China ≈ 25M km²), compound
        # scenes are the ONLY discriminative spatial signal. Restrict the
        # search space to their union to prevent grid search from drifting
        # into regions with no realistic match.
        # Use 100%-match compound scenes to RESTRICT the search space
        # (all conditions match exactly = strong spatial anchor).
        # 75-99% scenes only contribute bboxes for scoring, not restriction.
        #
        # SAFETY: Cross-validate perfect scenes with VLM's own likely_provinces.
        # If VLM says "Shanghai" but perfect-match compound scenes all point to
        # Zhengzhou/Taiyuan, the VLM made an element error (e.g. temperate vs
        # subtropical). Don't restrict — let scoring sort it out.
        perfect_scenes = [(n, b, r) for n, b, r in compound_matches if r >= 1.0]
        if perfect_scenes:
            # ── Cross-validation #1: province consistency ─────────────────
            province_bboxes = []
            likely_val = elements.get("likely_provinces", [])
            if isinstance(likely_val, list) and likely_val:
                for prov in likely_val:
                    pb = get_bboxes_for_element("likely_provinces", prov)
                    if pb:
                        province_bboxes.extend(pb)

            compound_only = []
            for _, bboxes, _ in perfect_scenes:
                compound_only.extend(bboxes)
            compound_only = _dedup_bboxes(compound_only)

            trust_restriction = True
            xval_failed = False
            if province_bboxes:
                overlap = intersect_bbox_lists([compound_only, province_bboxes])
                if not overlap:
                    trust_restriction = False
                    xval_failed = True
                    compound_xval_failed = True
                    ex.append(f"[COMPOUND-XVAL] perfect scenes contradict "
                              f"likely_provinces={likely_val} → weak restriction")

            # ── Cross-validation #2: spatial dispersion ───────────────────
            if trust_restriction and len(perfect_scenes) >= 2:
                centers = []
                for _, bboxes, _ in perfect_scenes:
                    for b in bboxes:
                        centers.append((b.center_lat, b.center_lng))
                if len(centers) >= 2:
                    max_dist = 0
                    for i in range(len(centers)):
                        for j in range(i + 1, len(centers)):
                            d = haversine_km(centers[i][0], centers[i][1],
                                            centers[j][0], centers[j][1])
                            max_dist = max(max_dist, d)
                    if max_dist > 500:
                        trust_restriction = False
                        compound_dispersed = True
                        ex.append(f"[COMPOUND-DISPERSE] perfect scenes scattered "
                                  f"{max_dist:.0f}km → not restricting")

            if trust_restriction:
                # Intersect compound-only space with elevation bboxes
                if elevation_bboxes:
                    trimmed = intersect_bbox_lists([compound_only, elevation_bboxes])
                    if trimmed:
                        compound_only = _dedup_bboxes(trimmed)
                # Keep only element-derived bboxes that intersect with perfect scenes
                restricted = list(compound_only)
                for bboxes in all_element_bboxes:
                    inter = intersect_bbox_lists([bboxes, compound_only])
                    if inter:
                        restricted.extend(inter)
                search_bboxes = _dedup_bboxes(restricted)
                ex.append(f"[COMPOUND-RESTRICT] {len(perfect_scenes)} perfect-match scenes "
                          f"→ search area {bbox_total_area_km2(search_bboxes):,.0f} km²")
            elif xval_failed:
                # XVAL contradicts: compound elements don't align with VLM provinces.
                # Strategy: trust VLM provinces over compounds. Intersect province
                # bboxes with strong compound bboxes for a tight search space.
                # If no intersection, fall back to province bboxes alone.
                strong_scenes = [(n, b, r) for n, b, r in compound_matches if r >= 0.75]
                strong_bboxes = []
                for _, scene_bboxes, _ in strong_scenes:
                    strong_bboxes.extend(scene_bboxes)
                strong_bboxes = _dedup_bboxes(strong_bboxes)

                # Intersect with province bboxes (trust VLM provinces)
                if province_bboxes:
                    inter_prov = intersect_bbox_lists([strong_bboxes, province_bboxes])
                    if inter_prov:
                        strong_bboxes = _dedup_bboxes(inter_prov)
                        ex.append(f"[COMPOUND-XVAL-TIGHT] province ∩ strong compounds "
                                  f"→ {bbox_total_area_km2(strong_bboxes):,.0f} km²")
                    else:
                        # Complete disagreement → trust provinces alone
                        strong_bboxes = province_bboxes
                        ex.append(f"[COMPOUND-XVAL-PROV] no overlap → trust provinces "
                                  f"({bbox_total_area_km2(province_bboxes):,.0f} km²)")

                if elevation_bboxes:
                    trimmed = intersect_bbox_lists([strong_bboxes, elevation_bboxes])
                    if trimmed:
                        strong_bboxes = _dedup_bboxes(trimmed)
                restricted = list(strong_bboxes)
                for bboxes in all_element_bboxes:
                    inter = intersect_bbox_lists([bboxes, strong_bboxes])
                    if inter:
                        restricted.extend(inter)
                if restricted:
                    search_bboxes = _dedup_bboxes(restricted)
                else:
                    search_bboxes = strong_bboxes
                ex.append(f"[COMPOUND-XVAL-WEAK] {len(strong_scenes)} strong(≥75%) scenes "
                          f"→ search area {bbox_total_area_km2(search_bboxes):,.0f} km²")
            else:
                # DISPERSE: genuinely scattered → compound scenes can't restrict.
                # Fall back to per-element bbox intersection: each individual
                # element (climate, terrain, vegetation, ...) has a GeoKB spatial
                # footprint. Their intersection is tighter than province bonus alone.
                element_inter = None
                for bboxes in all_element_bboxes:
                    if element_inter is None:
                        element_inter = list(bboxes)
                    else:
                        inter = intersect_bbox_lists([element_inter, bboxes])
                        if inter:
                            element_inter = _dedup_bboxes(inter)
                        else:
                            element_inter = None
                            break
                if element_inter:
                    elem_area = bbox_total_area_km2(element_inter)
                    # Safety floor: too-small intersection likely excludes truth.
                    # Only trust the intersection if it's ≥50,000 km² (~province-sized).
                    if elem_area >= 50_000:
                        ex.append(f"[COMPOUND-DISPERSE] → element-bbox intersection: "
                                  f"{elem_area:,.0f} km²")
                        search_bboxes = list(element_inter)
                    else:
                        ex.append(f"[COMPOUND-DISPERSE] element intersection too small "
                                  f"({elem_area:,.0f} km²) → fall back to compound bboxes")
                        for _, scene_bboxes, _ in compound_matches:
                            search_bboxes.extend(scene_bboxes)
                else:
                    ex.append(f"[COMPOUND-DISPERSE] element intersection empty "
                              f"→ fall back to compound bboxes")
                    for _, scene_bboxes, _ in compound_matches:
                        search_bboxes.extend(scene_bboxes)
        else:
            # No perfect scenes → extend search with all compound bboxes
            for _, scene_bboxes, _ in compound_matches:
                search_bboxes.extend(scene_bboxes)
    else:
        ex.append("[COMPOUND] No compound scenes matched")

    # ── Broad XVAL: province-compound contradiction at any match level ────
    # The narrow XVAL above only checks 100%-match scenes. But 75% matches
    # can also be confidently wrong: VLM says "Guangdong" while compounds all
    # point to Sichuan/Yangtze. When VLM provinces and compound bboxes have
    # zero overlap, treat it as XVAL failure so the province prior activates.
    if not compound_xval_failed and not compound_dispersed and compound_matches:
        all_compound_bboxes = []
        for _, bboxes, _ in compound_matches:
            all_compound_bboxes.extend(bboxes)
        all_compound_bboxes = _dedup_bboxes(all_compound_bboxes)

        prov_bboxes = []
        likely_val = elements.get("likely_provinces", [])
        if isinstance(likely_val, list) and likely_val:
            for prov in likely_val:
                pb = get_bboxes_for_element("likely_provinces", prov)
                if pb:
                    prov_bboxes.extend(pb)

        if prov_bboxes and all_compound_bboxes:
            overlap = intersect_bbox_lists([all_compound_bboxes, prov_bboxes])
            if not overlap:
                compound_xval_failed = True
                ex.append(f"[COMPOUND-XVAL-BROAD] all compound bboxes contradict "
                          f"likely_provinces={likely_val} → province prior active")

    # ── Step 4: Coarse search space dedup ─────────────────────────────────
    search_bboxes = _dedup_bboxes(search_bboxes)
    search_area = bbox_total_area_km2(search_bboxes)
    ex.append(f"[SEARCH-AREA] Union: {len(search_bboxes)} region(s), {search_area:,.0f} km²")

    # ── Step 4.5: Negative constraints — subtract ruled-out feature bboxes ──
    # VLM is better at "I don't see X" than "this is not Province Y".
    # Feature bboxes are much smaller than provinces → lower risk of
    # accidentally excluding the correct location.
    # Safety: max 5 features, skip if contradicted by positive selections.
    MAX_RULED_OUT = 5
    FEATURE_CATS = ['climate_zone', 'terrain_type', 'vegetation_zone',
                     'urbanization', 'building_height', 'pavement_type']
    ruled_out_features = elements.get("ruled_out_features", [])
    if isinstance(ruled_out_features, list) and 1 <= len(ruled_out_features) <= MAX_RULED_OUT:
        positive_values = set()
        for cat in FEATURE_CATS:
            val = elements.get(cat)
            if val and isinstance(val, str):
                positive_values.add(val)
        safe_to_exclude = [f for f in ruled_out_features if f not in positive_values]
        if safe_to_exclude:
            ruled_out_bboxes = []
            for feat_val in safe_to_exclude:
                found = False
                for cat in FEATURE_CATS:
                    bboxes = get_bboxes_for_element(cat, feat_val)
                    if bboxes:
                        ruled_out_bboxes.extend(bboxes)
                        found = True
                        break
                if not found:
                    ex.append(f"[RULED-OUT] '{feat_val}' not in GeoKB, skipping")
            if ruled_out_bboxes:
                old_area = bbox_total_area_km2(search_bboxes)
                search_bboxes = subtract_bboxes(search_bboxes, ruled_out_bboxes)
                if search_bboxes:
                    new_area = bbox_total_area_km2(search_bboxes)
                    ex.append(f"[RULED-OUT] Excluded {len(safe_to_exclude)} feature(s) "
                              f"{safe_to_exclude} → {old_area:,.0f}→{new_area:,.0f} km²")
                else:
                    search_bboxes = [BBox(18.0, 54.0, 73.0, 135.5, "China")]
                    ex.append(f"[RULED-OUT] Subtraction emptied search space, "
                              f"falling back to full China")
        else:
            ex.append(f"[RULED-OUT] All features contradict positive selections, skipping")
    elif isinstance(ruled_out_features, list) and len(ruled_out_features) > MAX_RULED_OUT:
        ex.append(f"[RULED-OUT] Skipped: {len(ruled_out_features)} features > {MAX_RULED_OUT} cap")

    # ── Step 5: GeoCoT cross-validation + adaptive weight ────────────────
    # When elevation sensors can't prune (low elevation → >80% of China),
    # GeoCoT proximity bonus is critical. When elevation prunes well
    # (<30% of China), sensors are reliable, GeoCoT bonus is minimal.
    geo_weight = 0.0
    if geocot_prediction is not None:
        glat, glng = geocot_prediction
        in_region = any(
            b.lat_min <= glat <= b.lat_max and b.lng_min <= glng <= b.lng_max
            for b in search_bboxes
        )
        if in_region:
            ex.append(f"[GEOCO-XVAL] ({glat:.2f}, {glng:.2f}) inside search space")
        else:
            dist = _distance_to_nearest_bbox(glat, glng, search_bboxes)
            ex.append(f"[GEOCO-XVAL] ({glat:.2f}, {glng:.2f}) outside by {dist:.0f}km")

        # Adaptive weight: low elevation = heavy GeoCoT reliance
        if elevation_bboxes:
            elev_pct = bbox_total_area_km2(elevation_bboxes) / 9_600_000 * 100
            if elev_pct < 30:
                geo_weight = 0.05   # sensors prune well, light GeoCoT touch
            elif elev_pct < 60:
                geo_weight = 0.15
            elif elev_pct < 80:
                geo_weight = 0.25
            else:
                geo_weight = 0.35   # sensors can't prune, lean on GeoCoT
        else:
            # No elevation sensor → moderate GeoCoT reliance
            geo_weight = 0.20

        # ── Step 5.5: Anti-bias correction ──────────────────────────────
        # VLM models systematically predict Sichuan/Yunnan (28.5-30.0°N,
        # 101.0-104.5°E) for non-Sichuan images. When GeoCoT lands in this
        # region, cross-check VLM elements + sensors for exonerating evidence.
        bias_lat_lo, bias_lat_hi = 28.5, 30.0
        bias_lng_lo, bias_lng_hi = 101.0, 104.5
        if bias_lat_lo <= glat <= bias_lat_hi and bias_lng_lo <= glng <= bias_lng_hi:
            n_exonerate = 0
            # Climate check: Sichuan Basin is subtropical (temp 25-32°C, humid 65-85%)
            climate_val = str(elements.get("climate_zone", "")).lower()
            if climate_val in ("temperate", "boreal", "arid", "alpine"):
                n_exonerate += 1
            # Temperature check: Sichuan July temp is 25-32°C
            if sensor_temperature_c is not None:
                if sensor_temperature_c < 20 or sensor_temperature_c > 35:
                    n_exonerate += 1
            # Humidity check: Sichuan July humidity is 65-85%
            if sensor_humidity_pct is not None:
                if sensor_humidity_pct < 50:
                    n_exonerate += 1
            # Vegetation check: Sichuan Basin is broadleaf_evergreen
            veg_val = str(elements.get("vegetation_zone", "")).lower()
            if veg_val in ("broadleaf_deciduous", "conifer_forest", "desert_scrub",
                           "grassland", "alpine_meadow"):
                n_exonerate += 1
            # Soil check: Sichuan Basin is red/purple soil
            soil_val = str(elements.get("soil_color", "")).lower()
            if soil_val in ("black", "yellow", "grey", "white"):
                n_exonerate += 1
            # Architecture check
            arch_val = str(elements.get("architecture_style", "")).lower()
            if arch_val in ("tibetan_stone", "hui_style", "russian", "mongolian_yurt"):
                n_exonerate += 1

            # Check if strong compound scenes point AWAY from the bias region
            compound_away = False
            for scene_name, scene_bboxes, ratio in compound_matches:
                if ratio < 0.75:
                    continue
                # Check if ALL scene bboxes are outside the bias region
                all_outside = all(
                    b.lat_max < bias_lat_lo or b.lat_min > bias_lat_hi or
                    b.lng_max < bias_lng_lo or b.lng_min > bias_lng_hi
                    for b in scene_bboxes
                )
                if all_outside:
                    compound_away = True
                    ex.append(f"[ANTI-BIAS] compound '{scene_name}' ({ratio:.0%}) "
                              f"points outside Sichuan → reducing GeoCoT trust")
                    break

            if n_exonerate >= 2 or (n_exonerate >= 1 and compound_away):
                geo_weight *= 0.25
                ex.append(f"[ANTI-BIAS] {n_exonerate} elements + compound "
                          f"exonerate Sichuan → geo_weight slashed to {geo_weight:.3f}")
            elif n_exonerate >= 1 or compound_away:
                geo_weight *= 0.50
                ex.append(f"[ANTI-BIAS] {'compound' if compound_away else 'element'} "
                          f"suggests non-Sichuan → geo_weight reduced to {geo_weight:.3f}")

        ex.append(f"[GEOCO-WEIGHT] adaptive weight={geo_weight:.2f} (elevation prune "
                  f"{elev_pct:.0f}% of China)" if elevation_bboxes else
                  f"[GEOCO-WEIGHT] adaptive weight={geo_weight:.2f} (no elevation sensor)")

    # ── Step 6: Grid-search scoring ──────────────────────────────────────
    element_infos = precompute_element_infos(elements, search_area)
    element_weights = compute_element_weights(
        elements, sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct)

    # ── Climate weight adjustment ───────────────────────────────────────
    # Apply climate validation multiplier from Step 1.25.
    if climate_weight_mul < 1.0:
        # Find the climate_zone key in element_weights and scale it
        climate_keys = [(cat, val) for (cat, val) in element_weights
                        if cat == "climate_zone"]
        for key in climate_keys:
            old_w = element_weights[key]
            element_weights[key] = old_w * climate_weight_mul
        if climate_keys:
            ex.append(f"[CLIMATE-WEIGHT] climate_zone weight ×{climate_weight_mul:.2f}")

    # ── Elevation veto: when VLM elevation_estimate_m is wildly wrong,
    #     tighten search with sensor-anchored elevation pre-pruning ──────────
    elev_key = None
    for (cat, val) in element_weights:
        if cat == "elevation_estimate_m":
            elev_key = (cat, val)
            break
    if elev_key and sensor_elevation_m is not None and dem is not None:
        elev_w = element_weights.get(elev_key, 0.70)
        if elev_w < 0.15:
            # VLM elevation is extremely inconsistent with sensor → veto it
            # Re-run elevation pre-pruning with tighter tolerance (±25%)
            tight_bboxes = dem.find_elevation_range(
                sensor_elevation_m, tolerance_pct=0.25)
            if tight_bboxes:
                # Intersect tight elevation with current search space
                new_search = []
                for sb in search_bboxes:
                    trimmed = intersect_bbox_lists([[sb], tight_bboxes])
                    if trimmed:
                        new_search.extend(trimmed)
                if new_search:
                    old_area = bbox_total_area_km2(search_bboxes)
                    new_area = bbox_total_area_km2(new_search)
                    ex.append(f"[ELEV-VETO] VLM elevation wt={elev_w:.2f}, "
                              f"sensor veto → area {old_area:,.0f}→{new_area:,.0f} km²")
                    search_bboxes = _dedup_bboxes(new_search)
                    search_area = new_area
                    # Recompute element infos with tightened search area
                    element_infos = precompute_element_infos(elements, search_area)
                    # Also reduce geo_weight since sensors are more reliable
                    geo_weight *= 0.5

    # ── Step 6.5: Province-level prior when compound is ambiguous ─────────
    # When compound scenes are dispersed or contradict VLM provinces,
    # likely_provinces provides a soft spatial prior via evidence discounting.
    # More provinces listed → lower trust per province (discount = 1/N).
    # Soft bonus only: adds score for locations inside province bboxes,
    # never penalizes outside locations.
    province_bonus_bboxes = None
    province_bonus_weight = 0.0
    if compound_dispersed or compound_xval_failed:
        likely_val = elements.get("likely_provinces", [])
        if isinstance(likely_val, list) and likely_val:
            bonus_bboxes = []
            for prov in likely_val:
                pb = get_bboxes_for_element("likely_provinces", prov)
                if pb:
                    bonus_bboxes.extend(pb)
            if bonus_bboxes:
                province_bonus_bboxes = _dedup_bboxes(bonus_bboxes)
                discount = 1.0 / len(likely_val)
                province_bonus_weight = 0.20 * discount
                ex.append(f"[PROVINCE-BONUS] compound dispersed/xval → province prior active "
                          f"(discount={discount:.2f}, weight={province_bonus_weight:.3f})")

    # ── Step 6.6: Cluster bonus zones (E+B hybrid, DISPERSE only) ──────────
    # When compound scenes are geographically scattered (DISPERSE), DBSCAN
    # cluster their bboxes and assign soft scoring bonuses: dense clusters
    # get strong bonus, isolated bboxes get weak bonus. No area is ever
    # hard-excluded. Only active in DISPERSE (not XVAL) — when compound
    # scenes contradict VLM provinces, we don't trust compound clusters.
    cluster_bonus_zones = None
    if compound_dispersed and compound_matches:
        clusters = cluster_compound_bboxes(compound_matches, eps_km=300, min_samples=2)
        if clusters:
            cluster_bonus_zones = cluster_zones_to_bonus_zones(clusters)
            n_clusters = sum(1 for c in clusters if c[0] >= 0.3)
            ex.append(f"[CLUSTER-BONUS] DISPERSE → {len(clusters)} cluster(s), "
                      f"top weight={clusters[0][0]:.2f} → soft scoring zones active")

    # ── Step 6.7: Grid search ────────────────────────────────────────────
    # Multi-hypothesis for DISPERSE: independent search per geographic cluster.
    # Within each cluster, compound scenes are concentrated → elevation gating
    # works normally (compound_dispersed=False). Pick the globally best result.
    if compound_dispersed and cluster_bonus_zones and len(cluster_bonus_zones) >= 2:
        major_clusters = [(w, b) for (w, b) in cluster_bonus_zones if w >= 0.15]
        if len(major_clusters) >= 2:
            best_score = -float('inf')
            best_lat, best_lng, uncertainty = 35.0, 105.0, 2500.0
            all_points = []
            n_searched = 0
            hypotheses_log = []

            for cluster_weight, cluster_bbox in major_clusters:
                cluster_search = intersect_bbox_lists([[cluster_bbox], search_bboxes])
                if not cluster_search:
                    continue

                c_lat, c_lng, c_uncertainty, c_points = grid_search(
                    cluster_search,
                    elements, element_infos,
                    compound_matches if compound_matches else [],
                    sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                    dem, climate,
                    geocot_prediction=geocot_prediction,
                    geo_weight=geo_weight,
                    element_weights=element_weights,
                    province_bboxes=province_bonus_bboxes,
                    province_bonus_weight=province_bonus_weight,
                    cluster_bonus_zones=None,
                    compound_dispersed=False,
                )

                if c_points:
                    n_searched += 1
                    c_best_score = max(p[2] for p in c_points)
                    hypotheses_log.append(
                        f"cluster_w{cluster_weight:.2f}: ({c_lat:.3f},{c_lng:.3f}) "
                        f"score={c_best_score:.3f} uncert={c_uncertainty:.0f}km "
                        f"area={bbox_total_area_km2(cluster_search):,.0f}km²"
                    )
                    if c_best_score > best_score:
                        best_score = c_best_score
                        best_lat, best_lng, uncertainty = c_lat, c_lng, c_uncertainty
                        all_points = c_points

            if n_searched >= 2:
                # Score margin check: if top 2 clusters are within 10%,
                # the clusters are indistinguishable → fall back to single
                # search with cluster_bonus_zones (V4 behavior). This avoids
                # randomly picking between equally-plausible clusters.
                cluster_scores = []
                for h in hypotheses_log:
                    # Parse score from log: "cluster_wX.XX: (lat,lng) score=Y.YYY ..."
                    parts = h.split("score=")
                    if len(parts) > 1:
                        cluster_scores.append(float(parts[1].split()[0]))
                cluster_scores.sort(reverse=True)
                margin = (cluster_scores[0] - cluster_scores[1]) / max(cluster_scores[0], 0.001)

                if margin < 0.10:
                    ex.append(f"[MULTI-HYPOTHESIS] Top-2 clusters too close "
                              f"(margin={margin:.1%}<10%), falling back to single search")
                    for h in hypotheses_log:
                        ex.append(f"  {h}")
                    best_lat, best_lng, uncertainty, all_points = grid_search(
                        search_bboxes, elements, element_infos,
                        compound_matches if compound_matches else [],
                        sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                        dem, climate,
                        geocot_prediction=geocot_prediction,
                        geo_weight=geo_weight, element_weights=element_weights,
                        province_bboxes=province_bonus_bboxes,
                        province_bonus_weight=province_bonus_weight,
                        cluster_bonus_zones=cluster_bonus_zones,
                        compound_dispersed=compound_dispersed,
                    )
                else:
                    ex.append(f"[MULTI-HYPOTHESIS] DISPERSE → {n_searched} clusters, "
                              f"margin={margin:.1%} → using best cluster")
                    for h in hypotheses_log:
                        ex.append(f"  {h}")
            elif n_searched == 1:
                ex.append(f"[MULTI-HYPOTHESIS] Only 1 valid cluster, using it directly")
            else:
                # Fallback: no valid cluster search areas
                ex.append(f"[MULTI-HYPOTHESIS] No valid clusters, falling back to full search")
                best_lat, best_lng, uncertainty, all_points = grid_search(
                    search_bboxes, elements, element_infos,
                    compound_matches if compound_matches else [],
                    sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                    dem, climate,
                    geocot_prediction=geocot_prediction,
                    geo_weight=geo_weight, element_weights=element_weights,
                    province_bboxes=province_bonus_bboxes,
                    province_bonus_weight=province_bonus_weight,
                    cluster_bonus_zones=cluster_bonus_zones,
                    compound_dispersed=compound_dispersed,
                )
        else:
            # Not enough major clusters → single search with cluster bonuses
            best_lat, best_lng, uncertainty, all_points = grid_search(
                search_bboxes, elements, element_infos,
                compound_matches if compound_matches else [],
                sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                dem, climate,
                geocot_prediction=geocot_prediction,
                geo_weight=geo_weight, element_weights=element_weights,
                province_bboxes=province_bonus_bboxes,
                province_bonus_weight=province_bonus_weight,
                cluster_bonus_zones=cluster_bonus_zones,
                compound_dispersed=compound_dispersed,
            )
    else:
        # Normal mode: single grid search
        best_lat, best_lng, uncertainty, all_points = grid_search(
            search_bboxes, elements, element_infos,
            compound_matches if compound_matches else [],
            sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
            dem, climate,
            geocot_prediction=geocot_prediction,
            geo_weight=geo_weight, element_weights=element_weights,
            province_bboxes=province_bonus_bboxes,
            province_bonus_weight=province_bonus_weight,
            cluster_bonus_zones=cluster_bonus_zones,
            compound_dispersed=compound_dispersed,
        )
    ex.append(f"[GRID-SEARCH] {len(all_points)} cells scored, best=({best_lat:.4f}, {best_lng:.4f}) uncertainty={uncertainty:.0f}km")

    # ── Step 7: Confidence estimation ─────────────────────────────────────
    # Uses element quality (sensor consistency), not just count.
    #  - avg_weight: mean per-element sensor-consistency score [0.05, 1.0]
    #  - uncertainty: grid-search spread — larger = less confident
    #  - compound matches: tight spatial anchors boost confidence
    # Formula produces 0.15–0.85 instead of the old flat 0.80–0.85.
    n_comp = len(compound_matches)
    n_sensors = sum(1 for s in [sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct] if s is not None)
    avg_weight = sum(element_weights.values()) / max(1, len(element_weights))
    uncertainty_norm = max(0.0, min(1.0, (2000.0 - uncertainty) / 1900.0)) if uncertainty > 0 else 1.0
    compound_bonus = 0.12 * min(n_comp, 4) / 4.0

    confidence = (0.08
                  + 0.42 * avg_weight
                  + compound_bonus
                  + 0.05 * (n_sensors / 3.0)
                  + 0.15 * uncertainty_norm)
    confidence = max(0.06, min(confidence, 0.92))

    ex.append(f"[CONFIDENCE] avg_elem_wt={avg_weight:.3f} uncertainty={uncertainty:.0f}km "
              f"compounds={n_comp} → confidence={confidence:.2f}")

    # ── Step 8: Build FusionResult ───────────────────────────────────────
    active_names = [f"{cat}={val}" for (cat, val) in element_infos.keys()]
    if compound_matches:
        active_names.extend([f"compound_{m[0]}" for m in compound_matches])

    candidate = CandidateRegion(
        best_lat, best_lng,
        uncertainty, confidence,
        search_bboxes,
        active_elements=active_names,
        dropped_elements=[],
        explanation="\n".join(ex),
    )

    return FusionResult(
        best_lat, best_lng,
        uncertainty, confidence,
        candidate, ex,
    )


def _dedup_bboxes(bboxes: list[BBox]) -> list[BBox]:
    """Remove duplicate and fully-contained bboxes, keeping larger ones."""
    if len(bboxes) <= 1:
        return list(bboxes)
    # Sort by area descending
    sorted_boxes = sorted(bboxes, key=lambda b: b.area_km2, reverse=True)
    kept = []
    for b in sorted_boxes:
        if b.area_km2 <= 0:
            continue
        # Check if b is fully contained in any already-kept bbox
        contained = any(
            k.lat_min <= b.lat_min and k.lat_max >= b.lat_max and
            k.lng_min <= b.lng_min and k.lng_max >= b.lng_max
            for k in kept
        )
        if not contained:
            kept.append(b)
    return kept


def _distance_to_nearest_bbox(
    lat: float, lng: float, bboxes: list[BBox],
) -> float:
    """Minimum distance from a point to any point in any bbox."""
    if not bboxes:
        return 9999.0
    min_dist = float('inf')
    for b in bboxes:
        # Clamp point to bbox
        clamp_lat = max(b.lat_min, min(lat, b.lat_max))
        clamp_lng = max(b.lng_min, min(lng, b.lng_max))
        d = haversine_km(lat, lng, clamp_lat, clamp_lng)
        min_dist = min(min_dist, d)
    return min_dist


# ═══════════════════════════════════════════════════════════════════════════
# High-level API
# ═══════════════════════════════════════════════════════════════════════════

def run_fusion_pipeline(
    macro_text: str,
    regional_text: str,
    local_text: str,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
    geocot_prediction: Optional[tuple[float, float]] = None,
) -> FusionResult:
    """Complete pipeline: GeoCoT text → structured elements → fusion result.

    If geocot_prediction is not provided, attempts to parse it from
    local_text via the COORDINATES: line.
    """
    macro = parse_geocot_json(macro_text)
    regional = parse_geocot_json(regional_text)
    local = parse_geocot_json(local_text)
    elements = extract_all_elements(macro, regional, local)

    # Parse GeoCoT coordinate prediction from LOCAL text if not provided
    if geocot_prediction is None:
        m = re.search(r'COORDINATES:\s*([\d.]+)\s*,?\s*([\d.]+)', local_text)
        if m:
            geocot_prediction = (float(m.group(1)), float(m.group(2)))

    return fuse_elements(elements, sensor_elevation_m, sensor_temperature_c,
                         sensor_humidity_pct, geocot_prediction)


def run_fusion_pipeline_v3(
    macro_text: str,
    regional_text: str,
    local_text: str,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
    geocot_prediction: Optional[tuple[float, float]] = None,
) -> FusionResult:
    """Sensor-first pipeline: GeoCoT text -> elements -> constraint solver -> grid search.

    Uses fuse_elements_v3() which:
      1. DEM reverse lookup + climate filter = sensor constraints (hard)
      2. VLM element GeoKB bboxes = element constraints (hard/soft)
      3. ConstraintSolver intersection with graceful degradation
      4. Grid-search scoring within remaining region -> MAP location
    """
    macro = parse_geocot_json(macro_text)
    regional = parse_geocot_json(regional_text)
    local = parse_geocot_json(local_text)
    elements = extract_all_elements(macro, regional, local)

    # Parse GeoCoT coordinate prediction from LOCAL text if not provided
    if geocot_prediction is None:
        m = re.search(r'COORDINATES:\s*([\d.]+)\s*,?\s*([\d.]+)', local_text)
        if m:
            geocot_prediction = (float(m.group(1)), float(m.group(2)))

    return fuse_elements_v3(elements, sensor_elevation_m, sensor_temperature_c,
                            sensor_humidity_pct, geocot_prediction)


def run_simple_visual_pipeline(
    simple_text: str,
    sensor_elevation_m: Optional[float] = None,
    sensor_temperature_c: Optional[float] = None,
    sensor_humidity_pct: Optional[float] = None,
) -> FusionResult:
    """Simplified pipeline: single visual Q&A → elements → fusion result.

    Uses the simplified prompt where VLM answers direct visual questions
    (letter choices) instead of geographic abstractions.
    """
    answers = parse_simple_visual_json(simple_text)
    elements = simple_answers_to_elements(answers)
    return fuse_elements(elements, sensor_elevation_m, sensor_temperature_c,
                         sensor_humidity_pct, None)


# ── Reverse mapping (elements → simple answers) for testing ───────────────

def _elements_to_simple_answers(elements: dict) -> dict:
    """Convert GeoKB elements back to simple visual answers (for mock testing).

    This simulates what a VLM would answer with the simplified prompt,
    given perfect knowledge of the geographic elements.
    """
    answers = {}

    # Scene type → scene letter
    scene = elements.get('scene_type', '')
    scene_reverse = {v: k for k, v in _SCENE_MAP.items() if v is not None}
    for scene_key, (_, scene_val) in [(k, v) for k, v in _SCENE_MAP.items() if v is not None]:
        pass
    if scene:
        for letter, val in _SCENE_MAP.items():
            if val is not None and val[1] == scene:
                answers['scene'] = letter
                break

    # Soil color
    soil = elements.get('soil_color', '')
    if soil:
        for letter, val in _SOIL_COLOR_MAP.items():
            if val is not None and val[1] == soil:
                answers['soil_color'] = letter
                break

    # Building → architecture
    arch = elements.get('architecture_style', '')
    if arch:
        for letter, val in _BUILDING_MAP.items():
            if val is not None and val[1] == arch:
                answers['building'] = letter
                break
    elif elements.get('urbanization') in ('wilderness',):
        answers['building'] = 'A'  # no buildings

    # Vegetation
    veg = elements.get('vegetation_zone', '')
    if veg:
        for letter, val in _VEGETATION_MAP.items():
            if val is not None and val[1] == veg:
                answers['vegetation'] = letter
                break

    # Sky
    sky = elements.get('sky_quality', '')
    if sky:
        for letter, val in _SKY_MAP.items():
            if val is not None and val[1] == sky:
                answers['sky'] = letter
                break

    # Water
    water = elements.get('water_type', '')
    if water:
        for letter, val in _WATER_MAP.items():
            if val is not None and val[1] == water:
                answers['water'] = letter
                break
    elif elements.get('water_visible') is False:
        answers['water'] = 'A'

    # Text
    text = elements.get('language_script', '')
    if text:
        for letter, val in _TEXT_MAP.items():
            if val is not None and val[1] == text:
                answers['text'] = letter
                break

    # Terrain
    terrain = elements.get('terrain_type', '')
    if terrain:
        for letter, val in _TERRAIN_MAP.items():
            if val is not None and val[1] == terrain:
                answers['terrain'] = letter
                break

    # Rock
    rock = elements.get('mountain_rock_type', '')
    if rock:
        for letter, val in _ROCK_MAP.items():
            if val is not None and val[1] == rock:
                answers['rock'] = letter
                break

    # Elevation
    elev = elements.get('elevation_estimate_m')
    if elev is not None:
        if isinstance(elev, list):
            elev = sum(elev) / len(elev) if elev else 500
        if elev < 200:
            answers['elevation_perception'] = 'A'
        elif elev < 1000:
            answers['elevation_perception'] = 'B'
        elif elev < 2500:
            answers['elevation_perception'] = 'C'
        elif elev < 4000:
            answers['elevation_perception'] = 'D'
        else:
            answers['elevation_perception'] = 'E'

    return answers
