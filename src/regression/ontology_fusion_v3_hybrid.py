"""Ontology-based geographic fusion engine.

Replaces GeoKB bbox intersection with region fingerprint matching.
Key advantages:
  - No external GeoKB queries → ~100x faster (microseconds vs seconds)
  - Robust to individual field errors (soft matching, not hard intersection)
  - Sensor data as hard constraints, not just scoring hints
  - Ontological consistency validation detects impossible combinations
  - Hierarchical matching: coarse climate → regional terrain → local details
  - KG embedding similarity for semantic region matching (V2)
  - Hybrid mode: ontology region filtering + sensor grid refinement (V3)
"""

import json
import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Reuse FusionResult from element_fusion for drop-in compatibility
from .element_fusion import FusionResult, CandidateRegion

ONTOLOGY_PATH = Path(__file__).resolve().parent / "geo_ontology_v2.json"
EMBEDDING_PATH = Path(__file__).resolve().parent / "kg_embeddings.pkl"

# KG embedding blending weight (0 = fingerprint only, 1 = embedding only)
KG_BLEND_WEIGHT = 0.15

# ── DEM / climate imports (graceful fallback) ──────────────────────────
_DEM = None
_CLIMATE = None

def _get_dem():
    global _DEM
    if _DEM is None:
        try:
            from .dem_lookup import ChinaDEM
            dem_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "data", "dem", "china_elevation_005deg.npz")
            if os.path.exists(dem_path):
                _DEM = ChinaDEM(dem_path)
        except Exception:
            pass
    return _DEM

def _get_climate():
    global _CLIMATE
    if _CLIMATE is None:
        try:
            from .climate_lookup import ClimateValidator
            _CLIMATE = ClimateValidator()
        except Exception:
            pass
    return _CLIMATE


def _haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))

# Field hierarchy levels → weights (higher level = broader constraint)
# Level 0 (sensor) → hard filter, not scored
# Level 1 (macro) → strongest spatial signal
# Level 5 (local) → weakest individual signal, but valuable in aggregate
# Field weights calibrated by: geographic importance × VLM reliability (hit rate)
# Hit rates from 500-sample benchmark (diagnose_fusion_gap.py):
#   urbanization 100%, soil_color 99.8%, language_script 99.6%, sky_quality 99.6%,
#   terrain_type 98.6%, architecture_style 97.4%, climate_zone 97.2%,
#   scene_type 96.4%, pavement_type 95.0%, landform_detail 90.6%,
#   building_height 90.0%, water_type 85.4%, tree_species 81.8%,
#   vegetation_zone 72.4%, mountain_rock_type 7.8%
HIERARCHY_WEIGHTS = {
    "climate_zone": 0.95,        # 97.2% hit — macro climate, strong spatial signal
    "terrain_type": 0.85,        # 98.6% hit — terrain strongly constrains location
    "vegetation_zone": 0.25,     # 72.4% hit — VLM unreliable, reduce from 0.45
    "soil_color": 0.90,          # 99.8% hit — very reliable and discriminative
    "landform_detail": 0.70,     # 90.6% hit — good discrimination
    "mountain_rock_type": 0.05,  # 7.8% hit — VLM near-random, nearly zero weight
    "water_type": 0.55,          # 85.4% hit — water features, moderate reliability
    "urbanization": 0.60,        # 100% hit — perfect reliability
    "architecture_style": 0.60,  # 97.4% hit — strongly cultural, region-specific
    "language_script": 0.75,     # 99.6% hit — very strong cultural anchor
    "pavement_type": 0.45,       # 95.0% hit — weak signal but reliable
    "tree_species": 0.55,        # 81.8% hit — region-specific, moderate reliability
    "building_height": 0.45,     # 90.0% hit — modern development blurs signal
    "sky_quality": 0.40,         # 99.6% hit — weather-dependent, not stable signal
    "scene_type": 0.55,          # 96.4% hit — compound category, moderate signal
}

# Anchor fields: both geographically discriminative AND reliably predicted (>90% hit)
ANCHOR_FIELDS = {
    "language_script", "architecture_style",
    "soil_color", "climate_zone", "urbanization",
    "landform_detail", "terrain_type",
}

# Anchor fields get a multiplier when they match with high confidence
# These fields are geographically distinctive — correct prediction strongly
# narrows the region, wrong prediction strongly misleads.
ANCHOR_FIELDS = {
    "language_script", "architecture_style",
    "soil_color", "climate_zone", "tree_species",
    "landform_detail", "terrain_type",
}
ANCHOR_BOOST = 1.3  # multiplier when anchor field matches


@dataclass
class RegionMatch:
    """A single region match result."""
    region_id: str
    region_name: str
    score: float              # 0-1 fingerprint match score
    fingerprint_hits: int     # number of fields that matched fingerprint
    fingerprint_total: int    # total fields compared
    constraint_violations: list = field(default_factory=list)
    center_lat: float = 0.0
    center_lng: float = 0.0
    radius_km: float = 500.0


class OntologyFusion:
    """Ontology-based geographic fusion using region fingerprint matching."""

    def __init__(self, ontology_path: str | Path | None = None,
                 embedding_path: str | Path | None = None):
        path = Path(ontology_path) if ontology_path else ONTOLOGY_PATH
        with open(path, encoding="utf-8") as f:
            self.ontology = json.load(f)

        self.regions = self.ontology["regions"]
        self.constraints = self.ontology["constraints"]

        # Load TransE KG embeddings (optional)
        self._kg_embeddings = None
        self._region_emb = {}  # region_id → np.array
        emb_path = Path(embedding_path) if embedding_path else EMBEDDING_PATH
        if emb_path.exists():
            try:
                with open(emb_path, "rb") as f:
                    self._kg_embeddings = pickle.load(f)
                self._region_emb = {
                    rid: np.array(vec, dtype=np.float32)
                    for rid, vec in self._kg_embeddings["region_embeddings"].items()
                }
            except Exception:
                pass

        self._build_index()

    def _build_index(self):
        """Pre-build lookup indices for fast matching."""
        # Region ID → region dict
        self._region_by_id = {r["id"]: r for r in self.regions}

        # Impossible combinations as {(field1, value1, field2, value2): reason}
        self._impossible_pairs = {}
        for rule in self.constraints.get("impossible_combinations", []):
            fields_items = list(rule["fields"].items())
            if len(fields_items) == 2:
                (f1, v1), (f2, v2) = fields_items
                self._impossible_pairs[(f1, v1, f2, v2)] = rule["reason"]
                self._impossible_pairs[(f2, v2, f1, v1)] = rule["reason"]

        # Elevation → allowed climates lookup
        self._elevation_climate = self.constraints.get("sensor_constraints", {}).get(
            "elevation_to_climate", {})

        # Temperature → allowed climates lookup
        self._temperature_climate = self.constraints.get("sensor_constraints", {}).get(
            "temperature_to_climate", {})

        # IDF weights for field values (rare values get higher weight)
        self._field_value_idf = {}
        # Z-score stats: mean/std of fp_weight for each (field, value) across regions
        self._field_value_stats = {}  # (field, value) -> (mean_fp, std_fp)
        n_regions = len(self.regions)
        for region in self.regions:
            for field, dist in region.get("fingerprint", {}).items():
                for val in dist:
                    if (field, val) not in self._field_value_idf:
                        df = sum(1 for r in self.regions
                                if val in r.get("fingerprint", {}).get(field, {}))
                        self._field_value_idf[(field, val)] = math.log(
                            (n_regions + 1) / (df + 1))
                        # Precompute fp_weight stats for Z-scoring
                        weights = []
                        for r in self.regions:
                            w = r.get("fingerprint", {}).get(field, {}).get(val, 0.0)
                            weights.append(w)
                        mean_w = sum(weights) / len(weights)
                        var_w = sum((w - mean_w)**2 for w in weights) / len(weights)
                        std_w = math.sqrt(max(var_w, 1e-12))
                        self._field_value_stats[(field, val)] = (mean_w, std_w)

        # ── Reverse sensor mappings: elevation → valid field values ───
        # From scoring.py lookup tables: terrain/vegetation/climate ranges
        self._elev_to_terrains = {}  # elev → set of compatible terrain values
        self._elev_to_vegetations = {}  # elev → set of compatible vegetation values

        _TERRAIN_ELEV = {
            "urban_flat": (0, 1500), "farmland_plain": (0, 1200),
            "rolling_hills": (50, 2500), "sharp_mountains": (800, 5500),
            "karst_peaks": (0, 2500), "sandstone_pillars": (200, 2500),
            "desert_dunes": (-100, 2000), "grassland_steppe": (0, 3500),
            "plateau": (2000, 5000),
        }
        _VEGETATION_ELEV = {
            "tropical_rainforest": (0, 1500, 22, 35),
            "broadleaf_evergreen": (0, 2500, 15, 32),
            "broadleaf_deciduous": (0, 3000, 5, 28),
            "conifer_forest": (500, 4000, 0, 22),
            "mixed_forest": (0, 3000, 5, 28),
            "alpine_meadow": (2500, 5500, -5, 18),
            "desert_scrub": (-100, 3500, 5, 38),
            "grassland": (0, 4000, 0, 28),
            "bamboo_forest": (0, 2000, 12, 32),
            "cropland": (0, 2500, 5, 32),
            "sparse": (0, 6000, -10, 35),
        }

        # Build lookup: for each elevation band, which values are compatible?
        for elev_band in range(0, 6000, 100):
            e = elev_band + 50  # midpoint
            compatible_t = set()
            for t, (lo, hi) in _TERRAIN_ELEV.items():
                # Relaxed: ±30% range extension
                w = hi - lo
                if (lo - 0.30*w) <= e <= (hi + 0.30*w):
                    compatible_t.add(t)
            self._elev_to_terrains[e] = compatible_t

            compatible_v = set()
            for v, (lo, hi, _, _) in _VEGETATION_ELEV.items():
                w = hi - lo
                if (lo - 0.30*w) <= e <= (hi + 0.30*w):
                    compatible_v.add(v)
            self._elev_to_vegetations[e] = compatible_v

    def _sensor_filter(self, elevation_m, temperature_c, humidity_pct):
        """Relaxed sensor filter: expand region ranges by 30% to reduce false negatives.

        Returns regions compatible with sensor readings after range relaxation.
        The relaxation adds ±30% of range width to each bound, preventing the
        31% catastrophic failure rate caused by strict range boundaries.
        """
        compatible = []
        for region in self.regions:
            e_min, e_max = region.get("elevation_range", [-9999, 99999])
            t_min, t_max = region.get("temp_range", [-99, 99])
            h_min, h_max = region.get("humid_range", [0, 100])

            # Expand ranges by 30% of width to reduce false negatives
            e_pad = (e_max - e_min) * 0.30
            t_pad = (t_max - t_min) * 0.30
            h_pad = (h_max - h_min) * 0.30

            if elevation_m is not None:
                if elevation_m < (e_min - e_pad) or elevation_m > (e_max + e_pad):
                    continue
            if temperature_c is not None:
                if temperature_c < (t_min - t_pad) or temperature_c > (t_max + t_pad):
                    continue
            if humidity_pct is not None:
                if humidity_pct < (h_min - h_pad) or humidity_pct > (h_max + h_pad):
                    continue

            compatible.append(region)

        return compatible

    def _sensor_climate_filter(self, elevation_m, temperature_c, climate_pred):
        """Validate climate_zone prediction against sensor readings.

        Returns (is_compatible, reason).
        """
        if climate_pred is None:
            return True, ""

        allowed_by_elev = None
        allowed_by_temp = None

        if elevation_m is not None:
            for range_key, climates in self._elevation_climate.items():
                lo, hi = self._parse_range(range_key)
                if lo is not None and lo <= elevation_m <= hi:
                    allowed_by_elev = set(climates)
                    break

        if temperature_c is not None:
            for range_key, climates in self._temperature_climate.items():
                lo, hi = self._parse_range(range_key)
                if lo is not None and lo <= temperature_c <= hi:
                    allowed_by_temp = set(climates)
                    break

        # Combine: climate must be allowed by at least one sensor
        # If sensors disagree (rare), elevation is more reliable
        if allowed_by_elev is not None and climate_pred not in allowed_by_elev:
            if allowed_by_temp is not None and climate_pred in allowed_by_temp:
                return True, ""  # temperature allows it
            return False, (f"climate '{climate_pred}' impossible at "
                          f"{elevation_m:.0f}m elevation")

        if allowed_by_temp is not None and climate_pred not in allowed_by_temp:
            if allowed_by_elev is not None and climate_pred in allowed_by_elev:
                return True, ""  # elevation allows it
            return False, (f"climate '{climate_pred}' impossible at "
                          f"{temperature_c:.1f}°C")

        return True, ""

    @staticmethod
    def _parse_range(range_key: str) -> tuple:
        """Parse range key like '2500-4000m' or '<200m' or '>4000m'."""
        key = range_key.replace("m", "").replace("°C", "")
        if key.startswith("<"):
            return -float('inf'), float(key[1:])
        elif key.startswith(">"):
            return float(key[1:]), float('inf')
        elif "-" in key:
            parts = key.split("-")
            return float(parts[0]), float(parts[1])
        return None, None

    def _sensor_correct_fields(self, elements: dict,
                                sensor_elevation_m, sensor_temperature_c,
                                sensor_humidity_pct) -> dict:
        """Correct VLM field predictions that conflict with sensor data.

        Uses elevation sensor (100% accurate) to detect and fix impossible
        terrain, vegetation, and climate predictions. Based on the research
        finding that external verification >> VLM self-assessment.

        Returns a new elements dict with corrected values. Fields that are
        clearly wrong get their confidence reduced or value overridden.
        """
        if sensor_elevation_m is None:
            return elements

        corrected = dict(elements)
        elev_band = int(sensor_elevation_m // 100) * 100 + 50
        elev_band = max(50, min(5950, elev_band))

        compatible_terrains = self._elev_to_terrains.get(elev_band, set())
        compatible_vegetations = self._elev_to_vegetations.get(elev_band, set())

        corrections = []

        # ── Check terrain_type ──────────────────────────────────────
        terrain_val = elements.get("terrain_type")
        if terrain_val:
            if isinstance(terrain_val, dict):
                tv = terrain_val.get("value", "")
                conf = terrain_val.get("confidence", 0.5)
            else:
                tv = str(terrain_val)
                conf = 0.5

            if tv and tv != "UNKNOWN" and tv not in compatible_terrains:
                corrections.append(f"terrain {tv} impossible at {sensor_elevation_m:.0f}m → UNKNOWN")
                corrected["terrain_type"] = {"value": "UNKNOWN", "confidence": 0.0}

        # ── Check vegetation_zone ───────────────────────────────────
        veg_val = elements.get("vegetation_zone")
        if veg_val:
            if isinstance(veg_val, dict):
                vv = veg_val.get("value", "")
                conf = veg_val.get("confidence", 0.5)
            else:
                vv = str(veg_val)
                conf = 0.5

            if vv and vv != "UNKNOWN" and vv not in compatible_vegetations:
                corrections.append(f"vegetation {vv} impossible at {sensor_elevation_m:.0f}m → UNKNOWN")
                corrected["vegetation_zone"] = {"value": "UNKNOWN", "confidence": 0.0}

        # ── Check climate_zone via ontology elevation→climate ──────
        climate_val = elements.get("climate_zone")
        if climate_val and sensor_elevation_m is not None:
            if isinstance(climate_val, dict):
                cv = climate_val.get("value", "")
                conf = climate_val.get("confidence", 0.5)
            else:
                cv = str(climate_val)
                conf = 0.5

            if cv and cv != "UNKNOWN":
                compatible, reason = self._sensor_climate_filter(
                    sensor_elevation_m, sensor_temperature_c, cv)
                if not compatible:
                    corrections.append(f"climate {cv}: {reason} → UNKNOWN")
                    corrected["climate_zone"] = {"value": "UNKNOWN", "confidence": 0.0}

        # ── Check scene_type elevation consistency ──────────────────
        scene_val = elements.get("scene_type")
        if scene_val and sensor_elevation_m is not None:
            if isinstance(scene_val, dict):
                sv = scene_val.get("value", "")
                conf = scene_val.get("confidence", 0.5)
            else:
                sv = str(scene_val)
                conf = 0.5
            # Alpine scenes at low elevation → discount
            if sv in ("alpine", "glacier", "snowfield") and sensor_elevation_m < 2000:
                corrections.append(f"scene {sv} impossible at {sensor_elevation_m:.0f}m → UNKNOWN")
                corrected["scene_type"] = {"value": "UNKNOWN", "confidence": 0.0}
            if sv in ("beach", "coastal_cliff", "coral_reef") and sensor_elevation_m > 500:
                corrections.append(f"scene {sv} implausible at {sensor_elevation_m:.0f}m → UNKNOWN")
                corrected["scene_type"] = {"value": "UNKNOWN", "confidence": 0.0}

        # ── Elevation-adaptive weight hints ─────────────────────────
        # These are informational, stored for downstream use in grid search
        if sensor_elevation_m >= 2500:
            corrected["_elevation_hint"] = "alpine"
        elif sensor_elevation_m >= 800:
            corrected["_elevation_hint"] = "mid_elevation"
        else:
            corrected["_elevation_hint"] = "lowland"

        if corrections:
            corrected["_sensor_corrections"] = corrections

        return corrected

    def _check_constraints(self, elements: dict, region_id: str) -> list[str]:
        """Check if element predictions violate ontological constraints.

        Returns list of violation descriptions (empty = no violations).
        """
        violations = []
        field_values = {}

        # Collect field values from elements (unwrapping confidence-wrapped dicts)
        for field in HIERARCHY_WEIGHTS:
            val = elements.get(field)
            if val is None:
                continue
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            if val and val != "UNKNOWN":
                field_values[field] = val

        # Check each pair against impossible combinations
        fields_list = list(field_values.items())
        for i in range(len(fields_list)):
            for j in range(i + 1, len(fields_list)):
                f1, v1 = fields_list[i]
                f2, v2 = fields_list[j]
                key = (f1, v1, f2, v2)
                if key in self._impossible_pairs:
                    violations.append(
                        f"{f1}={v1} + {f2}={v2}: {self._impossible_pairs[key]}")

        return violations

    def _build_query_embedding(self, elements: dict) -> np.ndarray | None:
        """Build TransE query embedding from field predictions.

        Weighted average of field value embeddings by confidence.
        """
        if not self._kg_embeddings:
            return None

        field_value_emb = self._kg_embeddings.get("field_value_embeddings", {})
        vecs = []
        weights = []
        for field in HIERARCHY_WEIGHTS:
            val = elements.get(field)
            if isinstance(val, dict):
                conf = val.get("confidence", 0.5)
                val = val.get("value", "UNKNOWN")
            else:
                conf = 0.5
                val = str(val) if val else "UNKNOWN"

            if val == "UNKNOWN" or val not in field_value_emb:
                continue
            vecs.append(np.array(field_value_emb[val], dtype=np.float32) * conf)
            weights.append(conf)

        if not vecs:
            return None
        total_w = sum(weights)
        if total_w > 0:
            avg = np.average(vecs, axis=0, weights=weights)
        else:
            avg = np.mean(vecs, axis=0)
        # Normalize
        norm = np.linalg.norm(avg)
        if norm > 1e-8:
            avg /= norm
        return avg

    def _score_region(self, region: dict, elements: dict) -> RegionMatch:
        """Score a region against GeoVLM field predictions.

        Uses confidence-weighted fingerprint matching with IDF bonus
        and mismatch penalty. Normalizes by predicted fields for
        fair comparison across samples with different field coverage.

        Returns raw score (can be negative).
        """
        fingerprint = region.get("fingerprint", {})
        total_score = 0.0
        hits = 0
        total_fields = 0

        for field, h_weight in HIERARCHY_WEIGHTS.items():
            fp = fingerprint.get(field, {})
            if not fp:
                continue

            field_entry = elements.get(field)
            if isinstance(field_entry, dict):
                pred_val = field_entry.get("value", "UNKNOWN")
                confidence = field_entry.get("confidence", 0.5)
            else:
                pred_val = field_entry
                confidence = 0.5

            if pred_val is None or pred_val == "UNKNOWN":
                continue

            total_fields += 1

            fp_weight = fp.get(pred_val, 0.0)
            idf = self._field_value_idf.get((field, pred_val), 0.0)

            if fp_weight > 0:
                hits += 1
                # Match: weighted by hierarchy, fingerprint strength, confidence, IDF
                field_score = h_weight * fp_weight * confidence * (1.0 + idf)
                if field in ANCHOR_FIELDS and confidence > 0.6:
                    field_score *= ANCHOR_BOOST
                total_score += field_score
            else:
                # Mismatch penalty for discriminative values with decent confidence
                if idf > 0.3 and confidence > 0.3:
                    penalty = h_weight * confidence * idf * 0.15
                    total_score -= penalty

        # Normalize by fields actually predicted (not all possible fields)
        n_predicted = max(1, total_fields)
        normalized = total_score / n_predicted

        # Region information bonus: smaller regions provide more information
        radius = region.get("radius_km", 150)
        info_bonus = 0.15 * math.log(max(50, 300) / max(radius, 30))
        normalized = normalized * (1.0 + max(0.0, info_bonus))

        return RegionMatch(
            region_id=region["id"],
            region_name=region["name"],
            score=normalized,
            fingerprint_hits=hits,
            fingerprint_total=total_fields,
            center_lat=region["center_lat"],
            center_lng=region["center_lng"],
            radius_km=region.get("radius_km", 500),
        )

    def _sensor_affinity(self, region: dict, elevation_m, temperature_c,
                         humidity_pct) -> float:
        """Compute how well sensor readings match a region's typical ranges.

        Returns 0.0-1.0 where 1.0 = perfect fit, lower = poor fit.
        Uses smooth Gaussian falloff outside range instead of hard cutoff.
        """
        affinities = []

        for sensor_val, range_key in [
            (elevation_m, "elevation_range"),
            (temperature_c, "temp_range"),
            (humidity_pct, "humid_range"),
        ]:
            if sensor_val is None:
                continue
            r = region.get(range_key)
            if r is None or len(r) != 2:
                continue
            lo, hi = r
            range_width = hi - lo
            if range_width <= 0:
                continue

            if lo <= sensor_val <= hi:
                affinities.append(1.0)
            else:
                # How far outside the range (as fraction of range width)
                excess = min(abs(sensor_val - lo), abs(sensor_val - hi))
                tolerance = range_width * 0.5
                aff = math.exp(-0.5 * (excess / tolerance) ** 2)
                affinities.append(max(0.1, aff))  # floor at 0.1

        if not affinities:
            return 1.0  # no sensor data → neutral
        return sum(affinities) / len(affinities)

    def _mixture_peak(self, matches, probs, softmax_temp=0.15):
        """Find peak of Gaussian mixture model over top regions.

        Uses softmax probabilities as mixture weights and Gaussian kernels
        centered at each region with sigma proportional to region radius.

        Key safety: only interpolates between regions within 800 km of each
        other. When top regions are far apart, anchors to the best region's
        center rather than creating false peaks in between.

        Returns (lat, lng, uncertainty_km).
        """
        n = len(matches)
        if n == 0:
            return 35.0, 105.0, 3000.0
        if n == 1:
            return matches[0].center_lat, matches[0].center_lng, matches[0].radius_km

        # Convert scores to softmax probabilities
        scores = np.array([m.score for m in matches])
        scores_shifted = scores - np.max(scores)
        probs = np.exp(scores_shifted / softmax_temp)
        probs /= probs.sum()

        # Select regions with meaningful probability (>2%)
        sig_idx = np.where(probs > 0.02)[0]
        if len(sig_idx) < 2:
            sig_idx = np.argsort(probs)[-3:]
        if len(sig_idx) > 8:
            sig_idx = np.argsort(probs)[-8:]

        sig_matches = [matches[i] for i in sig_idx]
        sig_probs = probs[sig_idx]
        sig_probs /= sig_probs.sum()

        # Check if top regions are geographically close enough to interpolate
        top = sig_matches[0]
        max_interp_dist = 800.0  # km: max distance for meaningful interpolation

        # Find the subset of regions within max_interp_dist of the top region
        nearby = [(top, sig_probs[0])]
        for i, m in enumerate(sig_matches[1:], 1):
            d = _haversine_km(top.center_lat, top.center_lng,
                             m.center_lat, m.center_lng)
            if d < max_interp_dist:
                nearby.append((m, sig_probs[i]))

        if len(nearby) == 1:
            # Only one cluster: use top region's center (no interpolation needed)
            return top.center_lat, top.center_lng, top.radius_km * 0.9

        # Build Gaussian mixture and find peak via local grid search
        nearby_matches, nearby_probs = zip(*nearby)
        nearby_probs = np.array(nearby_probs) / sum(nearby_probs)

        # Search bounds from nearby regions (tight, no padding)
        lats = [m.center_lat for m in nearby_matches]
        lngs = [m.center_lng for m in nearby_matches]
        radii = [m.radius_km for m in nearby_matches]

        # Extend bounds by half the max radius
        max_radius = max(radii)
        pad_deg = (max_radius * 0.6) / 111.0
        lat_min = max(18.0, min(lats) - pad_deg)
        lat_max = min(54.0, max(lats) + pad_deg)
        lng_min = max(73.0, min(lngs) - pad_deg)
        lng_max = min(135.5, max(lngs) + pad_deg)

        # Resolution: 0.03 deg (~3.3 km) for tight clusters, 0.06 for broader
        spread = max(lat_max - lat_min, lng_max - lng_min)
        resolution = 0.03 if spread < 2.0 else (0.05 if spread < 5.0 else 0.08)

        # Gaussian kernels: sigma = radius * 0.4
        kernels = [(m.center_lat, m.center_lng, p, m.radius_km * 0.4)
                   for m, p in zip(nearby_matches, nearby_probs)]

        best_score = float('-inf')
        best_lat, best_lng = top.center_lat, top.center_lng

        lat = lat_min
        while lat <= lat_max:
            cos_lat = math.cos(math.radians(max(18.0, min(54.0, lat))))
            lng_step = resolution / max(cos_lat, 0.3)
            lng = lng_min
            while lng <= lng_max:
                score = 0.0
                for cen_lat, cen_lng, prob, sigma in kernels:
                    d = _haversine_km(lat, lng, cen_lat, cen_lng)
                    # Only contribute if within 2 sigma
                    if d < sigma * 3.0:
                        score += prob * math.exp(-0.5 * (d / sigma) ** 2)

                if score > best_score:
                    best_score = score
                    best_lat, best_lng = lat, lng

                lng += lng_step
            lat += resolution

        # Uncertainty: weighted avg distance from peak to nearby region centers
        peak_dists = [_haversine_km(best_lat, best_lng, m.center_lat, m.center_lng)
                     for m in nearby_matches]
        avg_dist = sum(d * p for d, p in zip(peak_dists, nearby_probs))
        uncertainty = max(50.0, avg_dist + 40.0)

        return best_lat, best_lng, min(3000.0, uncertainty)

    def _multi_hypothesis_predict(self, active, dem, sensor_elev):
        """Multi-hypothesis tracking with DEM-based re-weighting.

        When fingerprint scores are ambiguous (low margin), top regions may
        span diverse geographies. Instead of blending them into one prediction,
        this method:

        1. Clusters top regions geographically (greedy, 800km threshold)
        2. Computes Gaussian mixture prediction for each cluster
        3. Scores each hypothesis by DEM elevation consistency
        4. Returns the most DEM-consistent hypothesis

        Inspired by Environment-Assisted Particle Filter (EA-PF, 2024):
        environmental data re-weights particles to correct mistaken priors.
        """
        n = len(active)
        if n < 3:
            return None

        # ── Geographic clustering ────────────────────────────────────
        clusters = []  # list of (indices, center_lat, center_lng)
        used = set()

        for i in range(n):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, n):
                if j in used:
                    continue
                d = _haversine_km(active[i].center_lat, active[i].center_lng,
                                  active[j].center_lat, active[j].center_lng)
                if d < 800:
                    cluster.append(j)
                    used.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)

        if len(clusters) < 2:
            return None  # only one cluster → no ambiguity to resolve

        # ── Predict location for each cluster ────────────────────────
        hypotheses = []
        for cluster in clusters:
            cl_matches = [active[i] for i in cluster]
            cl_lat, cl_lng, cl_unc = self._mixture_peak(
                cl_matches, None, softmax_temp=0.15)

            # Score by DEM elevation consistency
            dem_score = 0.0
            if dem is not None and sensor_elev is not None:
                dem_elev = dem.query(cl_lat, cl_lng)
                if dem_elev is not None:
                    sigma = max(abs(sensor_elev) * 0.35, 200.0)
                    z = abs(dem_elev - sensor_elev) / sigma
                    dem_score = -z  # higher (less negative) = better
                else:
                    dem_score = -3.0  # no DEM data → penalize

            # Cluster quality score
            best_in_cluster = cl_matches[0]
            cl_score = best_in_cluster.score

            hypotheses.append({
                'lat': cl_lat, 'lng': cl_lng, 'unc': cl_unc,
                'dem_score': dem_score,
                'fp_score': cl_score,
                'size': len(cluster),
            })

        # ── Re-weight: combined score = fp_score + 0.5 * dem_score ──
        for h in hypotheses:
            h['combined'] = h['fp_score'] + 0.5 * h['dem_score']

        hypotheses.sort(key=lambda h: h['combined'], reverse=True)
        best = hypotheses[0]

        return best['lat'], best['lng'], best['unc'], hypotheses

    def _sensor_grid_refine(self, top_regions, sensor_elev, sensor_temp,
                           sensor_humid, dem, climate, mixture_lat, mixture_lng,
                           window_deg=2.5, w_fp=0.35):
        """Local grid refinement around Gaussian mixture prediction.

        Uses DEM + climate sensors for LOCAL refinement.
        The Gaussian mixture provides the coarse location, and DEM elevation
        provides fine-grained discrimination within that local area.

        Args:
            window_deg: half-width of search window in degrees
            w_fp: fingerprint prior weight (lower = more DEM-dependent)

        Returns (lat, lng, uncertainty_km).
        """
        n_regions = len(top_regions)
        if n_regions == 0:
            return mixture_lat, mixture_lng, 3000.0

        # ── Pre-compute fingerprint kernel params ────────────────────
        scores = np.array([r.score for r in top_regions])
        scores_shifted = scores - np.max(scores)
        probs = np.exp(scores_shifted / 0.15)
        probs /= probs.sum()

        fp_kernels = [(r.center_lat, r.center_lng, float(p), r.radius_km * 0.5)
                      for r, p in zip(top_regions, probs)]
        max_prob = float(probs.max())

        # ── Elevation-adaptive weight (stronger for alpine) ──────────
        if sensor_elev is not None:
            if sensor_elev < 200:
                w_elev = 0.15
            elif sensor_elev < 500:
                w_elev = 0.25
            elif sensor_elev < 2000:
                w_elev = 0.30
            else:
                w_elev = 0.50
        else:
            w_elev = 0.0
        w_temp = 0.10
        w_humid = 0.03

        # ── Local search window around mixture prediction ────────────
        lat_min = max(18.0, mixture_lat - window_deg)
        lat_max = min(54.0, mixture_lat + window_deg)
        lng_min = max(73.0, mixture_lng - window_deg)
        lng_max = min(135.5, mixture_lng + window_deg)

        resolution = 0.03

        best_score = float('-inf')
        best_lat = mixture_lat
        best_lng = mixture_lng

        lat = lat_min
        while lat <= lat_max:
            cos_lat = math.cos(math.radians(max(18.0, min(54.0, lat))))
            lng_step = resolution / max(cos_lat, 0.3)
            lng = lng_min
            while lng <= lng_max:
                score = 0.0

                # Fingerprint prior
                fp_density = 0.0
                for cen_lat, cen_lng, prob, sigma in fp_kernels:
                    d = _haversine_km(lat, lng, cen_lat, cen_lng)
                    if d < sigma * 3.0:
                        fp_density += prob * math.exp(-0.5 * (d / sigma) ** 2)
                fp_score = fp_density / max(max_prob, 0.001)
                score += w_fp * fp_score

                # Elevation
                if sensor_elev is not None and dem is not None:
                    dem_elev = dem.query(lat, lng)
                    if dem_elev is not None:
                        sigma_elev = max(abs(sensor_elev) * 0.35, 200.0)
                        diff = dem_elev - sensor_elev
                        score -= w_elev * (diff / sigma_elev) ** 2
                    else:
                        score -= w_elev * 4.0

                # Temperature
                if sensor_temp is not None and climate is not None:
                    dem_elev = dem.query(lat, lng) if dem else 500
                    elev_for_temp = dem_elev if dem_elev is not None else 500
                    t_lo, t_hi = climate.estimate_temperature_range(
                        lat, lng, elev_for_temp, month=7)
                    t_mid = (t_lo + t_hi) / 2
                    sigma_temp = max((t_hi - t_lo) * 0.5, 5.0)
                    diff = sensor_temp - t_mid
                    score -= w_temp * (diff / sigma_temp) ** 2

                # Humidity
                if sensor_humid is not None and climate is not None:
                    h_lo, h_hi = climate.estimate_humidity_range(lat, lng, month=7)
                    h_mid = (h_lo + h_hi) / 2
                    sigma_humid = max((h_hi - h_lo) * 0.5, 10.0)
                    diff = sensor_humid - h_mid
                    score -= w_humid * (diff / sigma_humid) ** 2

                if score > best_score:
                    best_score = score
                    best_lat, best_lng = lat, lng

                lng += lng_step
            lat += resolution

        # ── Uncertainty ───────────────────────────────────────────────
        dists = [_haversine_km(best_lat, best_lng, r.center_lat, r.center_lng)
                 for r in top_regions]
        weighted_dists = [d * p for d, p in zip(dists, probs)]
        uncertainty = sum(weighted_dists) * 0.6 + 100.0
        uncertainty = max(50.0, min(3000.0, uncertainty))

        return best_lat, best_lng, uncertainty

    def fuse(self,
             elements: dict,
             sensor_elevation_m: Optional[float] = None,
             sensor_temperature_c: Optional[float] = None,
             sensor_humidity_pct: Optional[float] = None,
             geocot_prediction: Optional[tuple[float, float]] = None,
             ) -> FusionResult:
        """Run ontology-based fusion.

        Algorithm:
          1. Sensor hard filter → compatible regions
          2. Fingerprint scoring → ranked region matches
          3. Constraint validation → penalty for impossible combinations
          4. Weighted coordinate prediction from top-K regions
          5. Uncertainty estimation from score distribution

        Args:
            elements: dict of field_name → value (or {"value": ..., "confidence": ...})
            sensor_elevation_m: sensor elevation in meters
            sensor_temperature_c: sensor temperature in Celsius
            sensor_humidity_pct: sensor humidity in percent
            geocot_prediction: (lat, lng) from GeoCoT (optional, used as tiebreaker)

        Returns:
            FusionResult with predicted coordinates, uncertainty, and confidence
        """
        ex = []

        # ── Step 1: Sensor hard filter ──────────────────────────────────
        if sensor_elevation_m is not None:
            ex.append(f"[SENSOR] Elevation: {sensor_elevation_m:.0f}m")
        if sensor_temperature_c is not None:
            ex.append(f"[SENSOR] Temperature: {sensor_temperature_c:.1f}°C")
        if sensor_humidity_pct is not None:
            ex.append(f"[SENSOR] Humidity: {sensor_humidity_pct:.0f}%")

        # ── Step 1: Relaxed sensor filter (geographic clustering) ────────
        candidates = self._sensor_filter(
            sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct)
        n_all = len(self.regions)

        if len(candidates) < 3:
            ex.append(f"[SENSOR-FILTER] Only {len(candidates)} regions → using all {n_all}")
            candidates = list(self.regions)
        else:
            ex.append(f"[SENSOR-FILTER] {len(candidates)}/{n_all} regions compatible "
                      f"(ranges relaxed ±30%)")

        # ── Step 1.3: Sensor-guided VLM prediction correction ──────────
        elements = self._sensor_correct_fields(
            elements, sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct)
        corrections = elements.pop("_sensor_corrections", [])
        elev_hint = elements.pop("_elevation_hint", None)
        if corrections:
            ex.append(f"[SENSOR-CORRECT] {len(corrections)} field(s) corrected:")
            for c in corrections:
                ex.append(f"  {c}")
        if elev_hint:
            ex.append(f"[SENSOR-HINT] Elevation regime: {elev_hint}")

        # Build KG query embedding once for all region comparisons
        query_emb = self._build_query_embedding(elements)
        if query_emb is not None and self._region_emb:
            ex.append("[KG] TransE re-rank enabled (blend=%.2f)" % KG_BLEND_WEIGHT)

        matches = []
        for region in candidates:
            match = self._score_region(region, elements)
            # Sensor affinity: soft geographic bias (0.3 + 0.7*s_aff)
            # Perfect fit: score unchanged. Poor fit: score × 0.37.
            # This biases toward sensor-compatible regions without hard-filtering.
            s_aff = self._sensor_affinity(
                region, sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct)
            match.score = match.score * (0.3 + 0.7 * s_aff)
            matches.append(match)

        matches.sort(key=lambda m: m.score, reverse=True)

        if not matches:
            ex.append("[FINGERPRINT] No regions → fallback to China center")
            return FusionResult(
                latitude=35.0, longitude=105.0,
                uncertainty_km=3000.0, confidence=0.05,
                explanation_parts=ex,
            )

        matches.sort(key=lambda m: m.score, reverse=True)

        best = matches[0]
        top_n = min(5, len(matches))
        top_str = ", ".join(
            f"#{i+1} {m.region_name}({m.score:.3f})" for i, m in enumerate(matches[:top_n]))
        ex.append(f"[FINGERPRINT] Top-{top_n}: {top_str}")

        # ── Step 3: Constraint validation ───────────────────────────────
        for match in matches[:top_n]:
            violations = self._check_constraints(elements, match.region_id)
            if violations:
                match.constraint_violations = violations
                penalty = 0.15 * len(violations)
                match.score -= penalty  # allow negative
                ex.append(f"[CONSTRAINT] {match.region_name}: {len(violations)} violation(s) "
                          f"→ score {match.score:.3f}")
                for v in violations[:3]:
                    ex.append(f"  {v}")

        matches.sort(key=lambda m: m.score, reverse=True)
        best = matches[0]

        # ── Step 3.5: KG re-rank of top candidates ──────────────────────
        if query_emb is not None and self._region_emb and len(matches) >= 2:
            re_rank_n = min(15, len(matches))
            kg_sims = []
            for m in matches[:re_rank_n]:
                if m.region_id in self._region_emb:
                    r_emb = self._region_emb[m.region_id]
                    sim = float(np.dot(query_emb, r_emb))
                    kg_sims.append(sim)
                else:
                    kg_sims.append(0.0)

            if kg_sims:
                mean_sim = sum(kg_sims) / len(kg_sims)
                for i, m in enumerate(matches[:re_rank_n]):
                    centered_sim = kg_sims[i] - mean_sim
                    m.score = m.score + KG_BLEND_WEIGHT * centered_sim
                    # No clipping — allow full range

                matches.sort(key=lambda m: m.score, reverse=True)
                best = matches[0]
                ex.append(f"[KG-RE-RANK] Top-{re_rank_n} re-ranked by embedding similarity")

        # ── Step 4: Sensor grid refinement (or Gaussian mixture fallback) ──
        positive = [m for m in matches if m.score > 0]
        if len(positive) >= 2:
            active = positive
        else:
            active = matches[:max(3, len(positive))]

        best = active[0]
        if len(active) >= 2:
            score_margin = (best.score - active[1].score) / max(abs(best.score), 0.01)
        else:
            score_margin = 1.0

        # ── Step 4: Gaussian mixture + optional local sensor refinement ──
        top_k = min(8, len(active))

        if len(active) == 1 or score_margin > 0.5:
            pred_lat = best.center_lat
            pred_lng = best.center_lng
            precision = min(1.0, best.fingerprint_hits / max(best.fingerprint_total, 1))
            uncertainty = best.radius_km * (1.0 - 0.5 * precision)
            ex.append(f"[COORDS] Strong match: {best.region_name} "
                      f"(score={best.score:.3f}, margin={score_margin:.2f})")
        else:
            # ── Multi-hypothesis tracking (low margin → ambiguous) ──────
            dem = _get_dem()
            multi_hyp = None
            if score_margin < 0.3 and dem is not None and sensor_elevation_m is not None:
                multi_result = self._multi_hypothesis_predict(
                    active[:min(10, len(active))], dem, sensor_elevation_m)
                if multi_result is not None:
                    mh_lat, mh_lng, mh_unc, hypotheses = multi_result
                    best_h = hypotheses[0]
                    # Check if multi-hypothesis DEM score is significantly better
                    # than original Gaussian mixture
                    mix_lat, mix_lng, _ = self._mixture_peak(
                        active[:top_k], None, softmax_temp=0.15)
                    dem_at_mix = dem.query(mix_lat, mix_lng)
                    mix_dem_z = 0.0
                    if dem_at_mix is not None and sensor_elevation_m is not None:
                        sigma = max(abs(sensor_elevation_m) * 0.35, 200.0)
                        mix_dem_z = abs(dem_at_mix - sensor_elevation_m) / sigma

                    if best_h['dem_score'] > -mix_dem_z + 0.5:  # MH improves DEM fit
                        pred_lat, pred_lng = mh_lat, mh_lng
                        uncertainty = mh_unc
                        multi_hyp = True
                        ex.append(f"[MULTI-HYP] {len(hypotheses)} clusters, "
                                  f"best cluster: {best_h['size']} regions, "
                                  f"DEM z={-best_h['dem_score']:.1f} vs mix z={mix_dem_z:.1f}")
                        # Skip local grid refinement for multi-hyp results
                        # (already DEM-verified)

            if multi_hyp is None:
                # Gaussian mixture for coarse prediction
                mix_lat, mix_lng, uncertainty = self._mixture_peak(
                    active[:top_k], None, softmax_temp=0.15)
                ex.append(f"[COORDS] Gaussian mixture: {top_k} regions "
                          f"(score={best.score:.3f}, margin={score_margin:.2f})")

            # Local sensor grid refinement around mixture prediction
            if multi_hyp is None:
                climate = _get_climate()
                has_sensors = (sensor_elevation_m is not None or
                              sensor_temperature_c is not None or
                              sensor_humidity_pct is not None)

                if dem is not None and has_sensors:
                    # Check if DEM at mixture prediction matches sensor elevation
                    dem_at_mix = dem.query(mix_lat, mix_lng)
                    dem_conflict = False
                    if dem_at_mix is not None and sensor_elevation_m is not None:
                        sigma_elev = max(abs(sensor_elevation_m) * 0.35, 200.0)
                        z_conflict = abs(dem_at_mix - sensor_elevation_m) / sigma_elev
                        dem_conflict = z_conflict > 2.0

                    if dem_conflict:
                        ex.append(f"[COORDS] DEM conflict at mixture "
                                  f"(DEM={dem_at_mix:.0f}m vs sensor={sensor_elevation_m:.0f}m, "
                                  f"z={z_conflict:.1f}) → expanding search to ±5°")
                        pred_lat, pred_lng, uncertainty = self._sensor_grid_refine(
                            active[:top_k],
                            sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                            dem, climate, mix_lat, mix_lng,
                            window_deg=5.0, w_fp=0.15)
                    else:
                        pred_lat, pred_lng, uncertainty = self._sensor_grid_refine(
                            active[:top_k],
                            sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct,
                            dem, climate, mix_lat, mix_lng)
                        ex.append(f"[COORDS] Sensor refine: ±2.5° local grid search "
                                  f"around ({mix_lat:.2f}, {mix_lng:.2f})")
                else:
                    pred_lat, pred_lng = mix_lat, mix_lng

        # ── Step 4.5: GeoCoT tiebreaker ─────────────────────────────────
        if geocot_prediction is not None and len(matches) >= 2 and score_margin < 0.25:
            glat, glng = geocot_prediction
            best_dist = float('inf')
            best_geo_region = None
            for m in matches[:5]:
                d = self._haversine(glat, glng, m.center_lat, m.center_lng)
                if d < best_dist:
                    best_dist = d
                    best_geo_region = m

            if best_geo_region and best_geo_region.region_id != best.region_id:
                ex.append(f"[GEOCO-TIEBREAK] GeoCoT ({glat:.2f},{glng:.2f}) favors "
                          f"{best_geo_region.region_name} (dist={best_dist:.0f}km)")
                alpha = 0.20
                pred_lat = pred_lat * (1 - alpha) + best_geo_region.center_lat * alpha
                pred_lng = pred_lng * (1 - alpha) + best_geo_region.center_lng * alpha
                uncertainty *= 0.90

        # ── Step 5: Confidence estimation ───────────────────────────────
        n_violations = sum(len(m.constraint_violations) for m in matches[:3])
        hit_ratio = best.fingerprint_hits / max(best.fingerprint_total, 1)

        # Sensor precision: fewer compatible regions = more precise
        sensor_precision = max(0.0, 1.0 - len(candidates) / max(n_all, 1))

        confidence = (
            0.10
            + 0.25 * max(0.0, best.score)        # best region score
            + 0.20 * min(1.0, score_margin)       # margin component
            + 0.20 * hit_ratio                     # evidence quality
            + 0.15 * sensor_precision              # sensor discrimination
            + 0.10 * max(0.0, 1.0 - len(active)/10)  # fewer active = more confident
            - 0.05 * n_violations                  # constraint penalty
        )
        confidence = max(0.05, min(0.95, confidence))

        uncertainty = max(50.0, min(3000.0, uncertainty))

        ex.append(f"[CONFIDENCE] score={best.score:.3f} margin={score_margin:.2f} "
                  f"hits={best.fingerprint_hits}/{best.fingerprint_total} "
                  f"active={len(active)} -> confidence={confidence:.3f}")

        # ── Step 6: Build result ────────────────────────────────────────
        candidate = CandidateRegion(
            center_lat=pred_lat,
            center_lng=pred_lng,
            radius_km=uncertainty,
            confidence=confidence,
            active_elements=[f"{m.region_name}({m.score:.3f})" for m in matches[:5]],
            dropped_elements=[],
            explanation="\n".join(ex),
        )

        return FusionResult(
            latitude=pred_lat,
            longitude=pred_lng,
            uncertainty_km=uncertainty,
            confidence=confidence,
            candidate_region=candidate,
            explanation_parts=ex,
        )

    @staticmethod
    def _haversine(lat1, lng1, lat2, lng2) -> float:
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlng / 2) ** 2)
        return 6371.0 * 2 * math.asin(math.sqrt(a))


# ── Convenience function (drop-in replacement for fuse_elements_v3) ──────

_ontology_fusion: Optional[OntologyFusion] = None


def ontology_fuse(elements: dict,
                  sensor_elevation_m: Optional[float] = None,
                  sensor_temperature_c: Optional[float] = None,
                  sensor_humidity_pct: Optional[float] = None,
                  geocot_prediction: Optional[tuple[float, float]] = None,
                  ) -> FusionResult:
    """Drop-in replacement for fuse_elements_v3() using geographic ontology.

    Same signature, same return type, radically different approach:
      - No GeoKB queries (pure in-memory fingerprint matching)
      - No hard bbox intersection (soft region scoring)
      - ~100x faster per sample
    """
    global _ontology_fusion
    if _ontology_fusion is None:
        _ontology_fusion = OntologyFusion()

    return _ontology_fusion.fuse(
        elements,
        sensor_elevation_m=sensor_elevation_m,
        sensor_temperature_c=sensor_temperature_c,
        sensor_humidity_pct=sensor_humidity_pct,
        geocot_prediction=geocot_prediction,
    )
