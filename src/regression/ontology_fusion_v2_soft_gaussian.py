"""Ontology-based geographic fusion engine.

Replaces GeoKB bbox intersection with region fingerprint matching.
Key advantages:
  - No external GeoKB queries → ~100x faster (microseconds vs seconds)
  - Robust to individual field errors (soft matching, not hard intersection)
  - Sensor data as hard constraints, not just scoring hints
  - Ontological consistency validation detects impossible combinations
  - Hierarchical matching: coarse climate → regional terrain → local details
  - KG embedding similarity for semantic region matching (V2)
"""

import json
import math
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

    def _sensor_filter(self, elevation_m, temperature_c, humidity_pct):
        """Hard filter: return regions compatible with sensor readings."""
        compatible = []
        for region in self.regions:
            e_min, e_max = region.get("elevation_range", [-9999, 99999])
            t_min, t_max = region.get("temp_range", [-99, 99])
            h_min, h_max = region.get("humid_range", [0, 100])

            if elevation_m is not None:
                if elevation_m < e_min or elevation_m > e_max:
                    continue
            if temperature_c is not None:
                if temperature_c < t_min or temperature_c > t_max:
                    continue
            if humidity_pct is not None:
                if humidity_pct < h_min or humidity_pct > h_max:
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

        Uses:
          - Confidence-weighted fingerprint matching with IDF bonus
          - Mismatch penalty: regions lacking a predicted value are penalized
          - Region information weighting: smaller regions get bonus

        Returns raw score (can be negative) — softmax in fuse() handles
        conversion to probabilities. No hard filtering at this stage.
        """
        fingerprint = region.get("fingerprint", {})
        total_score = 0.0
        max_score = 0.0
        hits = 0
        total_fields = 0

        for field, h_weight in HIERARCHY_WEIGHTS.items():
            fp = fingerprint.get(field, {})
            if not fp:
                continue

            max_score += h_weight

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
                field_score = h_weight * fp_weight * confidence * (1.0 + idf)
                if field in ANCHOR_FIELDS and confidence > 0.6:
                    field_score *= ANCHOR_BOOST
                total_score += field_score
            else:
                # Mismatch penalty: predicted value NOT in this region's fingerprint.
                # Only penalize for discriminative values (high IDF) with decent confidence.
                if idf > 0.5 and confidence > 0.4:
                    penalty = h_weight * confidence * idf * 0.10
                    total_score -= penalty

        # Normalize to [-2, 2] range — no hard clipping at zero.
        # Softmax in fuse() handles negative scores gracefully.
        normalized = total_score / max(max_score, 0.001)
        normalized = max(-2.0, min(2.0, normalized))

        # Region information bonus: smaller regions provide more information
        radius = region.get("radius_km", 150)
        info_bonus = 0.10 * math.log(max(50, 300) / max(radius, 30))
        normalized = normalized * (1.0 + max(0.0, info_bonus))
        normalized = max(-2.0, min(2.0, normalized))

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

        candidates = self._sensor_filter(
            sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct)

        n_all = len(self.regions)
        ex.append(f"[SENSOR-FILTER] {len(candidates)}/{n_all} regions compatible "
                  f"with sensor readings")

        # ── Step 1.5: Climate validation against sensor ─────────────────
        if sensor_elevation_m is not None or sensor_temperature_c is not None:
            climate_val = elements.get("climate_zone")
            if isinstance(climate_val, dict):
                climate_val = climate_val.get("value")
            compatible, reason = self._sensor_climate_filter(
                sensor_elevation_m, sensor_temperature_c, climate_val)
            if not compatible:
                ex.append(f"[CLIMATE-VETO] {reason} → down-weighting climate_zone")
                # Don't remove climate, just heavily discount in scoring
                if "climate_zone" in elements:
                    if isinstance(elements["climate_zone"], dict):
                        elements = dict(elements)
                        elements["climate_zone"] = {
                            "value": elements["climate_zone"]["value"],
                            "confidence": elements["climate_zone"].get("confidence", 0.5) * 0.3,
                        }

        # ── Step 2: Fingerprint scoring (all regions, no hard filter) ──
        if not candidates:
            ex.append("[FINGERPRINT] No sensor-compatible regions → scoring all regions")
            candidates = list(self.regions)

        # Build KG query embedding once for all region comparisons
        query_emb = self._build_query_embedding(elements)
        if query_emb is not None and self._region_emb:
            ex.append("[KG] TransE re-rank enabled (blend=%.2f)" % KG_BLEND_WEIGHT)

        matches = []
        for region in candidates:
            match = self._score_region(region, elements)
            # Sensor affinity: soft score modifier based on sensor fit
            s_aff = self._sensor_affinity(
                region, sensor_elevation_m, sensor_temperature_c, sensor_humidity_pct)
            if s_aff < 1.0:
                match.score = match.score * (0.9 + 0.1 * s_aff)
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

        # ── Step 4: Top-K selection + Gaussian mixture coordinate ──────
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

        if len(active) == 1 or score_margin > 0.5:
            pred_lat = best.center_lat
            pred_lng = best.center_lng
            precision = min(1.0, best.fingerprint_hits / max(best.fingerprint_total, 1))
            uncertainty = best.radius_km * (1.0 - 0.5 * precision)
            ex.append(f"[COORDS] Strong match: {best.region_name} "
                      f"(score={best.score:.3f}, margin={score_margin:.2f})")
        else:
            top_k = min(8, len(active))
            pred_lat, pred_lng, uncertainty = self._mixture_peak(
                active[:top_k], None, softmax_temp=0.15)
            ex.append(f"[COORDS] Gaussian mixture: {top_k} regions "
                      f"(score={best.score:.3f}, margin={score_margin:.2f})")

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
