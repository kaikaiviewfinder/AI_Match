"""Generate 3-stage GeoCoT training labels from GPS-tagged images.

Reverse-queries GeoKB + DEM + climate models at known coordinates to produce
the "correct" structured geographic elements that GeoCoT should have output.
Used as weak supervision for LoRA fine-tuning.

Usage:
    python scripts/generate_geocot_labels.py --dry-run --n 100   # preview quality
    python scripts/generate_geocot_labels.py --output data/vlm_finetune/geocot_3stage_train.jsonl
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.geo_kb import (
    BBox, CLIMATE_ZONES, TERRAIN_TYPES, VEGETATION_ZONES,
    TREE_SPECIES, ARCHITECTURE_STYLES, LANGUAGE_SCRIPTS,
    MOUNTAIN_ROCK_TYPES, SOIL_COLORS, SKY_QUALITY,
    INFRASTRUCTURE_TAGS, LANDFORM_DETAILS, WATER_TYPES,
    ROCK_COLORS, SCENE_TYPE_REGIONS, URBANIZATION_LEVELS,
    COMPOUND_SCENES, PROVINCE_BBOXES,
    get_bboxes_for_element,
)
from regression.dem_lookup import get_dem
from regression.climate_lookup import ClimateValidator
from regression.element_fusion import (
    parse_geocot_json, extract_all_elements, fuse_elements,
)


def _point_in_bbox(lat: float, lng: float, bbox: BBox) -> bool:
    """Check if a point falls inside a bounding box."""
    return (bbox.lat_min <= lat <= bbox.lat_max and
            bbox.lng_min <= lng <= bbox.lng_max)


def _reverse_lookup(lat: float, lng: float, category_dict: dict) -> list[str]:
    """Find all values in a GeoKB category whose bboxes contain (lat, lng)."""
    matches = []
    for value, bboxes in category_dict.items():
        for bbox in bboxes:
            if _point_in_bbox(lat, lng, bbox):
                matches.append(value)
                break  # one matching bbox is enough
    return matches


def _best_match(lat: float, lng: float, category_dict: dict, prefer_smallest: bool = True) -> str | None:
    """Pick the best-matching value for a category.

    When multiple bboxes contain the point, prefer the one with the smallest area
    (most specific). When prefer_smallest is False, take the first match.
    """
    candidates = []
    for value, bboxes in category_dict.items():
        for bbox in bboxes:
            if _point_in_bbox(lat, lng, bbox):
                candidates.append((bbox.area_km2, value))
                break
    if not candidates:
        return None
    if prefer_smallest:
        candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _bbox_contains_point(bbox: BBox, lat: float, lng: float) -> bool:
    return _point_in_bbox(lat, lng, bbox)


# China coastline waypoints (~2 degree spacing)
_COASTLINE = [
    (18.2, 109.5), (19.0, 110.5), (20.0, 110.5), (21.0, 111.0),
    (21.5, 111.5), (22.0, 113.0), (22.5, 114.0), (23.0, 114.5),
    (23.5, 117.0), (24.0, 118.0), (24.5, 118.5), (25.0, 119.0),
    (26.0, 119.5), (27.0, 120.5), (27.5, 121.0), (28.0, 121.5),
    (29.0, 121.5), (30.0, 122.0), (31.0, 121.8), (31.5, 122.0),
    (32.0, 121.5), (33.0, 121.0), (34.0, 120.5), (35.0, 120.0),
    (36.0, 120.0), (37.0, 121.0), (38.0, 121.5), (39.0, 122.0),
    (39.5, 121.5), (40.0, 122.0), (40.5, 121.5), (41.0, 121.5),
]


def _distance_to_coast_km(lat: float, lng: float) -> float:
    """Approximate distance to nearest coastline point in km."""
    min_dist = float("inf")
    for clat, clng in _COASTLINE:
        dlat = abs(lat - clat) * 111.0
        dlng = abs(lng - clng) * 102.0
        dist = (dlat**2 + dlng**2)**0.5
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _infer_scene_type(lat: float, lng: float, terrain_type: str | None,
                      urbanization: str, climate_zone: str | None) -> str:
    """Infer scene_type from terrain, urbanization, and coastal proximity."""
    # Urban scenes
    if urbanization in ("metropolis", "medium_city"):
        return "city_street"
    if urbanization in ("small_town", "village"):
        return "rural_village"

    # Coastal check
    if _distance_to_coast_km(lat, lng) < 50.0:
        return "coastal"

    # Terrain-based
    if terrain_type == "desert_dunes":
        return "desert"
    if terrain_type == "grassland_steppe":
        return "grassland"
    if terrain_type == "karst_peaks":
        return "mountain_trail_karst"
    if terrain_type == "sandstone_pillars":
        return "mountain_trail_karst"  # sandstone pillar = Zhangjiajie type
    if terrain_type == "sharp_mountains":
        if climate_zone == "alpine":
            return "mountain_trail_snow"
        return "mountain_trail_granite"
    if terrain_type == "plateau":
        if climate_zone == "arid":
            return "desert"
        return "grassland"

    # Defaults
    if urbanization == "rural":
        return "rural_village"
    if urbanization == "wilderness":
        if climate_zone == "arid":
            return "desert"
        if climate_zone == "alpine":
            return "mountain_trail_snow"
        return "wilderness_forest"

    return "wilderness_forest"


def generate_geocot_labels(
    geotagged_path: str = "data/merged_geotagged.json",
    output_path: str = "data/vlm_finetune/geocot_3stage_train.jsonl",
    dry_run: bool = False,
    n_samples: int | None = None,
    skip_self_check: bool = False,
):
    """Main entry point."""
    # ── Load geotagged data ──────────────────────────────────────────────
    with open(geotagged_path, encoding="utf-8") as f:
        geotagged = json.load(f)
    print(f"Loaded {len(geotagged)} geotagged images")

    # ── Load DEM + climate ──────────────────────────────────────────────
    dem = get_dem()
    climate = ClimateValidator()

    # ── Filter: point must have minimum GeoKB coverage ──────────────────
    # For auto-label quality, we need at least climate+terrain+vegetation
    # coverage and a province match.
    def _has_geokb_coverage(lat: float, lng: float) -> bool:
        """Check if GeoKB has adequate coverage for this point."""
        # Must be in at least one climate zone
        if not _reverse_lookup(lat, lng, CLIMATE_ZONES):
            return False
        # Must be in at least one province
        for bboxes in PROVINCE_BBOXES.values():
            for b in bboxes:
                if _point_in_bbox(lat, lng, b):
                    return True
        return False

    # ── Process each image ──────────────────────────────────────────────
    results = []
    skipped = {"no_geokb": 0, "self_check_fail": 0, "error": 0}
    items = list(geotagged.items())
    if n_samples:
        # Shuffle to get diverse samples
        import random
        random.seed(42)
        random.shuffle(items)
        items = items[:n_samples]

    for i, (img_path, coords) in enumerate(items):
        if isinstance(coords, list):
            lat, lng = coords[0], coords[1]
        else:
            lat, lng = coords.get("lat"), coords.get("lng")
            if lat is None:
                skipped["error"] += 1
                continue

        if not (17.0 <= lat <= 55.0 and 72.0 <= lng <= 136.0):
            skipped["no_geokb"] += 1
            continue

        # GeoKB coverage check
        if not _has_geokb_coverage(lat, lng):
            skipped["no_geokb"] += 1
            continue

        try:
            # ── Query DEM ────────────────────────────────────────────────
            elevation = None
            if dem is not None:
                elevation = dem.query(lat, lng)

            # ── Query climate zone ───────────────────────────────────────
            climate_zone = _best_match(lat, lng, CLIMATE_ZONES)
            terrain_type = _best_match(lat, lng, TERRAIN_TYPES)
            vegetation_zone = _best_match(lat, lng, VEGETATION_ZONES)

            # ── Urbanization ─────────────────────────────────────────────
            urbanization = "wilderness"  # default
            for level, bboxes in URBANIZATION_LEVELS.items():
                for b in bboxes:
                    if _point_in_bbox(lat, lng, b):
                        urbanization = level
                        break
                if urbanization != "wilderness":
                    break

            # ── REGIONAL elements ────────────────────────────────────────
            language_script = _best_match(lat, lng, LANGUAGE_SCRIPTS) or "simplified_chinese"
            architecture_style = _best_match(lat, lng, ARCHITECTURE_STYLES)
            tree_species = _reverse_lookup(lat, lng, TREE_SPECIES)[:2]
            mountain_rock_type = _best_match(lat, lng, MOUNTAIN_ROCK_TYPES)
            infrastructure_tags = _reverse_lookup(lat, lng, INFRASTRUCTURE_TAGS)[:2]
            sky_quality = _best_match(lat, lng, SKY_QUALITY)

            # ── Province ─────────────────────────────────────────────────
            likely_provinces = []
            for province, bboxes in PROVINCE_BBOXES.items():
                for b in bboxes:
                    if _point_in_bbox(lat, lng, b):
                        likely_provinces.append(province)
                        break
                if len(likely_provinces) >= 3:
                    break
            province_name = likely_provinces[0] if likely_provinces else "Unknown"

            # ── LOCAL elements ───────────────────────────────────────────
            # Scene type: use terrain + urbanization logic instead of
            # over-broad SCENE_TYPE_REGIONS bboxes
            scene_type = _infer_scene_type(
                lat, lng, terrain_type, urbanization, climate_zone)
            landform_detail = _best_match(lat, lng, LANDFORM_DETAILS)
            rock_color = _best_match(lat, lng, ROCK_COLORS)
            soil_color = _best_match(lat, lng, SOIL_COLORS)
            water_types = _reverse_lookup(lat, lng, WATER_TYPES)
            water_type = water_types[0] if water_types else None
            water_visible = water_type is not None

            # ── Compound scene match ─────────────────────────────────────
            matched_compound = None
            for name, config in COMPOUND_SCENES.items():
                anchor = config.get("anchor_bbox")
                if anchor and _point_in_bbox(lat, lng, anchor):
                    matched_compound = name
                    break

            # ── Distinctive features ─────────────────────────────────────
            distinctive_features = []
            if elevation and elevation > 3500:
                distinctive_features.append("high_altitude")
            if matched_compound:
                distinctive_features.append(matched_compound)
            if water_type:
                distinctive_features.append(f"{water_type}_nearby")
            if mountain_rock_type:
                distinctive_features.append(mountain_rock_type)

            # ── Elevation estimate ───────────────────────────────────────
            elev_min = max(0, (elevation or 500) * 0.7)
            elev_max = (elevation or 500) * 1.3

            # ── Build 3-stage JSON outputs ───────────────────────────────
            macro_output = json.dumps({
                "climate_zone": climate_zone or "temperate",
                "terrain_type": terrain_type or "rolling_hills",
                "vegetation_zone": vegetation_zone or "mixed_forest",
                "urbanization": urbanization,
                "elevation_estimate_m": [round(elev_min), round(elev_max)],
                "likely_continent": "Asia",
                "likely_region": f"{province_name} ({'wilderness' if urbanization == 'wilderness' else 'urban'})",
            }, ensure_ascii=False)

            regional_output = json.dumps({
                "language_script": language_script,
                "architecture_style": architecture_style or "none_visible",
                "tree_species": tree_species if tree_species else ["none_visible"],
                "mountain_rock_type": mountain_rock_type,
                "infrastructure_tags": infrastructure_tags if infrastructure_tags else ["none_visible"],
                "sky_quality": sky_quality or "clear_blue",
                "likely_country": "China",
                "likely_provinces": likely_provinces[:3] if likely_provinces else ["Unknown"],
            }, ensure_ascii=False)

            local_output = json.dumps({
                "scene_type": scene_type or "wilderness_forest",
                "landform_detail": landform_detail,
                "rock_color": rock_color,
                "soil_color": soil_color,
                "water_visible": water_visible,
                "water_type": water_type,
                "distinctive_features": distinctive_features[:4],
                "altitude_estimate_m": round(elevation) if elevation else 500,
                "confidence": "high",
                "location_name": matched_compound or province_name,
                "province": province_name,
                "latitude": lat,
                "longitude": lng,
            }, ensure_ascii=False)

            # ── Self-consistency check ────────────────────────────────────
            if not skip_self_check:
                parsed_macro = json.loads(macro_output)
                parsed_regional = json.loads(regional_output)
                parsed_local = json.loads(local_output)

                elements = extract_all_elements(parsed_macro, parsed_regional, parsed_local)
                if elements:
                    result = fuse_elements(elements, sensor_elevation_m=elevation)
                    error_km = _haversine(lat, lng, result.latitude, result.longitude)
                    if error_km > 200.0:
                        skipped["self_check_fail"] += 1
                        continue

            results.append({
                "image": img_path,
                "lat": lat,
                "lng": lng,
                "elevation": elevation,
                "compound_scene": matched_compound,
                "macro_output": macro_output,
                "regional_output": regional_output,
                "local_output": local_output,
                "province": province_name,
                "scene_type": scene_type or "unknown",
            })

        except Exception as e:
            skipped["error"] += 1
            if dry_run and skipped["error"] <= 5:
                print(f"  ERROR [{img_path}]: {e}")

        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(items)} ({len(results)} kept, "
                  f"{sum(skipped.values())} skipped)")

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} labeled examples")
    print(f"  Skipped: {skipped}")
    print(f"{'='*60}")

    # Scene type distribution
    scene_counts = {}
    for r in results:
        s = r.get("scene_type", "unknown")
        scene_counts[s] = scene_counts.get(s, 0) + 1
    print("  Scene distribution:")
    for s, c in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    if not dry_run and results:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved to {output_path}")

    return results


def _haversine(lat1, lng1, lat2, lng2):
    """Haversine distance in km."""
    import math
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-stage GeoCoT training labels from GPS-tagged images")
    parser.add_argument("--geotagged", default="data/merged_geotagged.json",
                        help="Path to merged_geotagged.json")
    parser.add_argument("--output", default="data/vlm_finetune/geocot_3stage_train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only, don't save")
    parser.add_argument("--n", type=int, default=None,
                        help="Max samples to process")
    parser.add_argument("--skip-self-check", action="store_true",
                        help="Skip fusion self-consistency check (faster)")
    args = parser.parse_args()

    generate_geocot_labels(
        geotagged_path=args.geotagged,
        output_path=args.output,
        dry_run=args.dry_run,
        n_samples=args.n,
        skip_self_check=args.skip_self_check,
    )


if __name__ == "__main__":
    main()
