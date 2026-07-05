"""Aggregate teacher labels from multiple historical test runs via cross-version voting.

Each version ran independently (different code/configs). Cross-version agreement
is a strong signal of label reliability - better than same-model 3-seed ensemble.

Expanded from 7 fields to 16 — includes spatial-narrowing fields.
"""
import json, ast
from pathlib import Path
from collections import Counter

VERSION_FILES = [
    "D:/Geocomp/output/selected_500_test_results_v2.json",
    "D:/Geocomp/output/selected_500_test_results_v3.json",
    "D:/Geocomp/output/selected_500_test_results_v4_final.json",
]
OUTPUT = Path("D:/Geocomp/output/geovlm_teacher_labels.jsonl")

# v5 elements_summary key → GeoVLM label key
FIELD_MAP = {
    "climate_zone":          "climate_zone_pred",
    "terrain_type":          "terrain_type_pred",
    "vegetation_zone":       "vegetation_zone_pred",
    "urbanization":          "urbanization_pred",
    "architecture_style":    "architecture_style_pred",
    "pavement_type":         "pavement_type_pred",
    "language_script":       "language_script_pred",
    # New fields
    "soil_color":            "soil_color_pred",
    "sky_quality":           "sky_quality_pred",
    "mountain_rock_type":    "mountain_rock_type_pred",
    "tree_species":          "tree_species_pred",
    "building_height":       "building_height_pred",
    "water_type":            "water_type_pred",
    "landform_detail":       "landform_detail_pred",
    "scene_type":            "scene_type_pred",
}

VOCABS = {
    "climate_zone_pred": ["tropical", "subtropical", "temperate", "arid", "alpine", "boreal"],
    "terrain_type_pred": ["urban_flat", "farmland_plain", "rolling_hills", "sharp_mountains",
                          "karst_peaks", "sandstone_pillars", "desert_dunes",
                          "grassland_steppe", "plateau"],
    "vegetation_zone_pred": ["tropical_rainforest", "broadleaf_evergreen",
                             "broadleaf_deciduous", "conifer_forest", "mixed_forest",
                             "alpine_meadow", "desert_scrub", "grassland",
                             "bamboo_forest", "cropland", "sparse"],
    "urbanization_pred": ["metropolis", "medium_city", "small_town",
                          "village", "rural", "wilderness"],
    "architecture_style_pred": ["modern_glass", "modern_residential", "old_residential",
                                "hui_style", "tibetan_stone", "courtyard", "arcade",
                                "stilt_house", "tulou", "shikumen", "traditional_official",
                                "soviet_industrial", "none_visible"],
    "pavement_type_pred": ["red_brick_tiles", "grey_concrete", "asphalt",
                           "natural", "not_visible"],
    "language_script_pred": ["simplified_chinese", "traditional_chinese", "tibetan",
                             "uyghur_arabic", "mongolian", "bilingual_cn_en", "none_visible"],
    "soil_color_pred": ["red", "yellow", "brown", "black", "grey", "loess_yellow", "white"],
    "sky_quality_pred": ["clear_blue", "grey_hazy", "thick_fog", "dusty_yellow"],
    "mountain_rock_type_pred": ["granite_spheroidal", "quartz_sandstone_pillars",
                                "limestone_karst", "red_sandstone_danxia",
                                "snow_peaks_glaciers", "alpine_lakes", "none_visible"],
    "tree_species_pred": ["banyan", "camphor", "plane_tree", "gingko", "poplar",
                          "coconut_palm", "chinese_red_pine", "bamboo", "none_visible"],
    "building_height_pred": ["super_tall", "high_rise", "mid_rise", "low_rise", "none_visible"],
    "water_type_pred": ["ocean", "lake", "glacier", "waterfall", "river", "stream", "none_visible"],
    "landform_detail_pred": ["limestone_tower_karst_scattered", "limestone_tower_karst_dense",
                             "granite_spheroidal_rounded", "granite_spheroidal_sharp",
                             "quartz_sandstone_pillars", "red_sandstone_danxia",
                             "snow_peaks_glaciers", "alpine_meadow", "plateau_plain",
                             "river_canyon", "granite_boulder_field", "coastal_shore",
                             "urban_flat", "none_visible"],
    "scene_type_pred": ["desert", "grassland", "coastal", "river_valley", "city_street",
                        "mountain_trail_karst", "mountain_trail_granite", "mountain_trail_snow",
                        "rural_village", "tourist_scenic", "wilderness_forest", "none_visible"],
}


def clean_value(val, field_key: str):
    """Normalize a value to match the field vocabulary."""
    if val is None or val == "" or val == "null":
        return None
    if isinstance(val, bool):
        return None

    # Handle lists (real Python lists)
    if isinstance(val, list):
        if len(val) == 0:
            return None
        val = val[0]

    # Handle string-formatted lists: "['plane_tree', 'poplar']"
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                val = parsed[0]
                val_str = str(val).strip()
        except (ValueError, SyntaxError):
            pass

    val_str = val_str.strip("'\" ").lower()

    if val_str in ("none", "null", "n/a", "unknown", "false", "true", ""):
        return None

    # Map known non-standard values to vocabulary terms
    MAPPINGS = {
        "tree_species_pred": {
            "willow": "none_visible",
            "huangshan_pine": "chinese_red_pine",
            "masson_pine": "chinese_red_pine",
            "oak": "none_visible",
            "fir": "none_visible",
            "spruce": "none_visible",
            "paulownia": "none_visible",
            "osmanthus": "none_visible",
            "cherry_blossom": "none_visible",
            "nanmu": "none_visible",
            "yew": "none_visible",
            "tamarisk": "none_visible",
            "euphrates_poplar": "poplar",
        },
        "soil_color_pred": {
            "green": "red",
        },
    }
    field_mappings = MAPPINGS.get(field_key, {})
    return field_mappings.get(val_str, val_str)


def parse_elevation(val):
    if isinstance(val, list) and len(val) == 2:
        return [float(val[0]), float(val[1])]
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) == 2:
                return [float(parsed[0]), float(parsed[1])]
        except (ValueError, SyntaxError):
            pass
    if isinstance(val, (int, float)):
        v = float(val)
        return [v * 0.85, v * 1.15]
    return [400.0, 600.0]


def main():
    all_data = []
    for path in VERSION_FILES:
        with open(path, encoding="utf-8") as f:
            all_data.append(json.load(f))
        print(f"Loaded {path}: {len(all_data[-1])} entries")

    N = len(all_data[0])
    n_versions = len(all_data)

    stats = {"unanimous": 0, "majority": 0, "split": 0, "missing_elements": 0}
    field_vote_stats = Counter()  # field → (unanimous/majority/split/missing)

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for i in range(N):
            entries = []
            for v_idx in range(n_versions):
                if i < len(all_data[v_idx]):
                    entries.append(all_data[v_idx][i])

            base = entries[0]
            img = base.get("image", "")
            img_path = base.get("image_path", "")

            summaries = []
            for e in entries:
                s = e.get("elements_summary", {})
                if s:
                    summaries.append(s)

            if len(summaries) < 2:
                stats["missing_elements"] += 1
                continue

            # Cross-version voting per field
            hard_labels = {}
            soft_labels = {}
            vote_quality = {}

            for v5_field, geo_field in FIELD_MAP.items():
                values = []
                for s in summaries:
                    v5_val = s.get(v5_field)
                    cleaned = clean_value(v5_val, geo_field)
                    if cleaned is not None:
                        # Check if value is in vocabulary
                        vocab = VOCABS.get(geo_field, [])
                        if cleaned in vocab or not vocab:
                            values.append(cleaned)

                if not values:
                    hard_labels[geo_field] = "UNKNOWN"
                    vote_quality[geo_field] = "split"
                    vocab = VOCABS.get(geo_field, [])
                    soft_labels[geo_field] = {v: 1.0/len(vocab) for v in vocab} if vocab else {}
                    field_vote_stats[(v5_field, "missing")] += 1
                    continue

                counter = Counter(values)
                most_common, count = counter.most_common(1)[0]
                total = len(values)

                hard_labels[geo_field] = most_common

                dist = {}
                for val, cnt in counter.items():
                    dist[val] = cnt / total
                soft_labels[geo_field] = dist

                if count == total:
                    vote_quality[geo_field] = "unanimous"
                    field_vote_stats[(v5_field, "unanimous")] += 1
                elif count >= total // 2 + 1:
                    vote_quality[geo_field] = "majority"
                    field_vote_stats[(v5_field, "majority")] += 1
                else:
                    vote_quality[geo_field] = "split"
                    field_vote_stats[(v5_field, "split")] += 1

            # Elevation: median across versions
            elev_list = []
            for e in entries:
                ev = e.get("vlm_elevation_estimate")
                if ev is None:
                    s = e.get("elements_summary", {})
                    ev = s.get("elevation_estimate_m")
                parsed = parse_elevation(ev)
                elev_list.append((parsed[0] + parsed[1]) / 2)

            if elev_list:
                elev_list.sort()
                med = elev_list[len(elev_list) // 2]
                hard_labels["elevation_estimate_m"] = [round(med * 0.9), round(med * 1.1)]
                vote_quality["elevation_estimate_m"] = "majority"
                soft_labels["elevation_estimate_m"] = {"hard": 1.0}

            n_unanimous = sum(1 for v in vote_quality.values() if v == "unanimous")
            n_split = sum(1 for v in vote_quality.values() if v == "split")
            if n_split > 4:
                stats["split"] += 1
            elif n_unanimous >= 10:
                stats["unanimous"] += 1
            else:
                stats["majority"] += 1

            rec = {
                "image": img,
                "image_path": img_path,
                "true_lat": base.get("true_lat"),
                "true_lng": base.get("true_lng"),
                "sensor_elevation": base.get("sensor_elevation_m", 500),
                "sensor_temperature": base.get("sensor_temperature_c", 20),
                "sensor_humidity": base.get("sensor_humidity_pct", 60),
                "hard_labels": hard_labels,
                "soft_labels": soft_labels,
                "vote_quality": vote_quality,
                "n_versions": len(summaries),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nConverted {N - stats['missing_elements']}/{N} entries")
    print(f"  Unanimous (>=10 fields agree): {stats['unanimous']}")
    print(f"  Majority: {stats['majority']}")
    print(f"  Split (>4 fields disagree): {stats['split']}")
    print(f"\nPer-field vote quality:")
    for v5_field in FIELD_MAP:
        total = sum(field_vote_stats[(v5_field, q)] for q in ["unanimous", "majority", "split", "missing"])
        uni = field_vote_stats[(v5_field, "unanimous")]
        maj = field_vote_stats[(v5_field, "majority")]
        spl = field_vote_stats[(v5_field, "split")]
        mis = field_vote_stats[(v5_field, "missing")]
        if total > 0:
            print(f"  {v5_field:<25s}: uni={uni:3d} maj={maj:3d} split={spl:3d} miss={mis:3d}  "
                  f"coverage={(uni+maj)/total*100:.0f}%")
    print(f"\nOutput: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
