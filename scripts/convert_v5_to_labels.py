"""Convert selected_500_test_results_v5.json to teacher labels JSONL format.

v5 has single-pass 4B-Instruct GeoCoT results. Since there's no 3-seed ensemble,
we synthesize soft_labels as one-hot and mark all fields as "majority".
"""
import json
import ast
from pathlib import Path

V5_FILE = Path("D:/Geocomp/output/selected_500_test_results_v4_final.json")
OUTPUT = Path("D:/Geocomp/output/geovlm_teacher_labels.jsonl")

FIELD_MAP = {
    "climate_zone": "climate_zone_pred",
    "terrain_type": "terrain_type_pred",
    "vegetation_zone": "vegetation_zone_pred",
    "urbanization": "urbanization_pred",
    "architecture_style": "architecture_style_pred",
    "pavement_type": "pavement_type_pred",
    "language_script": "language_script_pred",
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
}


def parse_elevation(val):
    """Parse elevation from string or list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) == 2:
                return parsed
        except (ValueError, SyntaxError):
            pass
    if isinstance(val, (int, float)):
        return [val * 0.85, val * 1.15]
    return [400, 600]


def main():
    with open(V5_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from v5")

    converted = 0
    skipped = 0
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for entry in data:
            summary = entry.get("elements_summary", {})
            if not summary:
                skipped += 1
                continue

            # Build hard_labels
            hard_labels = {}
            for v5_field, label_field in FIELD_MAP.items():
                val = summary.get(v5_field)
                if val is None or val == "":
                    hard_labels[label_field] = "UNKNOWN"
                else:
                    hard_labels[label_field] = str(val)

            # Elevation
            elev_raw = summary.get("elevation_estimate_m", 500)
            hard_labels["elevation_estimate_m"] = parse_elevation(elev_raw)

            # Synthesize soft_labels (one-hot since single-pass)
            soft_labels = {}
            for label_field in FIELD_MAP.values():
                vocab = VOCABS.get(label_field, [])
                value = hard_labels.get(label_field, "UNKNOWN")
                dist = {}
                if value in vocab:
                    dist[value] = 1.0
                else:
                    # Uniform over vocab as fallback
                    for v in vocab:
                        dist[v] = 1.0 / len(vocab)
                soft_labels[label_field] = dist

            # Elevation soft label (same as hard)
            soft_labels["elevation_estimate_m"] = {"hard": 1.0}

            # Vote quality: all "majority" since single-pass
            vote_quality = {f: "majority" for f in FIELD_MAP.values()}
            vote_quality["elevation_estimate_m"] = "majority"

            # Difficulty score
            difficulty_score = entry.get("error_km", 500) / 500.0

            rec = {
                "image": entry.get("image", ""),
                "image_path": entry.get("image_path", ""),
                "true_lat": entry.get("true_lat"),
                "true_lng": entry.get("true_lng"),
                "sensor_elevation": entry.get("sensor_elevation_m", 500),
                "sensor_temperature": entry.get("sensor_temperature_c", 20),
                "sensor_humidity": entry.get("sensor_humidity_pct", 60),
                "hard_labels": hard_labels,
                "soft_labels": soft_labels,
                "vote_quality": vote_quality,
                "difficulty_score": difficulty_score,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted: {converted}, Skipped (no elements): {skipped}")
    print(f"Output: {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
