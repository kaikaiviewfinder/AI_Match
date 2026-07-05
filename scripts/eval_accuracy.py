"""Evaluate GeoVLM accuracy against teacher labels.

Metrics:
  - Per-field Top-1 accuracy (excluding UNKNOWN teacher labels)
  - Elevation MAE (mean absolute error)
  - Per-sample average accuracy
  - Confusion summary: where model disagrees with teacher
"""
import sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from geovlm import GeoVLM, GeoVLMConfig
from geovlm.vision_encoder import prepare_multi_scale_images

CHECKPOINT = "D:/Geocomp/output/geovlm_checkpoints/geovlm_final.pt"
LABELS_FILE = "D:/Geocomp/output/geovlm_teacher_labels.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fields to evaluate (teacher hard_labels keys → model FIELD_VOCABS keys)
EVAL_FIELDS = [
    ("climate_zone_pred", "climate_zone"),
    ("terrain_type_pred", "terrain_type"),
    ("vegetation_zone_pred", "vegetation_zone"),
    ("urbanization_pred", "urbanization"),
    ("architecture_style_pred", "architecture_style"),
    ("pavement_type_pred", "pavement_type"),
    ("language_script_pred", "language_script"),
    # New fields — skipped if teacher labels don't have them
    ("soil_color_pred", "soil_color"),
    ("sky_quality_pred", "sky_quality"),
    ("mountain_rock_type_pred", "mountain_rock_type"),
    ("tree_species_pred", "tree_species"),
    ("building_height_pred", "building_height"),
    ("water_type_pred", "water_type"),
    ("landform_detail_pred", "landform_detail"),
    ("scene_type_pred", "scene_type"),
]

FIELD_VOCABS = {
    "climate_zone": ["tropical", "subtropical", "temperate", "arid", "alpine", "boreal"],
    "terrain_type": ["urban_flat", "farmland_plain", "rolling_hills", "sharp_mountains",
                     "karst_peaks", "sandstone_pillars", "desert_dunes", "grassland_steppe", "plateau"],
    "vegetation_zone": ["tropical_rainforest", "broadleaf_evergreen", "broadleaf_deciduous",
                        "conifer_forest", "mixed_forest", "alpine_meadow", "desert_scrub",
                        "grassland", "bamboo_forest", "cropland", "sparse"],
    "urbanization": ["metropolis", "medium_city", "small_town", "village", "rural", "wilderness"],
    "architecture_style": ["modern_glass", "modern_residential", "old_residential", "hui_style",
                           "tibetan_stone", "courtyard", "arcade", "stilt_house", "tulou",
                           "shikumen", "traditional_official", "soviet_industrial", "none_visible"],
    "pavement_type": ["red_brick_tiles", "grey_concrete", "asphalt", "natural", "not_visible"],
    "language_script": ["simplified_chinese", "traditional_chinese", "tibetan", "uyghur_arabic",
                        "mongolian", "bilingual_cn_en", "none_visible"],
    "visible_text": ["local_government", "local_phone_code", "local_license_plate",
                     "local_street_sign", "cross_region_shop", "national_chain",
                     "national_slogan", "none_visible"],
    "soil_color": ["red", "yellow", "brown", "black", "grey", "loess_yellow", "white"],
    "sky_quality": ["clear_blue", "grey_hazy", "thick_fog", "dusty_yellow"],
    "mountain_rock_type": ["granite_spheroidal", "quartz_sandstone_pillars",
                           "limestone_karst", "red_sandstone_danxia",
                           "snow_peaks_glaciers", "alpine_lakes", "none_visible"],
    "tree_species": ["banyan", "camphor", "plane_tree", "gingko", "poplar",
                     "coconut_palm", "chinese_red_pine", "bamboo", "none_visible"],
    "building_height": ["super_tall", "high_rise", "mid_rise", "low_rise", "none_visible"],
    "water_type": ["ocean", "lake", "glacier", "waterfall", "river", "stream", "none_visible"],
    "landform_detail": ["limestone_tower_karst_scattered", "limestone_tower_karst_dense",
                        "granite_spheroidal_rounded", "granite_spheroidal_sharp",
                        "quartz_sandstone_pillars", "red_sandstone_danxia",
                        "snow_peaks_glaciers", "alpine_meadow", "plateau_plain",
                        "river_canyon", "granite_boulder_field", "coastal_shore",
                        "urban_flat", "none_visible"],
    "scene_type": ["desert", "grassland", "coastal", "river_valley", "city_street",
                   "mountain_trail_karst", "mountain_trail_granite", "mountain_trail_snow",
                   "rural_village", "tourist_scenic", "wilderness_forest", "none_visible"],
}

# Field display names (Chinese)
FIELD_CN = {
    "climate_zone": "气候带", "terrain_type": "地形", "vegetation_zone": "植被",
    "urbanization": "城市化", "architecture_style": "建筑风格", "pavement_type": "路面",
    "language_script": "语言文字", "visible_text": "可见文字",
    "soil_color": "土壤颜色", "sky_quality": "天空质量",
    "mountain_rock_type": "山体岩石", "tree_species": "树种",
    "building_height": "建筑高度", "water_type": "水体类型",
    "landform_detail": "地貌细节", "scene_type": "场景类型",
}


def main():
    print("=" * 60)
    print("GeoVLM Accuracy Evaluation")
    print("=" * 60)

    # Load model
    print(f"\n[1] Loading model ({DEVICE})...")
    config = GeoVLMConfig()
    model = GeoVLM(config)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = {k: v for k, v in state["model"].items()
                  if not k.startswith("heads.elevation_head")}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load labels
    print(f"\n[2] Loading teacher labels...")
    samples = []
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                path = rec.get("image_path", "")
                if path and Path(path).exists():
                    samples.append(rec)
    print(f"  Valid samples: {len(samples)}")

    # Evaluate
    print(f"\n[3] Running inference on {len(samples)} samples...\n")
    correct = defaultdict(int)
    total = defaultdict(int)
    elev_errors = []
    confusion = defaultdict(lambda: defaultdict(int))  # field → (pred → teacher)

    for rec in tqdm(samples, desc="Evaluating"):
        img = Image.open(rec["image_path"]).convert("RGB")
        scales = prepare_multi_scale_images(img)
        sensor = torch.tensor([[
            rec.get("sensor_elevation", 500.0),
            rec.get("sensor_temperature", 20.0),
            rec.get("sensor_humidity", 60.0),
        ]], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            output = model(scales, sensor)
            elements = output["elements"]

        for teacher_key, model_key in EVAL_FIELDS:
            teacher_val = rec["hard_labels"].get(teacher_key)
            if teacher_val is None or teacher_val == "UNKNOWN":
                continue
            total[model_key] += 1
            pred_val = elements.get(f"{model_key}_pred")
            if pred_val == teacher_val:
                correct[model_key] += 1
            else:
                confusion[model_key][(pred_val, teacher_val)] += 1

        # Elevation MAE
        teacher_elev = rec["hard_labels"].get("elevation_estimate_m")
        if teacher_elev and len(teacher_elev) == 2:
            pred_elev = elements.get("elevation_estimate_m")
            if pred_elev and len(pred_elev) == 2:
                teacher_mid = (teacher_elev[0] + teacher_elev[1]) / 2
                pred_mid = (pred_elev[0] + pred_elev[1]) / 2
                elev_errors.append(abs(teacher_mid - pred_mid))

    # Results
    print("\n" + "=" * 60)
    print("Per-Field Accuracy (Top-1, excluding UNKNOWN teacher labels)")
    print("=" * 60)
    print(f"{'Field':<25s} {'Acc':>8s} {'Correct':>8s} {'Total':>8s}")
    print("-" * 55)

    total_correct = 0
    total_samples = 0
    for teacher_key, model_key in EVAL_FIELDS:
        c = correct[model_key]
        t = total[model_key]
        acc = c / t * 100 if t > 0 else 0
        total_correct += c
        total_samples += t
        cn = FIELD_CN.get(model_key, model_key)
        print(f"{cn:<25s} {acc:7.1f}% {c:8d} {t:8d}")

    print("-" * 55)
    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"{'Overall':<25s} {overall:7.1f}% {total_correct:8d} {total_samples:8d}")

    if elev_errors:
        mae = np.mean(elev_errors)
        print(f"\n  Elevation MAE: {mae:.1f}m (on {len(elev_errors)} samples)")

    # Top confusion pairs
    print(f"\nTop Prediction → Teacher confusions (model predicted X, teacher said Y):")
    for model_key in EVAL_FIELDS:
        cm = confusion[model_key[1]]
        if cm:
            top = sorted(cm.items(), key=lambda x: -x[1])[:3]
            cn = FIELD_CN.get(model_key[1], model_key[1])
            for (pred, teacher), count in top:
                print(f"  {cn}: pred={pred} → teacher={teacher} ({count}x)")

    print(f"\nNote: 'visible_text' field has no teacher labels, skipped.")
    print(f"      'UNKNOWN' teacher labels (teacher disagreement) excluded from eval.")


if __name__ == "__main__":
    main()
