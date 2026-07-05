"""GeoVLM V1 full evaluation with detailed per-sample result saving.

Saves every prediction to JSONL for deep analysis.
"""
import sys, json, math, os, pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from geovlm import GeoVLM, GeoVLMConfig
from geovlm.vision_encoder import prepare_multi_scale_images
from regression.element_fusion import fuse_elements_v3

CHECKPOINT = "D:/Geocomp/output/geovlm_checkpoints/geovlm_final.pt"
LABELS_FILE = "D:/Geocomp/output/geovlm_teacher_labels.jsonl"
OUTPUT_FILE = "D:/Geocomp/output/geovlm_v1_results.jsonl"
ANALYSIS_FILE = "D:/Geocomp/output/geovlm_v1_analysis.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEOVLM_TO_FUSION = [
    "climate_zone", "terrain_type", "vegetation_zone",
    "urbanization", "architecture_style", "pavement_type",
    "language_script",
    "soil_color", "sky_quality", "mountain_rock_type",
    "tree_species", "building_height", "water_type",
    "landform_detail", "scene_type",
]

# Value categories for per-field analysis
FIELD_CATEGORY_ANALYSIS = {
    "climate_zone": ["tropical", "subtropical", "temperate", "arid", "alpine", "boreal"],
    "urbanization": ["metropolis", "medium_city", "small_town", "village", "rural", "wilderness"],
    "terrain_type": ["urban_flat", "farmland_plain", "rolling_hills", "sharp_mountains",
                     "karst_peaks", "sandstone_pillars", "desert_dunes", "grassland_steppe", "plateau"],
    "vegetation_zone": ["tropical_rainforest", "broadleaf_evergreen", "broadleaf_deciduous",
                        "conifer_forest", "mixed_forest", "alpine_meadow", "desert_scrub",
                        "grassland", "bamboo_forest", "cropland", "sparse"],
}


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def geovlm_elements_to_fusion(geovlm_elements, sensor):
    result = {}
    for key in GEOVLM_TO_FUSION:
        val = geovlm_elements.get(f"{key}_pred")
        if val is not None and val != "UNKNOWN":
            result[key] = val
    elev = geovlm_elements.get("elevation_estimate_m")
    if elev and len(elev) == 2:
        result["elevation_estimate_m"] = elev
    ruled = geovlm_elements.get("ruled_out_features")
    if ruled:
        result["ruled_out_features"] = ruled
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_FILE}")

    # Load model
    config = GeoVLMConfig()
    model = GeoVLM(config)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = {k: v for k, v in state["model"].items()
                  if not k.startswith("heads.elevation_head")}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # Load labels
    samples = []
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                path = rec.get("image_path", "")
                if path and Path(path).exists() and "true_lat" in rec and "true_lng" in rec:
                    samples.append(rec)

    if args.max_samples > 0 and args.max_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_samples)

    # Resume
    processed_indices = set()
    results = []
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                results.append(r)
                processed_indices.add(r["idx"])
        print(f"Resumed {len(results)} results")

    print(f"Running {len(samples)} samples...")

    for idx, rec in enumerate(tqdm(samples)):
        if idx in processed_indices:
            continue

        true_lat = rec["true_lat"]
        true_lng = rec["true_lng"]
        sensor_elev = rec.get("sensor_elevation", 500.0)
        sensor_temp = rec.get("sensor_temperature", 20.0)
        sensor_humid = rec.get("sensor_humidity", 60.0)

        result_row = {
            "idx": idx,
            "image_path": rec["image_path"],
            "true_lat": true_lat,
            "true_lng": true_lng,
            "sensor_elevation": sensor_elev,
        }

        # Step 1: GeoVLM inference
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            scales = prepare_multi_scale_images(img)
            sensor_tensor = torch.tensor([[
                sensor_elev, sensor_temp, sensor_humid,
            ]], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                output = model(scales, sensor_tensor)
                elements = output["elements"]
                raw_probs = output["raw"]["probs"]
                raw_confidence = output["raw"]["confidence"]

            # Save field predictions with confidence
            field_preds = {}
            for key in GEOVLM_TO_FUSION:
                pred_val = elements.get(f"{key}_pred", "UNKNOWN")
                conf = elements.get(f"{key}_confidence", 0.0)
                field_preds[key] = {"value": pred_val, "confidence": conf}
            result_row["field_predictions"] = field_preds
            result_row["ruled_out_features"] = elements.get("ruled_out_features", [])
            result_row["field_consistency"] = float(output.get("field_consistency", [0])[0].item())

        except Exception as e:
            result_row["error"] = f"GeoVLM: {e}"
            results.append(result_row)
            processed_indices.add(idx)
            continue

        # Step 2: Element fusion
        fusion_input = geovlm_elements_to_fusion(elements, {
            "elevation": sensor_elev,
            "temperature": sensor_temp,
            "humidity": sensor_humid,
        })

        try:
            fusion_result = fuse_elements_v3(
                fusion_input,
                sensor_elevation_m=sensor_elev,
                sensor_temperature_c=sensor_temp,
                sensor_humidity_pct=sensor_humid,
                geocot_prediction=None,
            )
        except Exception as e:
            result_row["error"] = f"Fusion: {e}"
            results.append(result_row)
            processed_indices.add(idx)
            continue

        if fusion_result.latitude is None or fusion_result.longitude is None:
            result_row["error"] = "Fusion returned None"
            results.append(result_row)
            processed_indices.add(idx)
            continue

        # Step 3: Compute error
        dist = haversine_km(true_lat, true_lng,
                            fusion_result.latitude, fusion_result.longitude)
        result_row["pred_lat"] = fusion_result.latitude
        result_row["pred_lng"] = fusion_result.longitude
        result_row["distance_km"] = round(dist, 3)
        result_row["uncertainty_km"] = round(fusion_result.uncertainty_km, 1)
        result_row["confidence"] = round(fusion_result.confidence, 3)
        result_row["calibrated"] = dist <= fusion_result.uncertainty_km + 1e-6

        results.append(result_row)
        processed_indices.add(idx)

        # Periodic save
        if len(processed_indices) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()

    # Final save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Analysis ──
    valid = [r for r in results if "distance_km" in r]
    dists = np.array([r["distance_km"] for r in valid])

    # Per-field-category analysis
    field_cat_errors = defaultdict(list)
    field_conf_errors = defaultdict(lambda: {"high": [], "low": []})
    for r in valid:
        fp = r.get("field_predictions", {})
        for field, categories in FIELD_CATEGORY_ANALYSIS.items():
            pred = fp.get(field, {})
            val = pred.get("value", "UNKNOWN")
            conf = pred.get("confidence", 0)
            if val in categories:
                field_cat_errors[f"{field}={val}"].append(r["distance_km"])
            if conf > 0.7:
                field_conf_errors[field]["high"].append(r["distance_km"])
            elif conf < 0.4:
                field_conf_errors[field]["low"].append(r["distance_km"])

    analysis = {
        "n_total": len(results),
        "n_valid": len(valid),
        "n_errors": sum(1 for r in results if "error" in r),
        "mean_km": float(np.mean(dists)),
        "median_km": float(np.median(dists)),
        "std_km": float(np.std(dists)),
        "min_km": float(np.min(dists)),
        "max_km": float(np.max(dists)),
        "percentiles": {
            f"P{p}": float(np.percentile(dists, p))
            for p in [5, 10, 25, 50, 75, 90, 95, 99]
        },
        "accuracy_at_km": {
            f"<={t}km": float(np.mean(dists <= t) * 100)
            for t in [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
        },
        "uncertainty": {
            "mean_km": float(np.mean([r["uncertainty_km"] for r in valid])),
            "median_km": float(np.median([r["uncertainty_km"] for r in valid])),
            "calibrated_pct": float(np.mean([r["calibrated"] for r in valid]) * 100),
        },
        "best_10": sorted(
            [{"idx": r["idx"], "km": r["distance_km"],
              "pred_lat": r["pred_lat"], "pred_lng": r["pred_lng"],
              "true_lat": r["true_lat"], "true_lng": r["true_lng"],
              "image": r["image_path"]}
             for r in valid], key=lambda x: x["km"]
        )[:10],
        "worst_10": sorted(
            [{"idx": r["idx"], "km": r["distance_km"],
              "pred_lat": r["pred_lat"], "pred_lng": r["pred_lng"],
              "true_lat": r["true_lat"], "true_lng": r["true_lng"],
              "image": r["image_path"]}
             for r in valid], key=lambda x: -x["km"]
        )[:10],
    }

    # Per-field confidence analysis
    field_conf_summary = {}
    for field in GEOVLM_TO_FUSION:
        high = field_conf_errors[field]["high"]
        low = field_conf_errors[field]["low"]
        field_conf_summary[field] = {
            "high_conf_mean_km": float(np.mean(high)) if high else None,
            "high_conf_n": len(high),
            "low_conf_mean_km": float(np.mean(low)) if low else None,
            "low_conf_n": len(low),
        }
    analysis["field_confidence_analysis"] = field_conf_summary

    # Top field-value categories
    top_cats = sorted(field_cat_errors.items(),
                      key=lambda x: len(x[1]), reverse=True)[:30]
    analysis["top_field_categories"] = [
        {"category": cat, "n": len(vals), "mean_km": float(np.mean(vals)),
         "median_km": float(np.median(vals))}
        for cat, vals in top_cats if len(vals) >= 10
    ]

    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"GeoVLM V1 Final Results")
    print(f"{'='*60}")
    print(f"  Valid: {len(valid)}/{len(results)}")
    print(f"  Mean:   {np.mean(dists):.0f} km")
    print(f"  Median: {np.median(dists):.0f} km")
    print(f"  Std:    {np.std(dists):.0f} km")
    print(f"  Min:    {np.min(dists):.1f} km")
    print(f"  Max:    {np.max(dists):.0f} km")
    print(f"\n  Accuracy:")
    for t in [10, 50, 100, 250, 500, 1000]:
        print(f"    ≤{t:4d} km: {np.mean(dists <= t)*100:5.1f}%")
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Analysis: {ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
