"""Compare ontology_fusion vs fuse_elements_v3 on GeoVLM V1 field predictions.

Re-runs ontology_fusion on the same GeoVLM field predictions from V1 eval,
then compares accuracy, speed, and behavior against fuse_elements_v3 results.
"""
import sys, json, math, time, os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from regression.ontology_fusion import ontology_fuse
from regression.element_fusion import fuse_elements_v3

V1_RESULTS = "D:/Geocomp/output/geovlm_v1_results.jsonl"
OUTPUT_COMPARISON = "D:/Geocomp/output/ontology_vs_geokb_comparison.json"
OUTPUT_DETAIL = "D:/Geocomp/output/ontology_vs_geokb_detail.jsonl"

GEOVLM_TO_FUSION = [
    "climate_zone", "terrain_type", "vegetation_zone",
    "urbanization", "architecture_style", "pavement_type",
    "language_script",
    "soil_color", "sky_quality", "mountain_rock_type",
    "tree_species", "building_height", "water_type",
    "landform_detail", "scene_type",
]


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def field_predictions_to_elements(fp: dict) -> dict:
    """Convert V1 field_predictions to ontology_fusion input format."""
    elements = {}
    for field, pred in fp.items():
        if isinstance(pred, dict):
            elements[field] = pred
        else:
            elements[field] = {"value": str(pred), "confidence": 0.5}
    return elements


def field_predictions_to_flat(fp: dict) -> dict:
    """Convert V1 field_predictions to flat dict for fuse_elements_v3."""
    result = {}
    for field, pred in fp.items():
        if isinstance(pred, dict):
            val = pred.get("value", "UNKNOWN")
        else:
            val = str(pred)
        if val and val != "UNKNOWN":
            result[field] = val
    return result


def main():
    # Load V1 results
    results = []
    with open(V1_RESULTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    valid_v1 = [r for r in results if "distance_km" in r and "field_predictions" in r]
    print(f"Loaded {len(results)} V1 results, {len(valid_v1)} valid with coordinates")

    if len(valid_v1) < 10:
        print("Not enough valid results for comparison. Need at least 10.")
        return

    # Run ontology_fusion on each sample
    comparison_results = []
    onto_dists = []
    geokb_dists = []
    onto_times = []
    onto_confidences = []
    geokb_confidences = []
    onto_uncertainties = []
    geokb_uncertainties = []

    print(f"\nRunning ontology_fusion on {len(valid_v1)} samples...")
    for r in valid_v1:
        true_lat = r["true_lat"]
        true_lng = r["true_lng"]
        sensor_elev = r.get("sensor_elevation", 500.0)
        fp = r.get("field_predictions", {})

        # Ontology fusion (with confidence from field_predictions)
        onto_elements = field_predictions_to_elements(fp)
        t0 = time.perf_counter()
        try:
            onto_result = ontology_fuse(
                onto_elements,
                sensor_elevation_m=sensor_elev,
                sensor_temperature_c=None,
                sensor_humidity_pct=None,
            )
            onto_time = time.perf_counter() - t0
            onto_dist = haversine_km(true_lat, true_lng,
                                     onto_result.latitude, onto_result.longitude)
        except Exception as e:
            print(f"  [{r['idx']}] ontology_fusion error: {e}")
            continue

        # GeoKB result from V1 (already computed)
        geokb_dist = r["distance_km"]
        geokb_conf = r.get("confidence", 0)
        geokb_uncertainty = r.get("uncertainty_km", 0)

        onto_dists.append(onto_dist)
        geokb_dists.append(geokb_dist)
        onto_times.append(onto_time)
        onto_confidences.append(onto_result.confidence)
        geokb_confidences.append(geokb_conf)
        onto_uncertainties.append(onto_result.uncertainty_km)
        geokb_uncertainties.append(geokb_uncertainty)

        comparison_results.append({
            "idx": r["idx"],
            "image_path": r["image_path"],
            "true_lat": true_lat,
            "true_lng": true_lng,
            "sensor_elevation": sensor_elev,
            "geokb_pred_lat": r["pred_lat"],
            "geokb_pred_lng": r["pred_lng"],
            "geokb_distance_km": geokb_dist,
            "geokb_uncertainty_km": geokb_uncertainty,
            "geokb_confidence": geokb_conf,
            "onto_pred_lat": onto_result.latitude,
            "onto_pred_lng": onto_result.longitude,
            "onto_distance_km": round(onto_dist, 3),
            "onto_uncertainty_km": round(onto_result.uncertainty_km, 1),
            "onto_confidence": round(onto_result.confidence, 3),
            "onto_time_ms": round(onto_time * 1000, 1),
            "onto_top_regions": onto_result.explanation_parts[3] if len(onto_result.explanation_parts) > 3 else "",
        })

    if not onto_dists:
        print("No valid ontology fusion results!")
        return

    onto_dists = np.array(onto_dists)
    geokb_dists = np.array(geokb_dists)

    # ── Comparison statistics ──────────────────────────────────────────
    n_better = np.sum(onto_dists < geokb_dists)
    n_worse = np.sum(onto_dists > geokb_dists)
    n_tie = np.sum(np.abs(onto_dists - geokb_dists) < 0.001)

    improvement = geokb_dists - onto_dists  # positive = ontology better
    mean_improvement = np.mean(improvement)
    median_improvement = np.median(improvement)

    comparison = {
        "n_samples": len(onto_dists),
        "ontology_fusion": {
            "mean_km": float(np.mean(onto_dists)),
            "median_km": float(np.median(onto_dists)),
            "std_km": float(np.std(onto_dists)),
            "min_km": float(np.min(onto_dists)),
            "max_km": float(np.max(onto_dists)),
            "accuracy_at_km": {
                f"<={t}km": float(np.mean(onto_dists <= t) * 100)
                for t in [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
            },
            "percentiles": {
                f"P{p}": float(np.percentile(onto_dists, p))
                for p in [5, 10, 25, 50, 75, 90, 95, 99]
            },
            "mean_confidence": float(np.mean(onto_confidences)),
            "mean_uncertainty_km": float(np.mean(onto_uncertainties)),
            "mean_time_ms": float(np.mean(onto_times) * 1000),
            "calibrated_pct": float(np.mean(
                onto_dists <= np.array(onto_uncertainties) + 1e-6
            ) * 100),
        },
        "geokb_fusion": {
            "mean_km": float(np.mean(geokb_dists)),
            "median_km": float(np.median(geokb_dists)),
            "std_km": float(np.std(geokb_dists)),
            "min_km": float(np.min(geokb_dists)),
            "max_km": float(np.max(geokb_dists)),
            "accuracy_at_km": {
                f"<={t}km": float(np.mean(geokb_dists <= t) * 100)
                for t in [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
            },
            "percentiles": {
                f"P{p}": float(np.percentile(geokb_dists, p))
                for p in [5, 10, 25, 50, 75, 90, 95, 99]
            },
            "mean_confidence": float(np.mean(geokb_confidences)),
            "mean_uncertainty_km": float(np.mean(geokb_uncertainties)),
            "calibrated_pct": float(np.mean(
                geokb_dists <= np.array(geokb_uncertainties) + 1e-6
            ) * 100),
        },
        "head_to_head": {
            "ontology_wins": int(n_better),
            "geokb_wins": int(n_worse),
            "ties": int(n_tie),
            "ontology_win_pct": float(n_better / len(onto_dists) * 100),
            "mean_improvement_km": float(mean_improvement),
            "median_improvement_km": float(median_improvement),
        },
        "time_comparison": {
            "ontology_mean_ms": float(np.mean(onto_times) * 1000),
            "ontology_total_ms": float(np.sum(onto_times) * 1000),
            "geokb_estimated_total_s": "5-26s per sample (external GeoKB queries)",
        },
        "best_10_ontology": sorted(
            [{"idx": r["idx"], "km": r["onto_distance_km"],
              "geokb_km": r["geokb_distance_km"],
              "pred_lat": r["onto_pred_lat"], "pred_lng": r["onto_pred_lng"],
              "image": r["image_path"]}
             for r in comparison_results], key=lambda x: x["km"]
        )[:10],
        "worst_10_ontology": sorted(
            [{"idx": r["idx"], "km": r["onto_distance_km"],
              "geokb_km": r["geokb_distance_km"],
              "pred_lat": r["onto_pred_lat"], "pred_lng": r["onto_pred_lng"],
              "image": r["image_path"]}
             for r in comparison_results], key=lambda x: -x["km"]
        )[:10],
    }

    # ── By improvement magnitude ───────────────────────────────────────
    big_win = improvement > 500   # ontology >500km better
    big_loss = improvement < -500  # ontology >500km worse
    comparison["improvement_breakdown"] = {
        "ontology_much_better_500km": int(np.sum(big_win)),
        "ontology_much_worse_500km": int(np.sum(big_loss)),
        "ontology_better": int(np.sum(improvement > 0)),
        "ontology_worse": int(np.sum(improvement < 0)),
    }

    # ── Save ──────────────────────────────────────────────────────────
    with open(OUTPUT_COMPARISON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DETAIL, "w", encoding="utf-8") as f:
        for r in comparison_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Ontology Fusion vs GeoKB Fusion Comparison ({len(onto_dists)} samples)")
    print(f"{'='*65}")
    print(f"{'Metric':<25} {'Ontology':>12} {'GeoKB':>12} {'Delta':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for metric in ["mean_km", "median_km", "std_km"]:
        o_val = comparison["ontology_fusion"][metric]
        g_val = comparison["geokb_fusion"][metric]
        delta = o_val - g_val
        sign = "+" if delta > 0 else ""
        print(f"{metric:<25} {o_val:>11.1f}  {g_val:>11.1f}  {sign}{delta:>11.1f}")

    print(f"\n  Accuracy @ thresholds:")
    for t in [10, 50, 100, 250, 500, 1000]:
        o = comparison["ontology_fusion"]["accuracy_at_km"][f"<={t}km"]
        g = comparison["geokb_fusion"]["accuracy_at_km"][f"<={t}km"]
        print(f"    ≤{t:4d} km:  Ontology {o:5.1f}%  |  GeoKB {g:5.1f}%")

    print(f"\n  Head-to-head:")
    print(f"    Ontology wins:     {n_better}/{len(onto_dists)} ({100*n_better/len(onto_dists):.1f}%)")
    print(f"    GeoKB wins:        {n_worse}/{len(onto_dists)} ({100*n_worse/len(onto_dists):.1f}%)")
    print(f"    Mean improvement:  {mean_improvement:+.1f} km")
    print(f"    Median improvement:{median_improvement:+.1f} km")

    print(f"\n  Speed:")
    print(f"    Ontology mean: {np.mean(onto_times)*1000:.2f} ms/sample")
    print(f"    GeoKB mean:    5000-26000 ms/sample (external queries)")

    print(f"\n  Saved: {OUTPUT_COMPARISON}")
    print(f"  Detail: {OUTPUT_DETAIL}")


if __name__ == "__main__":
    main()
