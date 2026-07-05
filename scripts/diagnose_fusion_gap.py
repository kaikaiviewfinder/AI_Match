"""Diagnose whether accuracy gap is from GeoVLM predictions or ontology fusion.

Three analyses:
  1. True-region hit rate: how often does the correct region rank in top-K?
  2. Field-level prediction quality: which fields are reliable?
  3. Oracle ceiling: what accuracy would perfect field predictions achieve?
"""
import sys, json, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from regression.ontology_fusion import OntologyFusion, HIERARCHY_WEIGHTS, ANCHOR_FIELDS

V1_RESULTS = "D:/Geocomp/output/geovlm_v1_results.jsonl"


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def field_predictions_to_elements(fp: dict) -> dict:
    elements = {}
    for field, pred in fp.items():
        if isinstance(pred, dict):
            elements[field] = pred
        else:
            elements[field] = {"value": str(pred), "confidence": 0.5}
    return elements


def find_true_region(lat, lng, fusion: OntologyFusion):
    """Find which region the true coordinates fall into (nearest center)."""
    best_rid, best_name, best_dist = None, None, float("inf")
    for region in fusion.regions:
        d = haversine_km(lat, lng, region["center_lat"], region["center_lng"])
        if d < best_dist:
            best_dist = d
            best_rid = region["id"]
            best_name = region["name"]
    return best_rid, best_name, best_dist


def build_oracle_elements(region: dict) -> dict:
    """Build 'perfect' elements from a region's fingerprint (top value per field)."""
    elements = {}
    fp = region.get("fingerprint", {})
    for field in HIERARCHY_WEIGHTS:
        field_dist = fp.get(field, {})
        if field_dist:
            top_val = max(field_dist, key=field_dist.get)
            elements[field] = {"value": top_val, "confidence": 0.9}
    return elements


def main():
    fusion = OntologyFusion()

    # Load V1 results
    results = []
    with open(V1_RESULTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    valid = [r for r in results if "distance_km" in r and "field_predictions" in r]
    print(f"Loaded {len(valid)} samples")

    # ═══════════════════════════════════════════════════════════════════
    # Analysis 1: True-region hit rate
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ANALYSIS 1: True-Region Hit Rate")
    print("=" * 60)

    true_region_ranks = defaultdict(int)  # rank → count
    true_region_distances = []
    oracle_dists = []
    vlm_dists = []
    field_hit_rates = defaultdict(lambda: {"hits": 0, "total": 0})

    for r in valid:
        true_lat, true_lng = r["true_lat"], r["true_lng"]
        fp = r.get("field_predictions", {})

        # Find true region
        true_rid, true_name, true_dist = find_true_region(true_lat, true_lng, fusion)

        # Get VLM elements
        vlm_elements = field_predictions_to_elements(fp)

        # Run ontology fusion with VLM predictions
        from regression.ontology_fusion import ontology_fuse
        vlm_result = ontology_fuse(vlm_elements, sensor_elevation_m=r.get("sensor_elevation", 500))
        vlm_dist = haversine_km(true_lat, true_lng, vlm_result.latitude, vlm_result.longitude)
        vlm_dists.append(vlm_dist)

        # Find true region's rank in matches
        # Re-run scoring manually to get all region scores
        scores = []
        for region in fusion.regions:
            score = fusion._score_region(region, vlm_elements).score
            scores.append((region["id"], region["name"], score))
        scores.sort(key=lambda x: -x[2])

        # Find rank of true region
        true_rank = None
        for i, (rid, name, score) in enumerate(scores):
            if rid == true_rid:
                true_rank = i + 1
                break

        if true_rank is not None:
            true_region_ranks[min(true_rank, 16)] += 1  # bucket 16+ as "very low"
        true_region_distances.append(true_dist)

        # Oracle test: what if VLM gave perfect top-1 field values?
        oracle_elements = build_oracle_elements(
            fusion._region_by_id.get(true_rid, fusion.regions[0]))
        oracle_result = ontology_fuse(oracle_elements, sensor_elevation_m=r.get("sensor_elevation", 500))
        oracle_dist = haversine_km(true_lat, true_lng, oracle_result.latitude, oracle_result.longitude)
        oracle_dists.append(oracle_dist)

        # Per-field hit rate
        for field in HIERARCHY_WEIGHTS:
            region_fp = fusion._region_by_id.get(true_rid, {}).get("fingerprint", {}).get(field, {})
            if not region_fp:
                continue
            vlm_val = vlm_elements.get(field, {})
            if isinstance(vlm_val, dict):
                vlm_val = vlm_val.get("value", "UNKNOWN")
            field_hit_rates[field]["total"] += 1
            if vlm_val in region_fp:
                field_hit_rates[field]["hits"] += 1

    # Print hit rate results
    print(f"\n  True region rank distribution (out of {len(valid)} samples):")
    for rank in [1, 2, 3, 5, 10, 16]:
        count = true_region_ranks.get(rank, 0)
        cumulative = sum(true_region_ranks[k] for k in true_region_ranks if k <= rank)
        print(f"    Top-{rank:>2}: {cumulative:>4} ({100*cumulative/len(valid):.1f}%)")
    print(f"    True region mean distance from sample: {np.mean(true_region_distances):.0f} km")

    # ═══════════════════════════════════════════════════════════════════
    # Analysis 2: Field-level prediction quality
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ANALYSIS 2: GeoVLM Field Prediction Quality")
    print("=" * 60)
    print(f"  {'Field':<25} {'Hit Rate':>10} {'Samples':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")

    sorted_fields = sorted(field_hit_rates.items(), key=lambda x: x[1]["hits"] / max(x[1]["total"], 1), reverse=True)
    for field, stats in sorted_fields:
        rate = stats["hits"] / max(stats["total"], 1) * 100
        marker = " [ANCHOR]" if field in ANCHOR_FIELDS else ""
        print(f"  {field:<25} {rate:>9.1f}% {stats['total']:>10}{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # Analysis 3: Oracle ceiling
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Oracle Ceiling (Perfect Field Predictions)")
    print("=" * 60)

    vlm_arr = np.array(vlm_dists)
    oracle_arr = np.array(oracle_dists)

    print(f"  {'Metric':<20} {'VLM (actual)':>14} {'Oracle (perfect)':>16}")
    print(f"  {'-'*20} {'-'*14} {'-'*16}")
    for label, v_arr, o_arr in [
        ("Mean (km)", vlm_arr, oracle_arr),
        ("Median (km)", vlm_arr, oracle_arr),
        ("Std (km)", vlm_arr, oracle_arr),
    ]:
        v = np.mean(v_arr) if "Mean" in label else (np.median(v_arr) if "Median" in label else np.std(v_arr))
        o = np.mean(o_arr) if "Mean" in label else (np.median(o_arr) if "Median" in label else np.std(o_arr))
        print(f"  {label:<20} {v:>13.1f}  {o:>15.1f}")

    for t in [10, 50, 100, 250, 500, 1000]:
        v = np.mean(vlm_arr <= t) * 100
        o = np.mean(oracle_arr <= t) * 100
        print(f"  <= {t:>4} km{'':<11} {v:>13.1f}% {o:>15.1f}%")

    # Gap decomposition
    print(f"\n  --- Gap Decomposition ---")
    vlm_mean = np.mean(vlm_arr)
    oracle_mean = np.mean(oracle_arr)
    gap = vlm_mean - oracle_mean
    print(f"  Total gap: {gap:.1f} km")
    print(f"  VLM prediction error gap: {gap:.1f} km")
    print(f"  Ontology engine ceiling: {oracle_mean:.1f} km")

    if gap > 0:
        pct_vlm = gap / vlm_mean * 100
        pct_engine = oracle_mean / vlm_mean * 100
        print(f"  VLM error contribution: ~{pct_vlm:.0f}% of total error")
        print(f"  Engine ceiling contribution: ~{pct_engine:.0f}% of total error")


if __name__ == "__main__":
    main()
