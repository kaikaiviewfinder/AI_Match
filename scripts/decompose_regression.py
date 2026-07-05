"""Isolate: is accuracy regression from GeoVLM or Ontology fusion?

Tests both combinations to decompose the gap:
  A: GeoVLM + GeoKB (fuse_elements_v3)  — new VLM, old fusion
  B: GeoVLM + Ontology                   — new VLM, new fusion (current)

Output: terminal progress + periodic saves to bench_decompose_snapshot.json
"""
import sys, json, math, time, os, gc
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from regression.ontology_fusion import ontology_fuse
from regression.element_fusion import fuse_elements_v3

V1_RESULTS = "D:/Geocomp/output/geovlm_v1_results_augmented.jsonl"
SNAPSHOT = Path(__file__).resolve().parent.parent / "output" / "bench_decompose_snapshot.json"
SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)

SAVE_INTERVAL = 50

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


def fp_to_onto(fp):
    elements = {}
    for field, pred in fp.items():
        if isinstance(pred, dict):
            elements[field] = pred
        else:
            elements[field] = {"value": str(pred), "confidence": 0.5}
    return elements


def fp_to_flat(fp):
    result = {}
    for field in GEOVLM_TO_FUSION:
        pred = fp.get(field, "UNKNOWN")
        if isinstance(pred, dict):
            val = pred.get("value", "UNKNOWN")
        else:
            val = str(pred)
        if val and val != "UNKNOWN":
            result[field] = val
    return result


def save_snapshot(geo_dists, geo_fails, onto_dists, onto_fails, elapsed, done, total):
    geo_arr = np.array(geo_dists)
    onto_arr = np.array(onto_dists)
    out = {
        "timestamp": datetime.now().isoformat(),
        "completed": done,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "geo": {
            "fails": geo_fails,
            "mean_km": round(float(np.mean(geo_arr)), 1),
            "median_km": round(float(np.median(geo_arr)), 1),
            "std_km": round(float(np.std(geo_arr)), 1),
            "pct_le_50": round(float(np.mean(geo_arr <= 50) * 100), 1),
            "pct_le_100": round(float(np.mean(geo_arr <= 100) * 100), 1),
            "pct_le_250": round(float(np.mean(geo_arr <= 250) * 100), 1),
            "pct_le_500": round(float(np.mean(geo_arr <= 500) * 100), 1),
            "pct_le_1000": round(float(np.mean(geo_arr <= 1000) * 100), 1),
        },
        "onto": {
            "fails": onto_fails,
            "mean_km": round(float(np.mean(onto_arr)), 1),
            "median_km": round(float(np.median(onto_arr)), 1),
            "std_km": round(float(np.std(onto_arr)), 1),
            "pct_le_50": round(float(np.mean(onto_arr <= 50) * 100), 1),
            "pct_le_100": round(float(np.mean(onto_arr <= 100) * 100), 1),
            "pct_le_250": round(float(np.mean(onto_arr <= 250) * 100), 1),
            "pct_le_500": round(float(np.mean(onto_arr <= 500) * 100), 1),
            "pct_le_1000": round(float(np.mean(onto_arr <= 1000) * 100), 1),
        },
    }
    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main():
    with open(V1_RESULTS, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    valid = [r for r in results if "distance_km" in r and "field_predictions" in r]
    n = len(valid)
    print(f"Decompose benchmark: {n} samples")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Engine A: GeoVLM + GeoKB  |  Engine B: GeoVLM + Ontology")
    print(f"{'='*70}")

    geo_dists = []; geo_fails = 0; geo_times = []
    onto_dists = []; onto_fails = 0; onto_times = []
    t_start = time.time()

    for i, r in enumerate(valid):
        fp = r.get("field_predictions", {})
        true_lat, true_lng = r["true_lat"], r["true_lng"]
        sensor_elev = r.get("sensor_elevation", 500)
        sensor_temp = r.get("sensor_temperature")
        sensor_humid = r.get("sensor_humidity")

        # ── A: GeoVLM + GeoKB ────────────────────────────────────────
        t0 = time.time()
        try:
            flat = fp_to_flat(fp)
            result_a = fuse_elements_v3(flat)
            dist_a = haversine_km(true_lat, true_lng, result_a.latitude, result_a.longitude)
            geo_dists.append(dist_a)
        except Exception:
            geo_fails += 1
            geo_dists.append(float('nan'))
        geo_times.append(time.time() - t0)

        # ── B: GeoVLM + Ontology ─────────────────────────────────────
        t0 = time.time()
        try:
            elements = fp_to_onto(fp)
            result_b = ontology_fuse(elements, sensor_elevation_m=sensor_elev,
                                     sensor_temperature_c=sensor_temp,
                                     sensor_humidity_pct=sensor_humid)
            dist_b = haversine_km(true_lat, true_lng, result_b.latitude, result_b.longitude)
            onto_dists.append(dist_b)
        except Exception:
            onto_fails += 1
            onto_dists.append(float('nan'))
        onto_times.append(time.time() - t0)

        # ── Progress ─────────────────────────────────────────────────
        da = f"{geo_dists[-1]:.0f}km" if not math.isnan(geo_dists[-1]) else "FAIL"
        db = f"{onto_dists[-1]:.0f}km" if not math.isnan(onto_dists[-1]) else "FAIL"
        ta = geo_times[-1]; tb = onto_times[-1]
        # ETA
        eta_str = ""
        if i >= 4:
            avg_t = np.mean([geo_times[j]+onto_times[j] for j in range(max(0,i-4), i+1)])
            eta_s = avg_t * (n - i - 1)
            eta_str = f"ETA {eta_s/60:.0f}m{eta_s%60:.0f}s"
        line = f"  [{i+1:4d}/{n}] GeoKB:{da:>7s}({ta:.1f}s)  Ontology:{db:>7s}({tb:.1f}s)  {eta_str}"
        print(line, flush=True)

        # ── Periodic save ─────────────────────────────────────────────
        if (i + 1) % SAVE_INTERVAL == 0 or i == n - 1:
            elapsed = time.time() - t_start
            pct = (i + 1) / n * 100
            geo_arr = np.array([d for d in geo_dists if not math.isnan(d)])
            onto_arr = np.array([d for d in onto_dists if not math.isnan(d)])
            print(f"  --- [{i+1}/{n}] {pct:.0f}%  "
                  f"GeoKB: mean={np.mean(geo_arr):.0f}km median={np.median(geo_arr):.0f}km  "
                  f"Ontology: mean={np.mean(onto_arr):.0f}km median={np.median(onto_arr):.0f}km  "
                  f"elapsed={elapsed:.0f}s ---",
                  flush=True)
            save_snapshot(
                [d for d in geo_dists if not math.isnan(d)], geo_fails,
                [d for d in onto_dists if not math.isnan(d)], onto_fails,
                elapsed, i + 1, n)

        # ── OOM prevention: periodic gc ───────────────────────────────
        if (i + 1) % 100 == 0:
            gc.collect()

    # ── Final summary ──────────────────────────────────────────────────
    total_t = time.time() - t_start
    geo_arr = np.array([d for d in geo_dists if not math.isnan(d)])
    onto_arr = np.array([d for d in onto_dists if not math.isnan(d)])

    print(f"\n{'='*70}")
    print(f"Decomposition: GeoVLM + GeoKB  vs  GeoVLM + Ontology")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'GeoVLM+GeoKB':>14} {'GeoVLM+Ontology':>16} {'Delta':>10}")
    print(f"{'-'*20} {'-'*14} {'-'*16} {'-'*10}")

    for label in ["mean", "median", "std"]:
        g = getattr(np, label)(geo_arr)
        o = getattr(np, label)(onto_arr)
        d = o - g
        sign = "+" if d > 0 else ""
        if label == "mean":
            label_cn = "Mean (km)"
        elif label == "median":
            label_cn = "Median (km)"
        else:
            label_cn = "Std (km)"
        print(f"{label_cn:<20} {g:>13.1f}  {o:>15.1f}  {sign}{d:>9.1f}")

    print()
    for t in [10, 50, 100, 250, 500, 1000]:
        g = np.mean(geo_arr <= t) * 100
        o = np.mean(onto_arr <= t) * 100
        print(f"  <= {t:>4} km:  GeoKB {g:>5.1f}%  |  Ontology {o:>5.1f}%  |  Δ {o-g:+.1f}pp")

    print(f"\n  GeoKB fails: {geo_fails}, Ontology fails: {onto_fails}")
    print(f"  GeoKB mean: {np.mean(geo_arr):.1f} km")
    print(f"  Ontology mean: {np.mean(onto_arr):.1f} km")
    print(f"  Total time: {total_t:.0f}s ({total_t/60:.1f}m)")

    gap = np.mean(onto_arr) - np.mean(geo_arr)
    print(f"\n  >>> Ontology vs GeoKB gap (same GeoVLM inputs): {gap:+.1f} km")
    if abs(gap) < 50:
        print(f"  >>> CONCLUSION: GeoVLM field predictions are the main bottleneck.")
        print(f"  >>> Both fusion engines perform similarly given the same inputs.")
    else:
        print(f"  >>> CONCLUSION: Ontology fusion engine IS worse than GeoKB.")
        print(f"  >>> Gap of {gap:.0f} km is from the fusion method, not the VLM.")

    # Final save
    save_snapshot(
        [d for d in geo_dists if not math.isnan(d)], geo_fails,
        [d for d in onto_dists if not math.isnan(d)], onto_fails,
        total_t, n, n)
    print(f"\nSnapshot saved to: {SNAPSHOT}")


if __name__ == "__main__":
    main()
