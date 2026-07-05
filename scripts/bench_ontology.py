"""Benchmark Ontology fusion engine with progress visualization & periodic saves."""
import sys, json, math, time, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from regression.ontology_fusion import ontology_fuse

V1_RESULTS = "D:/Geocomp/output/geovlm_v1_results.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "output" / "bench_ontology_results.json"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def main():
    with open(V1_RESULTS, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    valid = [r for r in results if "distance_km" in r and "field_predictions" in r]
    n = len(valid)
    print(f"Benchmark Ontology fusion on {n} samples")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    dists = []
    fails = 0
    times = []
    save_interval = 50
    start_time = time.time()

    for i, r in enumerate(valid):
        # Build elements from field predictions
        fp = r.get("field_predictions", {})
        elements = {}
        for field, pred in fp.items():
            if isinstance(pred, dict):
                elements[field] = pred
            else:
                elements[field] = {"value": str(pred), "confidence": 0.5}

        t0 = time.time()
        try:
            result = ontology_fuse(elements, sensor_elevation_m=r.get("sensor_elevation", 500))
            dist = haversine_km(r["true_lat"], r["true_lng"],
                               result.latitude, result.longitude)
            dists.append(dist)
        except Exception as e:
            fails += 1
            dists.append(float('nan'))
        elapsed = time.time() - t0
        times.append(elapsed)

        # Progress line
        dist_str = f"{dists[-1]:.0f}km" if not math.isnan(dists[-1]) else "FAIL"
        eta_str = ""
        if i >= 9:
            avg_t = np.mean(times[-10:])
            eta_s = avg_t * (n - i - 1)
            eta_str = f" | ETA {eta_s/60:.0f}m{eta_s%60:.0f}s"

        print(f"  [{i+1:4d}/{n}] {dist_str:>8s}  ({elapsed:.1f}s){eta_str}")

        # Periodic save
        if (i + 1) % save_interval == 0 or i == n - 1:
            valid_dists = [d for d in dists if not math.isnan(d)]
            arr = np.array(valid_dists)
            snapshot = {
                "completed": i + 1,
                "total": n,
                "fails": fails,
                "mean_km": float(np.mean(arr)),
                "median_km": float(np.median(arr)),
                "std_km": float(np.std(arr)),
                "pct_le_50": float(np.mean(arr <= 50) * 100),
                "pct_le_100": float(np.mean(arr <= 100) * 100),
                "pct_le_250": float(np.mean(arr <= 250) * 100),
                "pct_le_500": float(np.mean(arr <= 500) * 100),
                "pct_le_1000": float(np.mean(arr <= 1000) * 100),
                "elapsed_s": time.time() - start_time,
                "avg_time_s": float(np.mean(times)),
            }
            with open(OUTPUT, "w") as f:
                json.dump(snapshot, f, indent=2)
            pct_done = (i + 1) / n * 100
            print(f"  --- [{i+1}/{n}] {pct_done:.0f}% done, "
                  f"mean={snapshot['mean_km']:.0f}km, "
                  f"median={snapshot['median_km']:.0f}km, "
                  f"<=50km={snapshot['pct_le_50']:.1f}% --- saved")

    # Final summary
    total_time = time.time() - start_time
    valid_dists = [d for d in dists if not math.isnan(d)]
    arr = np.array(valid_dists)

    print(f"\n{'='*60}")
    print(f"Final Results ({n} samples, {total_time:.0f}s total)")
    print(f"{'='*60}")
    print(f"  Mean:          {np.mean(arr):.0f} km")
    print(f"  Median:        {np.median(arr):.0f} km")
    print(f"  Std:           {np.std(arr):.0f} km")
    print(f"  Fails:          {fails}")
    print(f"  Avg time/sample: {np.mean(times):.1f}s")
    print()
    for t in [10, 50, 100, 250, 500, 1000]:
        pct = np.mean(arr <= t) * 100
        print(f"  <= {t:>4} km:  {pct:5.1f}%")

    # Final save with full dists
    final = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n,
        "fails": fails,
        "total_time_s": total_time,
        "mean_km": float(np.mean(arr)),
        "median_km": float(np.median(arr)),
        "std_km": float(np.std(arr)),
        "pct_le_10": float(np.mean(arr <= 10) * 100),
        "pct_le_50": float(np.mean(arr <= 50) * 100),
        "pct_le_100": float(np.mean(arr <= 100) * 100),
        "pct_le_250": float(np.mean(arr <= 250) * 100),
        "pct_le_500": float(np.mean(arr <= 500) * 100),
        "pct_le_1000": float(np.mean(arr <= 1000) * 100),
        "dists": [float(d) if not math.isnan(d) else None for d in dists],
    }
    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFull results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
