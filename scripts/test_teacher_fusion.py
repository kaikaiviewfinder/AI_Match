"""Test: use teacher hard_labels (perfect predictions) as fusion input.
This shows the UPPER BOUND of GeoVLM's 7-element curriculum through fusion.
"""
import sys, json, math, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from regression.element_fusion import fuse_elements_v3

LABELS_FILE = "D:/Geocomp/output/geovlm_teacher_labels.jsonl"

TEACHER_TO_FUSION = [
    "climate_zone", "terrain_type", "vegetation_zone",
    "urbanization", "architecture_style", "pavement_type",
    "language_script",
]


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def main():
    samples = []
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("true_lat") and rec.get("true_lng"):
                    samples.append(rec)

    random.seed(42)
    samples = random.sample(samples, 100)

    print(f"Testing {len(samples)} samples with PERFECT teacher labels\n")

    distances = []
    for rec in samples:
        true_lat = rec["true_lat"]
        true_lng = rec["true_lng"]
        hard = rec["hard_labels"]

        sensor_elev = rec.get("sensor_elevation", 500.0)
        sensor_temp = rec.get("sensor_temperature", 20.0)
        sensor_humid = rec.get("sensor_humidity", 60.0)

        # Build fusion input from teacher hard_labels (perfect predictions)
        fusion_input = {}
        for key in TEACHER_TO_FUSION:
            val = hard.get(f"{key}_pred")
            if val and val != "UNKNOWN":
                fusion_input[key] = val
        elev = hard.get("elevation_estimate_m")
        if elev and len(elev) == 2:
            fusion_input["elevation_estimate_m"] = elev

        result = fuse_elements_v3(
            fusion_input,
            sensor_elevation_m=sensor_elev,
            sensor_temperature_c=sensor_temp,
            sensor_humidity_pct=sensor_humid,
            geocot_prediction=None,
        )

        dist = haversine_km(true_lat, true_lng, result.latitude, result.longitude)
        distances.append(dist)

    dists = np.array(distances)
    print(f"  Valid: {len(dists)}/{len(samples)}")
    print(f"  Mean:   {np.mean(dists):.1f} km")
    print(f"  Median: {np.median(dists):.1f} km")
    print(f"  Min:    {np.min(dists):.1f} km")
    print(f"  Max:    {np.max(dists):.1f} km")
    print(f"\n  Accuracy @ threshold:")
    for t in [25, 50, 100, 250, 500, 1000, 2000]:
        pct = np.mean(dists <= t) * 100
        print(f"    <={t:5d} km: {pct:5.1f}%")

    # Compare: show best/worst cases
    print(f"\n  Best 3 (perfect labels):")
    best_idx = np.argsort(dists)[:3]
    for i in best_idx:
        rec = samples[i]
        print(f"    {Path(rec['image_path']).name}: true=({rec['true_lat']:.2f},{rec['true_lng']:.2f}) err={dists[i]:.0f}km")


if __name__ == "__main__":
    main()
