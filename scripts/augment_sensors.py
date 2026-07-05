"""Augment test data with estimated temperature & humidity sensor readings.

Uses ClimateValidator to estimate realistic sensor values from lat/lng/elevation.
Adds realistic sensor noise. Saves to a new file (does not modify original).
"""
import sys, json, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from regression.climate_lookup import ClimateValidator

INPUT = "D:/Geocomp/output/geovlm_v1_results.jsonl"
OUTPUT = "D:/Geocomp/output/geovlm_v1_results_augmented.jsonl"
MONTH = 7  # July (summer — typical for outdoor photos)
RNG = np.random.RandomState(42)


def main():
    climate = ClimateValidator()

    with open(INPUT, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"Augmenting {len(lines)} samples with temp/humidity sensors...")

    n_added = 0
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for r in lines:
            lat, lng = r["true_lat"], r["true_lng"]
            elev = r.get("sensor_elevation", 500)

            # Estimate temperature
            t_lo, t_hi = climate.estimate_temperature_range(lat, lng, elev, MONTH)
            t_mid = (t_lo + t_hi) / 2
            # Add realistic sensor noise (±2°C typical for consumer sensors)
            sensor_temp = t_mid + RNG.normal(0, 1.5)

            # Estimate humidity
            h_lo, h_hi = climate.estimate_humidity_range(lat, lng, MONTH)
            h_mid = (h_lo + h_hi) / 2
            # Add realistic sensor noise (±5% RH typical)
            sensor_humid = h_mid + RNG.normal(0, 3.0)
            sensor_humid = max(5, min(100, sensor_humid))

            r["sensor_temperature"] = round(float(sensor_temp), 1)
            r["sensor_humidity"] = round(float(sensor_humid), 1)
            n_added += 1

            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {n_added} augmented samples to: {OUTPUT}")
    # Show a few examples
    for r in lines[:3]:
        print(f"  {r['true_lat']:.1f}N, {r['true_lng']:.1f}E, elev={r.get('sensor_elevation',0):.0f}m "
              f"→ temp={r['sensor_temperature']:.1f}°C, humid={r['sensor_humidity']:.0f}%")


if __name__ == "__main__":
    main()
