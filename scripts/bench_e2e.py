"""End-to-end benchmark: Image → GeoVLM → Ontology Fusion → coordinates.

Pipeline:
  1. GeoVLM (116.5M): image + sensor → 16 field predictions
  2. Ontology Fusion: field predictions + sensors → lat/lng
  3. Haversine: pred vs ground truth distance

Features:
  - Per-sample progress with timing breakdown
  - Periodic snapshot saves every 50 samples
  - Resume from checkpoint on interrupt
  - Both mean and median reported at every interval
"""
import sys, json, math, time, os, pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from PIL import Image

from geovlm import GeoVLM, GeoVLMConfig
from geovlm.vision_encoder import prepare_multi_scale_images
from regression.ontology_fusion import ontology_fuse

CHECKPOINT = "D:/Geocomp/output/geovlm_checkpoints/geovlm_final.pt"
TEST_DATA = "D:/Geocomp/output/geovlm_v1_results_augmented.jsonl"
SNAPSHOT = Path(__file__).resolve().parent.parent / "output" / "bench_e2e_snapshot.json"
CHECKPOINT_PKL = Path(__file__).resolve().parent.parent / "output" / "bench_e2e_checkpoint.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_INTERVAL = 50

# Fields GeoVLM predicts → pass to ontology fusion
GEOVLM_FIELDS = [
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


def geovlm_to_onto_elements(geovlm_output: dict) -> dict:
    """Convert GeoVLM output elements → ontology_fuse input format."""
    elements = {}
    for field in GEOVLM_FIELDS:
        pred_key = f"{field}_pred"
        conf_key = f"{field}_confidence"
        val = geovlm_output.get(pred_key)
        conf = geovlm_output.get(conf_key, 0.5)
        if val and val != "UNKNOWN":
            elements[field] = {"value": str(val), "confidence": float(conf)}
        else:
            elements[field] = {"value": "UNKNOWN", "confidence": 0.0}
    return elements


def save_snapshot(dists, times_geovlm, times_onto, fails, elapsed, done, total):
    arr = np.array([d for d in dists if not math.isnan(d)])
    geo_t = np.array(times_geovlm)
    onto_t = np.array(times_onto)
    out = {
        "timestamp": datetime.now().isoformat(),
        "completed": done,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "fails": fails,
        "accuracy": {
            "mean_km": round(float(np.mean(arr)), 1),
            "median_km": round(float(np.median(arr)), 1),
            "std_km": round(float(np.std(arr)), 1),
            "pct_le_50": round(float(np.mean(arr <= 50) * 100), 1),
            "pct_le_100": round(float(np.mean(arr <= 100) * 100), 1),
            "pct_le_250": round(float(np.mean(arr <= 250) * 100), 1),
            "pct_le_500": round(float(np.mean(arr <= 500) * 100), 1),
            "pct_le_1000": round(float(np.mean(arr <= 1000) * 100), 1),
        },
        "timing": {
            "geovlm_mean_ms": round(float(np.mean(geo_t)) * 1000, 1),
            "geovlm_median_ms": round(float(np.median(geo_t)) * 1000, 1),
            "onto_mean_ms": round(float(np.mean(onto_t)) * 1000, 1),
            "onto_median_ms": round(float(np.median(onto_t)) * 1000, 1),
            "total_mean_ms": round(float(np.mean(geo_t + onto_t)) * 1000, 1),
        },
    }
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def save_checkpoint(processed, dists, times_geo, times_onto, fails):
    ckpt = {
        "processed_indices": list(processed),
        "distances": dists,
        "times_geovlm": times_geo,
        "times_onto": times_onto,
        "fails": fails,
    }
    with open(CHECKPOINT_PKL, "wb") as f:
        pickle.dump(ckpt, f)


def load_checkpoint():
    if CHECKPOINT_PKL.exists():
        with open(CHECKPOINT_PKL, "rb") as f:
            return pickle.load(f)
    return None


def main():
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)

    # Load test data
    with open(TEST_DATA, encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    total = len(all_samples)

    # Load GeoVLM
    print(f"Loading GeoVLM ({DEVICE})...")
    config = GeoVLMConfig()
    model = GeoVLM(config)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = {k: v for k, v in state["model"].items()
                  if not k.startswith("heads.elevation_head")}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    counts = model.count_parameters()
    print(f"  Params: {counts['total']/1e6:.1f}M  "
          f"(ViT={counts['vision_encoder']/1e6:.1f}M  "
          f"QFormer={counts['qformer']/1e6:.1f}M  "
          f"GNN={counts['constraint_graph']/1e6:.1f}M)")

    # Resume from checkpoint if exists
    ckpt = load_checkpoint()
    if ckpt:
        processed = set(ckpt["processed_indices"])
        distances = ckpt["distances"]
        times_geovlm = ckpt["times_geovlm"]
        times_onto = ckpt["times_onto"]
        fails = ckpt["fails"]
        print(f"Resumed: {len(processed)} done, {len(distances)} valid, {fails} fails")
    else:
        processed = set()
        distances = []
        times_geovlm = []
        times_onto = []
        fails = 0

    # Warmup: run one inference to prime CUDA/cache
    print(f"\nWarmup inference...")
    r0 = all_samples[0]
    img0 = Image.open(r0["image_path"]).convert("RGB")
    scales0 = prepare_multi_scale_images(img0)
    sensor0 = torch.tensor([[
        r0.get("sensor_elevation", 500),
        r0.get("sensor_temperature", 20),
        r0.get("sensor_humidity", 60),
    ]], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        _ = model(scales0, sensor0)

    print(f"\n{'='*70}")
    print(f"E2E Benchmark: Image → GeoVLM → Ontology Fusion")
    print(f"{'='*70}")
    print(f"  Samples: {total}")
    print(f"  Device:  {DEVICE}")
    print(f"  Snapshot: {SNAPSHOT}")
    print(f"  Save interval: every {SAVE_INTERVAL} samples")
    print(f"{'='*70}")

    t_start = time.time()

    for i, r in enumerate(all_samples):
        if i in processed:
            continue

        true_lat, true_lng = r["true_lat"], r["true_lng"]
        img_path = r["image_path"]
        sensor_elev = r.get("sensor_elevation", 500)
        sensor_temp = r.get("sensor_temperature", 20)
        sensor_humid = r.get("sensor_humidity", 60)

        # ── Step 1: GeoVLM inference ──────────────────────────────────
        try:
            img = Image.open(img_path).convert("RGB")
            scales = prepare_multi_scale_images(img)
            sensor_t = torch.tensor([[
                sensor_elev, sensor_temp, sensor_humid,
            ]], dtype=torch.float32, device=DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                output = model(scales, sensor_t)
                geovlm_elements = output["elements"]
            t_geo = time.perf_counter() - t0
        except Exception as e:
            fails += 1
            processed.add(i)
            print(f"  [{i+1:4d}/{total}] GeoVLM ERROR: {e}")
            continue

        # ── Step 2: Ontology Fusion ───────────────────────────────────
        try:
            elements = geovlm_to_onto_elements(geovlm_elements)
            t0 = time.perf_counter()
            result = ontology_fuse(
                elements,
                sensor_elevation_m=sensor_elev,
                sensor_temperature_c=sensor_temp,
                sensor_humidity_pct=sensor_humid,
            )
            t_onto = time.perf_counter() - t0
        except Exception as e:
            fails += 1
            processed.add(i)
            print(f"  [{i+1:4d}/{total}] Ontology ERROR: {e}")
            continue

        # ── Step 3: Distance ──────────────────────────────────────────
        dist = haversine_km(true_lat, true_lng, result.latitude, result.longitude)

        distances.append(dist)
        times_geovlm.append(t_geo)
        times_onto.append(t_onto)
        processed.add(i)

        # ── Progress ──────────────────────────────────────────────────
        done = len(processed)
        eta_str = ""
        if done >= 5:
            elapsed = time.time() - t_start
            avg_t = elapsed / done
            eta_s = avg_t * (total - done)
            eta_str = f"ETA {eta_s/60:.0f}m{eta_s%60:.0f}s"
        print(f"  [{done:4d}/{total}] err={dist:6.0f}km  "
              f"GeoVLM={t_geo*1000:5.0f}ms  Onto={t_onto*1000:5.0f}ms  {eta_str}",
              flush=True)

        # ── Periodic save ─────────────────────────────────────────────
        if done % SAVE_INTERVAL == 0 or done == total:
            elapsed = time.time() - t_start
            arr = np.array([d for d in distances if not math.isnan(d)])
            geo_t = np.array(times_geovlm)
            onto_t = np.array(times_onto)
            total_t = geo_t + onto_t
            print(f"  --- [{done}/{total}] "
                  f"mean={np.mean(arr):.0f}km median={np.median(arr):.0f}km  "
                  f"GeoVLM mean={np.mean(geo_t)*1000:.0f}ms  "
                  f"Onto mean={np.mean(onto_t)*1000:.0f}ms  "
                  f"Total mean={np.mean(total_t)*1000:.0f}ms  "
                  f"elapsed={elapsed:.0f}s ---",
                  flush=True)
            save_snapshot(distances, times_geovlm, times_onto, fails, elapsed, done, total)
            save_checkpoint(processed, distances, times_geovlm, times_onto, fails)

        # Periodic garbage collection
        if done % 100 == 0:
            import gc
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ── Final summary ──────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    arr = np.array([d for d in distances if not math.isnan(d)])
    geo_t = np.array(times_geovlm)
    onto_t = np.array(times_onto)

    print(f"\n{'='*70}")
    print(f"E2E Benchmark Results: Image → GeoVLM → Ontology Fusion")
    print(f"{'='*70}")
    print(f"  Valid: {len(arr)}/{total}  Fails: {fails}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"\n  Accuracy:")
    print(f"    Mean:   {np.mean(arr):.1f} km")
    print(f"    Median: {np.median(arr):.1f} km")
    print(f"    Std:    {np.std(arr):.1f} km")
    print(f"    Min:    {np.min(arr):.1f} km")
    print(f"    Max:    {np.max(arr):.1f} km")

    print(f"\n  Percentiles:")
    for t in [10, 50, 100, 250, 500, 1000]:
        pct = np.mean(arr <= t) * 100
        print(f"    <= {t:>4} km:  {pct:>5.1f}%")

    print(f"\n  Timing (per sample):")
    print(f"    GeoVLM:  mean={np.mean(geo_t)*1000:.0f}ms  median={np.median(geo_t)*1000:.0f}ms")
    print(f"    Ontology: mean={np.mean(onto_t)*1000:.0f}ms  median={np.median(onto_t)*1000:.0f}ms")
    total_t = geo_t + onto_t
    print(f"    Total:   mean={np.mean(total_t)*1000:.0f}ms  median={np.median(total_t)*1000:.0f}ms")

    # Final save
    save_snapshot(distances, times_geovlm, times_onto, fails, total_elapsed, len(arr), total)
    save_checkpoint(processed, distances, times_geovlm, times_onto, fails)
    print(f"\n  Snapshot: {SNAPSHOT}")

    # Clean checkpoint on success
    if CHECKPOINT_PKL.exists() and len(arr) == total:
        CHECKPOINT_PKL.unlink()
        print(f"  Checkpoint removed (complete)")


if __name__ == "__main__":
    main()
