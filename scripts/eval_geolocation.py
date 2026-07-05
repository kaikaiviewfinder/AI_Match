"""End-to-end geolocation accuracy: GeoVLM elements → element_fusion → lat/lng.

Pipeline:
  GeoVLM(image) → 8 field predictions + elevation from sensor
       ↓
  fuse_elements_v3(elements, sensors) → FusionResult(lat, lng, uncertainty)
       ↓
  haversine(pred_lat, pred_lng, true_lat, true_lng) → distance error (km)

Reports:
  - Mean/median/min/max distance error
  - Accuracy@k km (e.g. within 50km, 100km, 500km, 1000km)
  - Per-element-availability breakdown

Supports --resume to continue from checkpoint (auto-saved every 50 samples).
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
from regression.element_fusion import fuse_elements_v3, fuse_elements

CHECKPOINT = "D:/Geocomp/output/geovlm_checkpoints/geovlm_final.pt"
LABELS_FILE = "D:/Geocomp/output/geovlm_teacher_labels.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GeoVLM field keys → element_fusion keys (strip _pred suffix)
GEOVLM_TO_FUSION = [
    "climate_zone", "terrain_type", "vegetation_zone",
    "urbanization", "architecture_style", "pavement_type",
    "language_script",
    # New spatial-narrowing fields
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


def geovlm_elements_to_fusion(geovlm_elements: dict, sensor: dict) -> dict:
    """Convert GeoVLM output elements → element_fusion plain-key format."""
    result = {}
    for key in GEOVLM_TO_FUSION:
        val = geovlm_elements.get(f"{key}_pred")
        if val is not None and val != "UNKNOWN":
            result[key] = val

    # Elevation from GeoVLM (copied from sensor in _build_elements_dict)
    elev = geovlm_elements.get("elevation_estimate_m")
    if elev and len(elev) == 2:
        result["elevation_estimate_m"] = elev

    # Ruled-out features
    ruled = geovlm_elements.get("ruled_out_features")
    if ruled:
        result["ruled_out_features"] = ruled

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GeoVLM end-to-end geolocation evaluation")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit to first N samples (0=all)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint file")
    parser.add_argument("--checkpoint", type=str,
                        default="D:/Geocomp/output/eval_geolocation_checkpoint.pkl",
                        help="Checkpoint file path")
    args = parser.parse_args()

    print("=" * 60)
    print("GeoVLM End-to-End Geolocation Evaluation")
    print("=" * 60)

    # Load model
    print(f"\n[1] Loading GeoVLM ({DEVICE})...")
    config = GeoVLMConfig()
    model = GeoVLM(config)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = {k: v for k, v in state["model"].items()
                  if not k.startswith("heads.elevation_head")}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # Load labels
    print(f"\n[2] Loading teacher labels...")
    samples = []
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                path = rec.get("image_path", "")
                # Must have both image and ground truth coordinates
                if path and Path(path).exists() and "true_lat" in rec and "true_lng" in rec:
                    samples.append(rec)
    print(f"  Valid samples: {len(samples)}")
    if args.max_samples > 0 and args.max_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, args.max_samples)
        print(f"  Subsampled to:  {len(samples)}")
        random.shuffle(samples)

    # Evaluate
    print(f"\n[3] Running {len(samples)} samples through full pipeline...\n")

    # Resume from checkpoint if requested
    processed_indices = set()
    distances = []
    uncertainties = []
    confidences = []
    fusion_errors = []
    per_field_errors = defaultdict(list)

    if args.resume:
        resume_path = args.resume
    else:
        resume_path = args.checkpoint

    if os.path.exists(resume_path):
        print(f"  Loading checkpoint: {resume_path}")
        with open(resume_path, "rb") as f:
            ckpt = pickle.load(f)
        distances = ckpt.get("distances", [])
        uncertainties = ckpt.get("uncertainties", [])
        confidences = ckpt.get("confidences", [])
        fusion_errors = ckpt.get("fusion_errors", [])
        per_field_errors = defaultdict(list, ckpt.get("per_field_errors", {}))
        processed_indices = set(ckpt.get("processed_indices", []))
        print(f"  Resumed: {len(processed_indices)} already processed, {len(distances)} valid predictions")

    checkpoint_interval = 50

    for idx, rec in enumerate(tqdm(samples, desc="GeoVLM → fusion")):
        if idx in processed_indices:
            continue

        true_lat = rec["true_lat"]
        true_lng = rec["true_lng"]

        # Sensor data from teacher labels (simulates DK-2500 onboard sensors)
        sensor_elev = rec.get("sensor_elevation", 500.0)
        sensor_temp = rec.get("sensor_temperature", 20.0)
        sensor_humid = rec.get("sensor_humidity", 60.0)

        # ── Step 1: GeoVLM inference ──────────────────────────────────
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            scales = prepare_multi_scale_images(img)
            sensor_tensor = torch.tensor([[
                sensor_elev, sensor_temp, sensor_humid,
            ]], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                output = model(scales, sensor_tensor)
                elements = output["elements"]
        except Exception as e:
            fusion_errors.append(f"[{idx}] GeoVLM: {e}")
            processed_indices.add(idx)
            continue

        # ── Step 2: Convert + run element fusion ──────────────────────
        fusion_input = geovlm_elements_to_fusion(elements, {
            "elevation": sensor_elev,
            "temperature": sensor_temp,
            "humidity": sensor_humid,
        })

        try:
            result = fuse_elements_v3(
                fusion_input,
                sensor_elevation_m=sensor_elev,
                sensor_temperature_c=sensor_temp,
                sensor_humidity_pct=sensor_humid,
                geocot_prediction=None,  # No GeoCoT — pure GeoVLM + GeoKB
            )
        except Exception as e:
            fusion_errors.append(f"[{idx}] Fusion: {e}")
            processed_indices.add(idx)
            continue

        if result.latitude is None or result.longitude is None:
            fusion_errors.append(f"[{idx}] Fusion returned None coordinates")
            processed_indices.add(idx)
            continue

        # ── Step 3: Compute distance error ────────────────────────────
        dist = haversine_km(true_lat, true_lng, result.latitude, result.longitude)
        distances.append(dist)
        uncertainties.append(result.uncertainty_km)
        confidences.append(result.confidence)
        processed_indices.add(idx)

        # Per-field contribution analysis
        for key in GEOVLM_TO_FUSION:
            val = elements.get(f"{key}_pred")
            if val and val != "UNKNOWN":
                per_field_errors[f"has_{key}"].append(dist)

        # ── Periodic checkpoint + GPU memory cleanup ──────────────────
        if len(processed_indices) % checkpoint_interval == 0:
            ckpt = {
                "processed_indices": list(processed_indices),
                "distances": distances,
                "uncertainties": uncertainties,
                "confidences": confidences,
                "fusion_errors": fusion_errors,
                "per_field_errors": dict(per_field_errors),
            }
            os.makedirs(os.path.dirname(resume_path), exist_ok=True)
            with open(resume_path, "wb") as f:
                pickle.dump(ckpt, f)
            torch.cuda.empty_cache()

    if not distances:
        print(f"\nFATAL: No valid predictions ({len(fusion_errors)} errors)")
        print("Sample errors:", fusion_errors[:5])
        return

    # ── Statistics ────────────────────────────────────────────────────
    dists = np.array(distances)
    print("\n" + "=" * 60)
    print("End-to-End Geolocation Accuracy")
    print("=" * 60)
    print(f"  Valid predictions: {len(dists)} / {len(samples)}")
    if fusion_errors:
        print(f"  Fusion errors:      {len(fusion_errors)}")

    print(f"\n  Mean distance:   {np.mean(dists):.1f} km")
    print(f"  Median distance: {np.median(dists):.1f} km")
    print(f"  Std:             {np.std(dists):.1f} km")
    print(f"  Min:             {np.min(dists):.1f} km")
    print(f"  Max:             {np.max(dists):.1f} km")

    # Accuracy@k
    print(f"\n  Accuracy @ thresholds:")
    thresholds = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
    for t in thresholds:
        pct = np.mean(dists <= t) * 100
        bar = "#" * int(pct / 5)
        print(f"    ≤{t:5d} km: {pct:5.1f}%  {bar}")

    # Percentiles
    print(f"\n  Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:2d}: {np.percentile(dists, p):.1f} km")

    print(f"\n  Mean uncertainty: {np.mean(uncertainties):.0f} km (model's own estimate)")
    print(f"  Mean confidence:  {np.mean(confidences):.2f}")

    # Calibration: how often is true location within the model's uncertainty radius?
    within_uncertainty = sum(
        d <= u + 1e-6 for d, u in zip(distances, uncertainties)
    )
    print(f"  Within uncertainty radius: {within_uncertainty}/{len(dists)} ({100*within_uncertainty/len(dists):.1f}%)")

    # Per-field analysis: how well does each field alone narrow down location?
    print(f"\n  Avg distance when each field IS predicted:")
    for key in sorted(per_field_errors.keys(), key=lambda k: np.mean(per_field_errors[k])):
        vals = per_field_errors[key]
        print(f"    {key:<30s} mean={np.mean(vals):.0f} km (n={len(vals)})")

    print(f"\nNote: Uses fuse_elements_v3 (no GeoCoT), pure GeoVLM + GeoKB fusion.")
    print(f"      Sensor data from teacher labels (elevation, temperature, humidity).")

    # Clean up checkpoint on successful completion
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print(f"\n  Checkpoint removed: {resume_path}")


if __name__ == "__main__":
    main()
