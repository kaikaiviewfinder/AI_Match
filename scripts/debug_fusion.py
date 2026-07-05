"""Debug: show what fusion engine does internally for a few samples."""
import sys, json, math, io
from pathlib import Path

# Fix GBK encoding issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
from PIL import Image

from geovlm import GeoVLM, GeoVLMConfig
from geovlm.vision_encoder import prepare_multi_scale_images
from regression.element_fusion import fuse_elements_v3

CHECKPOINT = "D:/Geocomp/output/geovlm_checkpoints/geovlm_final.pt"
LABELS_FILE = "D:/Geocomp/output/geovlm_teacher_labels.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEOVLM_TO_FUSION = [
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
    print("Loading model...")
    config = GeoVLMConfig()
    model = GeoVLM(config)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = {k: v for k, v in state["model"].items()
                  if not k.startswith("heads.elevation_head")}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # Load a few samples
    samples = []
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                path = rec.get("image_path", "")
                if path and Path(path).exists() and "true_lat" in rec:
                    samples.append(rec)

    import random
    random.seed(42)
    samples = random.sample(samples, 5)

    for i, rec in enumerate(samples):
        print("\n" + "=" * 70)
        print(f"Sample {i+1}: {Path(rec['image_path']).name}")
        print(f"  True: ({rec['true_lat']:.4f}, {rec['true_lng']:.4f})")
        sensor_elev = rec.get("sensor_elevation", 500.0)
        sensor_temp = rec.get("sensor_temperature", 20.0)
        sensor_humid = rec.get("sensor_humidity", 60.0)
        print(f"  Sensor: elev={sensor_elev:.0f}m, temp={sensor_temp:.1f}C, humid={sensor_humid:.0f}%")

        # GeoVLM inference
        img = Image.open(rec["image_path"]).convert("RGB")
        scales = prepare_multi_scale_images(img)
        sensor_tensor = torch.tensor([[
            sensor_elev, sensor_temp, sensor_humid,
        ]], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            output = model(scales, sensor_tensor)
            elements = output["elements"]

        print(f"\n  GeoVLM predictions:")
        for key in GEOVLM_TO_FUSION:
            val = elements.get(f"{key}_pred", "MISSING")
            conf = elements.get(f"{key}_confidence", 0)
            print(f"    {key:<20s} -> {str(val):<25s} (conf={conf:.2f})")

        # Convert and fuse
        fusion_input = {}
        for key in GEOVLM_TO_FUSION:
            val = elements.get(f"{key}_pred")
            if val is not None and val != "UNKNOWN":
                fusion_input[key] = val
        elev = elements.get("elevation_estimate_m")
        if elev and len(elev) == 2:
            fusion_input["elevation_estimate_m"] = elev

        result = fuse_elements_v3(
            fusion_input,
            sensor_elevation_m=sensor_elev,
            sensor_temperature_c=sensor_temp,
            sensor_humidity_pct=sensor_humid,
            geocot_prediction=None,
        )

        dist = haversine_km(rec["true_lat"], rec["true_lng"],
                           result.latitude, result.longitude)
        print(f"\n  Prediction: ({result.latitude:.4f}, {result.longitude:.4f})")
        print(f"  Uncertainty: {result.uncertainty_km:.0f} km, Confidence: {result.confidence:.2f}")
        print(f"  Distance error: {dist:.1f} km")
        print(f"\n  Fusion explanation:")
        for line in result.explanation_parts:
            print(f"    {line}")


if __name__ == "__main__":
    main()
