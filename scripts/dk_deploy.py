"""DK-2500 geolocation inference — minimal deps: onnxruntime + numpy + pillow.

Single entry point for end-to-end inference on edge device:
  python dk_deploy.py --image photo.jpg --elev 500 --temp 25 --humid 60

No PyTorch, no transformers, no CUDA. Peak memory < 3 GB (safe on 8 GB DK-2500).
"""
import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── Resolve repo root ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "geovlm_fp16.onnx"
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── Field vocabularies (must match model.py FIELD_VOCABS exactly) ──
FIELD_VOCABS = {
    "climate_zone": ["tropical", "subtropical", "temperate", "arid", "alpine", "boreal"],
    "terrain_type": ["urban_flat", "farmland_plain", "rolling_hills", "sharp_mountains",
                     "karst_peaks", "sandstone_pillars", "desert_dunes", "grassland_steppe", "plateau"],
    "vegetation_zone": ["tropical_rainforest", "broadleaf_evergreen", "broadleaf_deciduous",
                        "conifer_forest", "mixed_forest", "alpine_meadow", "desert_scrub",
                        "grassland", "bamboo_forest", "cropland", "sparse"],
    "urbanization": ["metropolis", "medium_city", "small_town", "village", "rural", "wilderness"],
    "architecture_style": ["modern_glass", "modern_residential", "old_residential", "hui_style",
                           "tibetan_stone", "courtyard", "arcade", "stilt_house", "tulou",
                           "shikumen", "traditional_official", "soviet_industrial", "none_visible"],
    "pavement_type": ["red_brick_tiles", "grey_concrete", "asphalt", "natural", "not_visible"],
    "language_script": ["simplified_chinese", "traditional_chinese", "tibetan", "uyghur_arabic",
                        "mongolian", "bilingual_cn_en", "none_visible"],
    "visible_text": ["local_government", "local_phone_code", "local_license_plate",
                     "local_street_sign", "cross_region_shop", "national_chain",
                     "national_slogan", "none_visible"],
    "soil_color": ["red", "yellow", "brown", "black", "grey", "loess_yellow", "white"],
    "sky_quality": ["clear_blue", "grey_hazy", "thick_fog", "dusty_yellow"],
    "mountain_rock_type": ["granite_spheroidal", "quartz_sandstone_pillars",
                           "limestone_karst", "red_sandstone_danxia",
                           "snow_peaks_glaciers", "alpine_lakes", "none_visible"],
    "tree_species": ["banyan", "camphor", "plane_tree", "gingko", "poplar",
                     "coconut_palm", "chinese_red_pine", "bamboo", "none_visible"],
    "building_height": ["super_tall", "high_rise", "mid_rise", "low_rise", "none_visible"],
    "water_type": ["ocean", "lake", "glacier", "waterfall", "river", "stream", "none_visible"],
    "landform_detail": ["limestone_tower_karst_scattered", "limestone_tower_karst_dense",
                        "granite_spheroidal_rounded", "granite_spheroidal_sharp",
                        "quartz_sandstone_pillars", "red_sandstone_danxia",
                        "snow_peaks_glaciers", "alpine_meadow", "plateau_plain",
                        "river_canyon", "granite_boulder_field", "coastal_shore",
                        "urban_flat", "none_visible"],
    "scene_type": ["desert", "grassland", "coastal", "river_valley", "city_street",
                   "mountain_trail_karst", "mountain_trail_granite", "mountain_trail_snow",
                   "rural_village", "tourist_scenic", "wilderness_forest", "none_visible"],
}
FIELD_NAMES = list(FIELD_VOCABS.keys())
NUM_FIELDS = len(FIELD_NAMES)

# Fields that feed into ontology fusion
ONTO_FIELDS = [
    "climate_zone", "terrain_type", "vegetation_zone",
    "urbanization", "architecture_style", "pavement_type",
    "language_script",
    "soil_color", "sky_quality", "mountain_rock_type",
    "tree_species", "building_height", "water_type",
    "landform_detail", "scene_type",
]


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and resize image to 224², return [1,3,224,224] float32."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # CHW
    return arr[np.newaxis, :, :, :]  # add batch dim


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def build_onto_input(field_preds: dict) -> dict:
    """Convert GeoVLM field predictions → ontology_fuse input format."""
    elements = {}
    for field in ONTO_FIELDS:
        pred = field_preds.get(field)
        if pred and pred != "UNKNOWN":
            elements[field] = {"value": pred, "confidence": 0.5}
        else:
            elements[field] = {"value": "UNKNOWN", "confidence": 0.0}
    return elements


class DKGeoLocator:
    """Self-contained geolocation engine for DK-2500.

    Usage:
        locator = DKGeoLocator()
        result = locator.locate("photo.jpg", elev=500, temp=25, humid=60)
        print(f"{result['lat']:.4f}, {result['lng']:.4f}  error_est={result['uncertainty_km']:.0f}km")
    """

    def __init__(self, model_path: Optional[str] = None):
        import onnxruntime as ort

        mp = model_path or str(MODEL_PATH)
        if not os.path.exists(mp):
            raise FileNotFoundError(f"ONNX model not found: {mp}")

        print(f"[DK] Loading ONNX model: {Path(mp).name}")
        t0 = time.perf_counter()

        # Single-thread CPU execution — minimises memory on edge device
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        self.session = ort.InferenceSession(
            mp,
            sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[DK] Model loaded in {load_ms:.0f}ms  "
              f"({len(self.input_names)} inputs, {len(self.output_names)} outputs)")

        # Lazy-load ontology fusion (loads JSON + embeddings ~0.5 MB total)
        self._fusion = None

    def _get_fusion(self):
        if self._fusion is None:
            print("[DK] Loading ontology fusion engine...")
            from regression.ontology_fusion import OntologyFusion
            self._fusion = OntologyFusion()
        return self._fusion

    def _infer_geovlm(self, pixel: np.ndarray, sensor: np.ndarray) -> dict:
        """Run ONNX inference → decoded field predictions."""
        inputs = {
            "pixel_coarse": pixel,
            "pixel_mid": pixel,
            "pixel_fine": pixel,
            "sensor": sensor,
        }

        outputs = self.session.run(self.output_names, inputs)

        predictions = {}
        for i, field_name in enumerate(FIELD_NAMES):
            logits = outputs[i].flatten()
            vocab = FIELD_VOCABS[field_name]
            probs = softmax(logits)
            best_idx = int(np.argmax(probs))
            predictions[field_name] = vocab[best_idx]
            predictions[f"{field_name}_confidence"] = float(probs[best_idx])

        return predictions

    def locate(self, image_path: str,
               elevation_m: float = 500.0,
               temperature_c: float = 20.0,
               humidity_pct: float = 60.0,
               ) -> dict:
        """Run full pipeline: image + sensors → lat/lng.

        Returns dict with keys:
          lat, lng           — predicted coordinates
          uncertainty_km     — model uncertainty radius
          confidence         — fusion confidence score
          timing_ms          — per-stage timing breakdown
          field_predictions  — raw GeoVLM field outputs
        """
        timing = {}

        # ── 1. Preprocessing ──────────────────────────────────────
        t0 = time.perf_counter()
        pixel = preprocess_image(image_path)
        sensor = np.array([[elevation_m, temperature_c, humidity_pct]],
                          dtype=np.float32)
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # ── 2. GeoVLM ONNX inference ──────────────────────────────
        t0 = time.perf_counter()
        field_preds = self._infer_geovlm(pixel, sensor)
        timing["geovlm_ms"] = (time.perf_counter() - t0) * 1000

        # ── 3. Ontology fusion ────────────────────────────────────
        t0 = time.perf_counter()
        onto_input = build_onto_input(field_preds)
        result = self._get_fusion().fuse(
            onto_input,
            sensor_elevation_m=elevation_m,
            sensor_temperature_c=temperature_c,
            sensor_humidity_pct=humidity_pct,
        )
        timing["fusion_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "lat": round(result.latitude, 6),
            "lng": round(result.longitude, 6),
            "uncertainty_km": round(result.uncertainty_km, 1),
            "confidence": round(result.confidence, 4),
            "timing_ms": timing,
            "field_predictions": {
                f: field_preds.get(f, "UNKNOWN") for f in ONTO_FIELDS
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="DK-2500 visual geolocation inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default=None,
                        help=f"ONNX model path (default: {MODEL_PATH})")
    parser.add_argument("--elev", type=float, default=500.0,
                        help="Sensor elevation (m)")
    parser.add_argument("--temp", type=float, default=20.0,
                        help="Sensor temperature (C)")
    parser.add_argument("--humid", type=float, default=60.0,
                        help="Sensor humidity (%%)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    t_total = time.perf_counter()
    locator = DKGeoLocator(model_path=args.model)
    result = locator.locate(
        args.image,
        elevation_m=args.elev,
        temperature_c=args.temp,
        humidity_pct=args.humid,
    )
    total_ms = (time.perf_counter() - t_total) * 1000

    if args.json:
        result["total_ms"] = round(total_ms)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        t = result["timing_ms"]
        print(f"\n{'='*55}")
        print(f"  DK-2500 Geolocation Result")
        print(f"{'='*55}")
        print(f"  Image:   {args.image}")
        print(f"  Sensors: elev={args.elev}m  temp={args.temp}C  humid={args.humid}%")
        print(f"\n  Predicted:  {result['lat']:.6f}, {result['lng']:.6f}")
        print(f"  Uncertainty: {result['uncertainty_km']:.0f} km")
        print(f"  Confidence:  {result['confidence']:.2f}")
        print(f"\n  Timing:")
        print(f"    Preprocess:  {t['preprocess_ms']:5.0f} ms")
        print(f"    GeoVLM:      {t['geovlm_ms']:5.0f} ms")
        print(f"    Fusion:      {t['fusion_ms']:5.0f} ms")
        print(f"    Total:       {total_ms:5.0f} ms")
        print(f"\n  Field Predictions:")
        for field, value in result["field_predictions"].items():
            print(f"    {field:<20s} {value}")


if __name__ == "__main__":
    main()
