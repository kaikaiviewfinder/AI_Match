"""GeoVLM deployment inference — works on both PC (ONNX Runtime) and DK-2500 (AscendCL).

PC testing:
    python deploy_infer.py --image test.jpg --backend onnx

DK-2500 NPU (after ATC conversion to OM):
    python deploy_infer.py --image test.jpg --backend ascend --model geovlm_deploy.om
"""
import sys, argparse, json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ============================================================
# Field vocabularies — must match model.py FIELD_VOCABS exactly
# ============================================================
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


# ============================================================
# Preprocessing
# ============================================================
def preprocess_image(image_path: str) -> dict:
    """Prepare inputs for GeoVLM ONNX/OM model.

    All three pixel inputs are 224x224 — ViT internally resizes everything.
    Multi-granularity comes from ToMe r values (0, 13, 23), not resolution.
    """
    img = Image.open(image_path).convert("RGB")
    img_224 = img.resize((224, 224))

    # Convert to numpy array [H, W, C] → [C, H, W] → add batch dim
    pixel = np.array(img_224, dtype=np.float32).transpose(2, 0, 1)[np.newaxis, :, :, :]

    return {
        "pixel_coarse": pixel.copy(),
        "pixel_mid": pixel.copy(),
        "pixel_fine": pixel.copy(),
        "sensor": np.array([[500.0, 20.0, 60.0]], dtype=np.float32),
    }


# ============================================================
# ONNX Runtime backend (PC testing, also works on DK-2500 CPU)
# ============================================================
class ONNXBackend:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.output_names = [out.name for out in self.session.get_outputs()]

    def infer(self, inputs: dict) -> list:
        return self.session.run(self.output_names, inputs)


# ============================================================
# AscendCL backend (DK-2500 NPU)
# ============================================================
class AscendBackend:
    def __init__(self, model_path: str):
        import acl
        self.acl = acl
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")

        self.device_id = 0
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError(f"acl.rt.set_device failed: {ret}")

        # Load OM model
        self.model_id, _ = acl.mdl.load_from_file(model_path)
        self.model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.model_desc, self.model_id)

        self.input_dims = [acl.mdl.get_input_size(self.model_desc, i)
                          for i in range(acl.mdl.get_num_inputs(self.model_desc))]
        self.output_dims = [acl.mdl.get_output_size(self.model_desc, i)
                           for i in range(acl.mdl.get_num_outputs(self.model_desc))]

    def infer(self, inputs: dict) -> list:
        acl = self.acl

        # Create input dataset
        input_data = []
        for name, data_np in inputs.items():
            data_np = np.ascontiguousarray(data_np)
            data_ptr = acl.util.numpy_to_ptr(data_np)
            input_data.append({
                "data": data_ptr,
                "size": int(data_np.nbytes),
            })

        dataset_in = acl.mdl.create_dataset()
        for item in input_data:
            buf = acl.create_data_buffer(item["data"], item["size"])
            acl.mdl.add_dataset_buffer(dataset_in, buf)

        # Create output dataset
        dataset_out = acl.mdl.create_dataset()
        for dim in self.output_dims:
            buf = acl.create_data_buffer(acl.rt.malloc(dim, 0), dim)
            acl.mdl.add_dataset_buffer(dataset_out, buf)

        # Execute
        acl.mdl.execute(self.model_id, dataset_in, dataset_out)

        # Extract outputs
        outputs = []
        for i in range(acl.mdl.get_dataset_num_buffers(dataset_out)):
            buf = acl.mdl.get_dataset_buffer(dataset_out, i)
            data_ptr = acl.get_data_buffer_addr(buf)
            size = acl.get_data_buffer_size(buf)
            out_np = acl.util.ptr_to_numpy(data_ptr, (size // 4,), np.float32).copy()
            outputs.append(out_np)

        acl.mdl.destroy_dataset(dataset_in)
        acl.mdl.destroy_dataset(dataset_out)
        return outputs


# ============================================================
# Decode model outputs → structured prediction
# ============================================================
def decode_outputs(outputs: list, sensor: list, top_k: int = 3) -> dict:
    """Convert raw model outputs to human-readable predictions.
    Elevation comes from sensor data, not model output.
    """
    result = {}

    # 8 classification heads
    for i, field_name in enumerate(FIELD_NAMES):
        logits = outputs[i].flatten()
        vocab = FIELD_VOCABS[field_name]

        # Softmax
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()

        top_indices = np.argsort(probs)[::-1][:top_k]
        result[field_name] = {
            "best": vocab[top_indices[0]],
            "confidence": round(float(probs[top_indices[0]]), 3),
            "top_k": [
                (vocab[idx], round(float(probs[idx]), 3))
                for idx in top_indices
            ],
        }

    # Elevation: use sensor data directly (more accurate than model prediction)
    elev_m = sensor[0]
    result["elevation_estimate_m"] = [round(elev_m), round(elev_m)]

    return result


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GeoVLM deployment inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "ascend"],
                        help="Inference backend (default: onnx)")
    parser.add_argument("--model", default=None,
                        help="Model path (default: auto-detect in ../output/geovlm_deploy/)")
    parser.add_argument("--sensor", nargs=3, type=float, default=[500.0, 20.0, 60.0],
                        metavar=("ELEV_M", "TEMP_C", "HUMID_PCT"),
                        help="Sensor readings (default: 500 20 60)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top predictions per field (default: 3)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Auto-detect model path
    if args.model is None:
        repo_root = Path(__file__).resolve().parent.parent
        deploy_dir = repo_root / "models"
        if args.backend == "ascend":
            candidate = deploy_dir / "geovlm_deploy.om"
        else:
            candidate = deploy_dir / "geovlm_fp16.onnx"
            if not candidate.exists():
                candidate = deploy_dir / "geovlm_fp32.onnx"
        args.model = str(candidate)

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Preprocess
    inputs = preprocess_image(args.image)
    inputs["sensor"] = np.array([args.sensor], dtype=np.float32)

    # Inference
    if args.backend == "onnx":
        backend = ONNXBackend(args.model)
    else:
        backend = AscendBackend(args.model)

    outputs = backend.infer(inputs)
    result = decode_outputs(outputs, args.sensor, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\nGeoVLM Prediction for: {args.image}")
        print(f"  Sensor: elev={args.sensor[0]}m, temp={args.sensor[1]}C, humid={args.sensor[2]}%")
        print()
        for field_name, pred in result.items():
            if field_name == "elevation_estimate_m":
                print(f"  {field_name}: {pred[0]}m — {pred[1]}m")
            else:
                top_str = ", ".join(f"{v}({c:.2f})" for v, c in pred["top_k"])
                print(f"  {field_name}: {pred['best']} ({pred['confidence']:.3f})  [{top_str}]")


if __name__ == "__main__":
    main()
