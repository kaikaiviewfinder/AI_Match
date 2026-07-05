"""Test the geographic element fusion engine.

First: manual test cases to validate fusion logic (no VLM needed).
Then: full GeoCoT pipeline test on a few images (requires GPU).

Usage:
    python scripts/test_fusion_engine.py              # manual tests only
    python scripts/test_fusion_engine.py --geocot 5   # GeoCoT on 5 test images
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.element_fusion import (
    fuse_elements, run_fusion_pipeline, parse_geocot_json, extract_all_elements,
)
from regression.coord_regressor import haversine_distance_km


def manual_test_cases():
    """Test fusion engine with hand-crafted geographic elements."""
    print("=" * 70)
    print("MANUAL FUSION ENGINE TESTS")
    print("=" * 70)

    cases = [
        {
            "name": "阳朔喀斯特峰林",
            "elements": {
                "climate_zone": "subtropical",
                "terrain_type": "karst_peaks",
                "vegetation_zone": "broadleaf_evergreen",
                "language_script": "simplified_chinese",
                "soil_color": "red",
                "elevation_estimate_m": 300,
            },
            "expected_region": "广西/贵州",
        },
        {
            "name": "贡嘎雪山",
            "elements": {
                "climate_zone": "alpine",
                "terrain_type": "sharp_mountains",
                "vegetation_zone": "alpine_meadow",
                "mountain_rock_type": "snow_peaks_glaciers",
                "language_script": "tibetan",
                "elevation_estimate_m": 4500,
            },
            "expected_region": "川西/藏东南",
        },
        {
            "name": "深圳城市街景",
            "elements": {
                "climate_zone": "subtropical",
                "terrain_type": "urban_flat",
                "vegetation_zone": "broadleaf_evergreen",
                "language_script": "simplified_chinese",
                "architecture_style": "modern_glass",
                "tree_species": ["banyan"],
                "sky_quality": "grey_hazy",
                "elevation_estimate_m": 50,
                "urbanization": "metropolis",
                "likely_provinces": ["Guangdong"],
            },
            "expected_region": "广东/福建沿海",
        },
        {
            "name": "哈尔滨中央大街",
            "elements": {
                "climate_zone": "temperate",
                "terrain_type": "urban_flat",
                "vegetation_zone": "mixed_forest",
                "language_script": "simplified_chinese",
                "tree_species": ["poplar"],
                "soil_color": "black",
                "elevation_estimate_m": 150,
                "urbanization": "medium_city",
                "likely_provinces": ["Heilongjiang"],
            },
            "expected_region": "东北",
        },
        {
            "name": "鳌太线（秦岭高山草甸+针叶林+花岗岩）",
            "elements": {
                "climate_zone": "alpine",
                "terrain_type": "sharp_mountains",
                "vegetation_zone": "conifer_forest",
                "mountain_rock_type": "granite_spheroidal",
                "language_script": "simplified_chinese",
                "elevation_estimate_m": 3200,
            },
            "expected_region": "秦岭/华山",
        },
        {
            "name": "青藏高原荒漠",
            "elements": {
                "climate_zone": "arid",
                "terrain_type": "plateau",
                "vegetation_zone": "desert_scrub",
                "language_script": "tibetan",
                "sky_quality": "clear_blue",
                "elevation_estimate_m": 4500,
            },
            "expected_region": "青藏高原西部",
        },
    ]

    for case in cases:
        print(f"\n--- {case['name']} ---")
        print(f"  Elements: {list(case['elements'].keys())}")
        print(f"  Expected: {case['expected_region']}")

        result = fuse_elements(
            case["elements"],
            sensor_elevation_m=case["elements"].get("elevation_estimate_m"),
        )

        print(f"  Result: ({result.latitude:.2f}, {result.longitude:.2f})")
        print(f"  Uncertainty: ±{result.uncertainty_km:.0f} km")
        print(f"  Confidence: {result.confidence:.1%}")
        if result.candidate_region:
            print(f"  Active elements: {result.candidate_region.active_elements}")
            if result.candidate_region.dropped_elements:
                print(f"  Dropped (conflict): {result.candidate_region.dropped_elements}")
            print(f"  Candidate regions:")
            for b in result.candidate_region.bboxes:
                print(f"    - {b.label}: ({b.lat_min:.1f}-{b.lat_max:.1f}N, {b.lng_min:.1f}-{b.lng_max:.1f}E)")

    # Also test with sensor data
    print(f"\n{'='*70}")
    print("SENSOR FUSION TEST: 贡嘎雪山 + real altitude sensor")
    print(f"{'='*70}")
    result = fuse_elements(
        {
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
            "mountain_rock_type": "snow_peaks_glaciers",
            "elevation_estimate_m": 4500,
        },
        sensor_elevation_m=4556.0,  # Gongga summit
        sensor_temperature_c=-5.0,
    )
    print(f"  Result: ({result.latitude:.2f}, {result.longitude:.2f})")
    print(f"  Uncertainty: ±{result.uncertainty_km:.0f} km")
    print(f"  Confidence: {result.confidence:.1%}")

    # ── Climate fingerprint test: winter NE China vs summer Tibet ─────────
    print(f"\n{'='*70}")
    print("CLIMATE FINGERPRINT TEST: conifer forest + winter cold sensor")
    print(f"{'='*70}")
    # VLM sees: conifer forest, snow possible (ambiguous — could be NE China, Tianshan, Tibet)
    # Sensor: -18°C — only possible in NE China in deep winter, or very high Tibet
    # Combined with DEM: 500m elevation → can't be Tibet (too low), must be NE China
    result = fuse_elements(
        {
            "climate_zone": "temperate",
            "terrain_type": "rolling_hills",
            "vegetation_zone": "conifer_forest",
            "soil_color": "black",
        },
        sensor_elevation_m=500.0,     # Low elevation — rules out Tibet
        sensor_temperature_c=-18.0,   # Deep winter — NE China match
        sensor_humidity_pct=60.0,     # Moderate humidity — NE China winter
    )
    print(f"  Scenario: conifer forest + -18C + 500m + 60%RH (winter NE China)")
    print(f"  Result: ({result.latitude:.2f}, {result.longitude:.2f})")
    print(f"  Uncertainty: ±{result.uncertainty_km:.0f} km")
    print(f"  Confidence: {result.confidence:.1%}")
    if result.candidate_region:
        print(f"  Candidate regions:")
        for b in result.candidate_region.bboxes:
            print(f"    - {b.label}")

    print(f"\n{'='*70}")
    print("CLIMATE FINGERPRINT TEST: same scene + summer cool sensor")
    print(f"{'='*70}")
    # Same conifer forest, but: 12°C, 75% RH, 2500m → SW China mountains in summer
    result = fuse_elements(
        {
            "climate_zone": "temperate",
            "terrain_type": "rolling_hills",
            "vegetation_zone": "conifer_forest",
        },
        sensor_elevation_m=2500.0,
        sensor_temperature_c=12.0,
        sensor_humidity_pct=75.0,
    )
    print(f"  Scenario: conifer forest + 12C + 2500m + 75%RH (summer SW mountains)")
    print(f"  Result: ({result.latitude:.2f}, {result.longitude:.2f})")
    print(f"  Uncertainty: ±{result.uncertainty_km:.0f} km")
    print(f"  Confidence: {result.confidence:.1%}")
    if result.candidate_region:
        print(f"  Candidate regions:")
        for b in result.candidate_region.bboxes:
            print(f"    - {b.label}")

    print(f"\n{'='*70}")
    print("CLIMATE FINGERPRINT TEST: desert + hot dry sensor")
    print(f"{'='*70}")
    # Desert scene + 38°C and 15% RH → must be Taklamakan in summer, not Qaidam
    # (Qaidam at 2800m would be much cooler)
    result = fuse_elements(
        {
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
            "vegetation_zone": "desert_scrub",
            "scene_type": "desert",
        },
        sensor_elevation_m=800.0,
        sensor_temperature_c=38.0,
        sensor_humidity_pct=15.0,
    )
    print(f"  Scenario: desert + 38C + 800m + 15%RH (summer Taklamakan)")
    print(f"  Result: ({result.latitude:.2f}, {result.longitude:.2f})")
    print(f"  Uncertainty: ±{result.uncertainty_km:.0f} km")
    print(f"  Confidence: {result.confidence:.1%}")
    if result.candidate_region:
        print(f"  Candidate regions:")
        for b in result.candidate_region.bboxes:
            print(f"    - {b.label}")


def test_with_geocot(n_images: int = 5):
    """Run full pipeline on test images with GeoCoT."""
    print(f"\n{'='*70}")
    print(f"FULL GeoCoT + FUSION PIPELINE ({n_images} images)")
    print(f"{'='*70}")

    from PIL import Image
    from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl

    # Load model
    print("Loading Qwen3-VL-2B...")
    model, processor, model_fn = load_qwen2vl()
    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "src", "Geocot", "prompts")
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": prompts_dir})

    # Load test data
    with open("output/regression/test_images.json") as f:
        test_info = json.load(f)
    test_paths = list(test_info.keys())

    results = []
    for i, img_path in enumerate(test_paths[:n_images]):
        name = img_path.replace("\\", "/").split("/")[-1]
        true_lat, true_lng = test_info[img_path]
        print(f"\n[{i+1}/{n_images}] {name}")
        print(f"  Ground truth: ({true_lat:.4f}, {true_lng:.4f})")

        try:
            image = Image.open(img_path).convert("RGB")
            geocot_result = pipeline.run(image)

            # Extract structured outputs
            stage_outputs = geocot_result.stage_outputs
            macro_text = stage_outputs.get("macro", "")
            regional_text = stage_outputs.get("regional", "")
            local_text = stage_outputs.get("local", "")

            # Run fusion
            fusion_result = run_fusion_pipeline(
                macro_text, regional_text, local_text,
            )

            error_km = haversine_distance_km(
                torch.tensor(fusion_result.latitude), torch.tensor(fusion_result.longitude),
                torch.tensor(true_lat), torch.tensor(true_lng),
            ).item()

            print(f"  GeoCoT direct: ({geocot_result.final_prediction.latitude}, "
                  f"{geocot_result.final_prediction.longitude})")
            print(f"  Fusion result: ({fusion_result.latitude:.2f}, {fusion_result.longitude:.2f}) "
                  f"±{fusion_result.uncertainty_km:.0f}km")
            print(f"  Error: {error_km:.0f} km  Confidence: {fusion_result.confidence:.1%}")

            if fusion_result.candidate_region:
                print(f"  Active: {fusion_result.candidate_region.active_elements}")

            results.append({
                "name": name,
                "true_lat": true_lat, "true_lng": true_lng,
                "pred_lat": fusion_result.latitude, "pred_lng": fusion_result.longitude,
                "uncertainty_km": fusion_result.uncertainty_km,
                "error_km": error_km,
                "confidence": fusion_result.confidence,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        errors = [r["error_km"] for r in results]
        uncertainties = [r["uncertainty_km"] for r in results]
        print(f"\n{'='*70}")
        print(f"SUMMARY ({len(results)} images)")
        print(f"{'='*70}")
        print(f"  Mean error:  {np.mean(errors):.0f} km")
        print(f"  Median error: {np.median(errors):.0f} km")
        print(f"  Mean uncertainty: {np.mean(uncertainties):.0f} km")

        # Count: error < uncertainty (good calibration)
        calibrated = sum(1 for e, u in zip(errors, uncertainties) if e < u)
        print(f"  Within uncertainty: {calibrated}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Test geographic element fusion engine")
    parser.add_argument("--geocot", type=int, default=0,
                        help="Number of images to test with full GeoCoT pipeline")
    args = parser.parse_args()

    # Always run manual tests (fast, no VLM needed)
    manual_test_cases()

    # Optional: full GeoCoT pipeline
    if args.geocot > 0:
        test_with_geocot(args.geocot)


if __name__ == "__main__":
    main()
