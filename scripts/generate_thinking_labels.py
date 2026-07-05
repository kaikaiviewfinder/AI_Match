"""Phase 1B: 8B-Thinking offline labeling pipeline.

Runs Qwen3-VL-8B-Thinking over a stratified sample of Mapillary images
to produce high-quality 3-stage GeoCoT training labels for LoRA fine-tuning.

Usage:
  # Quick test (n=5, dry-run)
  python scripts/generate_thinking_labels.py --n 5 --dry-run

  # Full run (500 images, saves to JSONL)
  python scripts/generate_thinking_labels.py --n 500 --output data/vlm_finetune/thinking_3stage_train.jsonl

  # Resume from crash (auto-detects completed images)
  python scripts/generate_thinking_labels.py --n 500 --output data/vlm_finetune/thinking_3stage_train.jsonl
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Add project paths
_scripts_dir = Path(__file__).resolve().parent
_src_dir = _scripts_dir.parent / "src"
sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_scripts_dir))

from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl, create_qwen2vl_model_fn, resize_image_for_vlm
from regression.dem_lookup import get_dem
from regression.climate_lookup import ClimateValidator
from regression.element_fusion import (
    parse_geocot_json, extract_all_elements,
    run_fusion_pipeline_v3,
)
from regression.geo_kb import (
    BBox, CLIMATE_ZONES, TERRAIN_TYPES, VEGETATION_ZONES,
    URBANIZATION_LEVELS, PROVINCE_BBOXES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene type estimator (same logic as generate_geocot_labels.py)
# ═══════════════════════════════════════════════════════════════════════════════

_COASTLINE = [
    (18.2, 109.5), (19.0, 110.5), (20.0, 110.5), (21.0, 111.0),
    (21.5, 111.5), (22.0, 113.0), (22.5, 114.0), (23.0, 114.5),
    (23.5, 117.0), (24.0, 118.0), (24.5, 118.5), (25.0, 119.0),
    (26.0, 119.5), (27.0, 120.5), (27.5, 121.0), (28.0, 121.5),
    (29.0, 121.5), (30.0, 122.0), (31.0, 121.8), (31.5, 122.0),
    (32.0, 121.5), (33.0, 121.0), (34.0, 120.5), (35.0, 120.0),
    (36.0, 120.0), (37.0, 121.0), (38.0, 121.5), (39.0, 122.0),
    (39.5, 121.5), (40.0, 122.0), (40.5, 121.5), (41.0, 121.5),
]


def _distance_to_coast_km(lat: float, lng: float) -> float:
    min_dist = float("inf")
    for clat, clng in _COASTLINE:
        dlat = abs(lat - clat) * 111.0
        dlng = abs(lng - clng) * 102.0
        dist = math.sqrt(dlat ** 2 + dlng ** 2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def check_vlm_output_quality(macro_text: str, regional_text: str, local_text: str,
                              sensor_elevation_m: Optional[float] = None) -> tuple[bool, str]:
    """Check if VLM output is meaningful or just generic noise.

    Returns (is_good, reason).
    Flags as low-quality if:
      - Any stage output is empty or trivially short
      - Macro JSON can't be parsed
      - VLM explicitly says it can't see anything
      - Elevation estimate is wildly inconsistent with sensor
    """
    # Empty or minimal output (CoT outputs are 1000-5000 chars, even bare JSON is 100+)
    if len(macro_text.strip()) < 100:
        return False, f"macro output too short ({len(macro_text.strip())} chars)"
    if len(regional_text.strip()) < 100:
        return False, f"regional output too short ({len(regional_text.strip())} chars)"
    if len(local_text.strip()) < 80:
        return False, f"local output too short ({len(local_text.strip())} chars)"

    # Check for VLM refusal / image quality patterns
    # CRITICAL: These patterns must ONLY match when the IMAGE itself is bad
    # (blurry, wall-only, corrupted, empty), NEVER when VLM is doing normal
    # scene reasoning. Each pattern below has been vetted:
    #   "看不到山" = valid analysis → NOT flagged
    #   "远处有一堵墙" = scene description → NOT flagged (no bare "一堵墙")
    #   "图片只有一堵墙" = image quality problem → flagged
    #   "不太模糊" → NOT flagged (no bare "太模糊")
    refusal_patterns = [
        # Blurry image (each pattern verified safe from negation):
        #   "图片不模糊"  → 不 breaks "图片模糊" substring
        #   "图片不太模糊" → 不 breaks "图片太模糊" substring
        "完全模糊", "一片模糊", "图片模糊", "图像模糊",
        "图片太模糊", "图像太模糊", "照片太模糊",
        "completely blurry", "totally blurry",
        # Wall-only images (require "only" qualifier to avoid false positives)
        "只有一堵墙", "只有一面墙", "只有墙",
        "just a wall", "only a wall",
        # Solid color / blank images
        "纯色背景", "纯色图片",
        # Corrupted / unreadable images
        "图片损坏", "图像损坏", "无法加载图片",
        "image corrupted", "image is corrupted",
        # Empty scene (but NOT "远处什么都没有" which describes empty horizon)
        "图片什么都没有", "图像什么都没有", "图片中什么都没有",
        "no scene visible", "nothing visible in the image",
    ]
    # Hard refusal: VLM explicitly says it cannot process this
    hard_refusal_phrases = [
        "cannot analyze this image", "can't analyze this",
        "无法分析这张", "无法处理这张",
    ]
    combined = (macro_text + regional_text + local_text).lower()
    for pat in refusal_patterns:
        if pat in combined:
            return False, f"VLM refusal pattern: '{pat}'"
    for pat in hard_refusal_phrases:
        if pat in combined:
            return False, f"VLM hard refusal: '{pat}'"

    # Try parsing macro JSON
    try:
        macro = parse_geocot_json(macro_text)
        if not macro:
            return False, "macro JSON is empty"
        # If all values are defaults/fallbacks, VLM didn't engage
        non_default = sum(1 for v in macro.values() if v and v not in ([], "", None))
        if non_default == 0:
            return False, "macro JSON has no meaningful values"
    except Exception:
        return False, "macro JSON unparseable"

    # Elevation sanity check
    if sensor_elevation_m is not None:
        try:
            macro = parse_geocot_json(macro_text)
            elev_est = macro.get("elevation_estimate_m")
            if isinstance(elev_est, list) and len(elev_est) == 2:
                est_mid = (elev_est[0] + elev_est[1]) / 2
                ratio = max(sensor_elevation_m, est_mid) / max(min(sensor_elevation_m, est_mid), 1)
                if ratio > 10:
                    return False, f"elevation wildly off: sensor={sensor_elevation_m:.0f}m, vlm={est_mid:.0f}m (ratio={ratio:.1f}x)"
        except Exception:
            pass

    return True, "ok"


def _point_in_bbox(lat: float, lng: float, bbox: BBox) -> bool:
    return bbox.lat_min <= lat <= bbox.lat_max and bbox.lng_min <= lng <= bbox.lng_max


def _best_match(lat: float, lng: float, category_dict: dict) -> Optional[str]:
    candidates = []
    for value, bboxes in category_dict.items():
        for bbox in bboxes:
            if _point_in_bbox(lat, lng, bbox):
                candidates.append((bbox.area_km2, value))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def estimate_scene_type(lat: float, lng: float, dem=None) -> str:
    """Estimate scene type from coordinates using GeoKB + DEM.

    Uses terrain type and elevation as primary signals, urbanization as secondary.
    The URBANIZATION_LEVELS small_town/village/rural bboxes are too broad (cover
    all of eastern China), so we only use metropolis/medium_city from them.
    """
    terrain_type = _best_match(lat, lng, TERRAIN_TYPES)
    climate_zone = _best_match(lat, lng, CLIMATE_ZONES)

    # Elevation from DEM if available (strong signal for wilderness)
    elevation = None
    if dem is not None:
        elevation = dem.query(lat, lng)

    # ── Step 1: Distinctive terrain → "special" ──────────────────────────
    if terrain_type in ("desert_dunes",):
        return "desert"
    if terrain_type in ("karst_peaks", "sandstone_pillars"):
        return "mountain_trail_karst"
    if terrain_type == "sharp_mountains" and climate_zone == "alpine":
        return "mountain_trail_snow"

    # ── Step 2: High elevation → "wilderness" ────────────────────────────
    if elevation is not None and elevation > 2500:
        return "mountain_trail_snow" if climate_zone == "alpine" else "grassland"
    if elevation is not None and elevation > 1500 and terrain_type in ("sharp_mountains", "plateau"):
        return "mountain_trail_granite"

    # ── Step 3: Urbanization (metros + medium cities only) ───────────────
    for level in ("metropolis", "medium_city"):
        for bbox in URBANIZATION_LEVELS.get(level, []):
            if _point_in_bbox(lat, lng, bbox):
                return "city_street"

    # ── Step 4: Coastal → "town" ─────────────────────────────────────────
    if _distance_to_coast_km(lat, lng) < 50.0:
        return "coastal"

    # ── Step 5: Terrain-based wilderness ──────────────────────────────────
    if terrain_type == "grassland_steppe":
        return "grassland"
    if terrain_type == "sharp_mountains":
        return "mountain_trail_granite"
    if terrain_type == "plateau":
        return "grassland"
    if terrain_type == "rolling_hills" and elevation is not None and elevation > 1000:
        return "wilderness_forest"

    # ── Default: town/rural (the vast majority of inhabited China) ───────
    return "rural_village"


# ═══════════════════════════════════════════════════════════════════════════════
# Stratified sampling
# ═══════════════════════════════════════════════════════════════════════════════

STRATA_QUOTA = {
    "wilderness": 0.40,
    "town": 0.30,
    "city": 0.20,
    "special": 0.10,
}

SCENE_TO_STRATUM = {
    "wilderness_forest": "wilderness",
    "mountain_trail_granite": "wilderness",
    "mountain_trail_snow": "wilderness",
    "grassland": "wilderness",
    "desert": "special",
    "mountain_trail_karst": "special",
    "coastal": "town",
    "rural_village": "town",
    "city_street": "city",
}


def sample_stratified(geotagged: dict, n: int, seed: int = 42,
                      dem=None) -> list[tuple[str, float, float]]:
    """Sample n images with stratification by estimated scene type."""
    import random
    rng = random.Random(seed)

    # Classify all images
    pools = {k: [] for k in STRATA_QUOTA}
    for img_path, coords in geotagged.items():
        if isinstance(coords, list):
            lat, lng = float(coords[0]), float(coords[1])
        elif isinstance(coords, dict):
            lat, lng = float(coords.get("lat", 0)), float(coords.get("lng", 0))
        else:
            continue
        if not (17.0 <= lat <= 55.0 and 72.0 <= lng <= 136.0):
            continue
        scene = estimate_scene_type(lat, lng, dem=dem)
        stratum = SCENE_TO_STRATUM.get(scene, "wilderness")
        pools[stratum].append((img_path, lat, lng))

    # Sample from each pool
    selected = []
    for stratum, quota in STRATA_QUOTA.items():
        pool = pools[stratum]
        n_stratum = int(n * quota)
        if n_stratum > len(pool):
            n_stratum = len(pool)
        rng.shuffle(pool)
        selected.extend(pool[:n_stratum])

    # Top up with random sampling if quotas can't be met
    if len(selected) < n:
        all_images = []
        for pool in pools.values():
            all_images.extend(pool)
        rng.shuffle(all_images)
        existing = set(img for img, _, _ in selected)
        for img_path, lat, lng in all_images:
            if len(selected) >= n:
                break
            if img_path not in existing:
                selected.append((img_path, lat, lng))

    rng.shuffle(selected)
    return selected[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# Timeout-aware inference
# ═══════════════════════════════════════════════════════════════════════════════

class TimeoutError(Exception):
    pass


def _handler(signum, frame):
    raise TimeoutError("Inference timed out")


def run_with_timeout(fn, timeout_seconds: float = 1800.0):
    """Run fn() with a timeout. Only works on Unix; Windows uses polling fallback."""
    if hasattr(signal, 'SIGALRM'):
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout_seconds))
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        # Windows: can't use signal.SIGALRM
        return fn()


# ═══════════════════════════════════════════════════════════════════════════════
# Self-consistency check
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def self_consistency_check(
    macro_text: str, regional_text: str, local_text: str,
    true_lat: float, true_lng: float,
    sensor_elevation_m: Optional[float],
    sensor_temperature_c: Optional[float],
    sensor_humidity_pct: Optional[float],
    dem, climate,
    threshold_km: float = 500.0,
) -> tuple[float, bool, str]:
    """Run fusion engine on VLM outputs and check error vs ground truth.

    Returns (error_km, is_consistent, explanation).
    """
    try:
        result = run_fusion_pipeline_v3(
            macro_text=macro_text,
            regional_text=regional_text,
            local_text=local_text,
            sensor_elevation_m=sensor_elevation_m,
            sensor_temperature_c=sensor_temperature_c,
            sensor_humidity_pct=sensor_humidity_pct,
        )
        error_km = haversine_km(true_lat, true_lng, result.latitude, result.longitude)
        is_consistent = error_km <= threshold_km
        explanation = f"fusion_pred=({result.latitude:.2f},{result.longitude:.2f}) error={error_km:.0f}km"
        return error_km, is_consistent, explanation
    except Exception as e:
        return -1.0, False, f"fusion check failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-stage GeoCoT labels with Qwen3-VL-8B-Thinking")
    parser.add_argument("--geotagged", default="data/merged_geotagged.json",
                        help="Path to merged_geotagged.json")
    parser.add_argument("--output", default="data/vlm_finetune/thinking_3stage_train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of images to label")
    parser.add_argument("--model-name", default="D:/Geocomp/models/Qwen/Qwen3-VL-8B-Thinking",
                        help="Path to Thinking model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview sample distribution, don't run inference")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip self-consistency check (faster, for quick previews)")
    parser.add_argument("--filter-low-quality", action="store_true",
                        help="Detect and flag low-quality images (blurry, wall-only, etc.)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Timeout per image in seconds (default 3600 = 60min)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="max_new_tokens for VLM (default 8192 for full CoT)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index for multi-machine parallel runs (0-based)")
    parser.add_argument("--total-shards", type=int, default=1,
                        help="Total number of shards (machines)")
    parser.add_argument("--month", type=int, default=7,
                        help="Month for climate estimation (default July)")
    args = parser.parse_args()

    # ── Load geotagged data ──────────────────────────────────────────────────
    geotagged_path = Path(args.geotagged)
    if not geotagged_path.is_absolute():
        geotagged_path = _scripts_dir.parent / geotagged_path
    with open(geotagged_path, encoding="utf-8") as f:
        geotagged = json.load(f)
    print(f"Loaded {len(geotagged)} geotagged images from {geotagged_path.name}")

    # ── Load DEM (needed for sampling stratification) ──────────────────────
    dem = get_dem()

    # ── Sample + shard ──────────────────────────────────────────────────────
    sampled = sample_stratified(geotagged, args.n, seed=args.seed, dem=dem)
    # Multi-machine: each shard takes a disjoint slice
    if args.total_shards > 1:
        shard_size = len(sampled) // args.total_shards
        start = args.shard * shard_size
        end = start + shard_size if args.shard < args.total_shards - 1 else len(sampled)
        sampled = sampled[start:end]
        print(f"Sampled {len(sampled)} images (shard {args.shard}/{args.total_shards}, slice [{start}:{end}])")
    else:
        print(f"Sampled {len(sampled)} images")

    # Show stratum distribution
    counts = {}
    for _, lat, lng in sampled:
        scene = estimate_scene_type(lat, lng, dem=dem)
        stratum = SCENE_TO_STRATUM.get(scene, "wilderness")
        counts[stratum] = counts.get(stratum, 0) + 1
    print("Stratum distribution:")
    for s, c in sorted(counts.items(), key=lambda x: -x[1]):
        pct = c / len(sampled) * 100
        print(f"  {s}: {c} ({pct:.0f}%)")

    if args.dry_run:
        print("\n[Dry run] Sample preview:")
        for i, (img_path, lat, lng) in enumerate(sampled[:10]):
            scene = estimate_scene_type(lat, lng, dem=dem)
            print(f"  [{i+1}] {Path(img_path).name} ({lat:.3f}, {lng:.3f}) -> {scene}")
        print("  ...")
        return

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"\nLoading 8B-Thinking model with CoT: {args.model_name}")
    model, processor, _ = load_qwen2vl(
        model_name=args.model_name,
        load_in_4bit=True,
        max_new_tokens=args.max_tokens,
        temperature=0.6,
        top_p=0.8,
        enable_thinking=True,
    )
    model_fn = create_qwen2vl_model_fn(
        model, processor,
        max_new_tokens=args.max_tokens,
        temperature=0.6,
        top_p=0.8,
        enable_thinking=True,
    )
    pipeline = GeoCoTPipeline(model_fn, {
        'prompts_dir': str(_src_dir / 'Geocot' / 'prompts'),
    })
    climate = ClimateValidator()

    # ── Resume from checkpoint ───────────────────────────────────────────────
    output_path = Path(args.output)
    # Shard suffix for multi-machine runs
    if args.total_shards > 1:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.with_name(f"{stem}_shard{args.shard:02d}{suffix}")
    if not output_path.is_absolute():
        output_path = _scripts_dir.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record.get("image", ""))
                except json.JSONDecodeError:
                    pass
        print(f"Resume: {len(completed)} already completed")

    pending = [(p, lat, lng) for p, lat, lng in sampled if p not in completed]
    print(f"Pending: {len(pending)} images\n")

    if not pending:
        print("All images already processed. Done.")
        return

    # ── Process each image ───────────────────────────────────────────────────
    stats = {"success": 0, "timeout": 0, "error": 0, "low_quality": 0, "filtered": 0}
    f_out = open(output_path, "a", encoding="utf-8")
    f_lowq = None
    lowq_path = None
    if args.filter_low_quality:
        lowq_path = output_path.with_suffix(".low_quality.jsonl")
        f_lowq = open(lowq_path, "a", encoding="utf-8")
    t0_total = time.time()

    for i, (img_path, true_lat, true_lng) in enumerate(pending):
        name = Path(img_path).basename if hasattr(Path(img_path), 'basename') else os.path.basename(img_path)
        stats_line = f"[{i+1}/{len(pending)}] {name}"

        # Sensor data from ground truth
        sensor_elev = dem.query(true_lat, true_lng) if dem else None
        if sensor_elev is not None:
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, sensor_elev, args.month)
            sensor_temp = round((t_lo + t_hi) / 2, 1)
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, args.month)
            sensor_humid = round((h_lo + h_hi) / 2, 1)
        else:
            sensor_temp = None
            sensor_humid = None

        def do_inference():
            image = Image.open(img_path).convert("RGB")
            return pipeline.run(
                image,
                sensor_elevation_m=sensor_elev,
                sensor_temperature_c=sensor_temp,
                sensor_humidity_pct=sensor_humid,
            )

        t0 = time.time()
        try:
            geocot_result = run_with_timeout(do_inference, timeout_seconds=args.timeout)
            dt = time.time() - t0
        except TimeoutError:
            dt = time.time() - t0
            print(f"  {stats_line}  TIMEOUT ({dt:.0f}s) - skipping")
            stats["timeout"] += 1
            continue
        except Exception as e:
            dt = time.time() - t0
            print(f"  {stats_line}  ERROR ({dt:.0f}s): {e}")
            stats["error"] += 1
            continue

        macro_text = geocot_result.stage_outputs.get("macro", "")
        regional_text = geocot_result.stage_outputs.get("regional", "")
        local_text = geocot_result.stage_outputs.get("local", "")

        # VLM output quality check (blurry, wall-only, unparseable, etc.)
        quality_flag = "ok"
        quality_reason = ""
        if args.filter_low_quality:
            is_good, reason = check_vlm_output_quality(
                macro_text, regional_text, local_text, sensor_elev)
            if not is_good:
                quality_flag = "low_quality_vlm"
                quality_reason = reason
                stats["low_quality"] += 1

        # Self-consistency check
        if not args.no_check and quality_flag == "ok":
            error_km, is_consistent, check_msg = self_consistency_check(
                macro_text, regional_text, local_text,
                true_lat, true_lng,
                sensor_elev, sensor_temp, sensor_humid,
                dem, climate,
                threshold_km=500.0,
            )
            if not is_consistent:
                quality_flag = "low_quality_fusion"
                stats["low_quality"] += 1
            if quality_reason:
                quality_reason += "; " + check_msg
            else:
                quality_reason = check_msg
            print(f"  {stats_line}  {dt:.0f}s  quality: {quality_flag} ({quality_reason})")
        elif quality_flag == "low_quality_vlm":
            print(f"  {stats_line}  {dt:.0f}s  quality: {quality_flag} ({quality_reason})")
        else:
            print(f"  {stats_line}  {dt:.0f}s (no self-check)")

        # Extract province and scene type for metadata
        province_name = "Unknown"
        for province, bboxes in PROVINCE_BBOXES.items():
            for b in bboxes:
                if _point_in_bbox(true_lat, true_lng, b):
                    province_name = province
                    break
            if province_name != "Unknown":
                break

        scene_type = estimate_scene_type(true_lat, true_lng, dem=dem)

        # Parse GeoCoT output for compound scene detection
        compound_scene = None
        try:
            local_parsed = parse_geocot_json(local_text)
            distinctive = local_parsed.get("distinctive_features", [])
            compound_keywords = {
                "aotai_ridgeline", "karst_peaks_kingdom", "zhangjiajie_pillars",
                "guilin_karst", "tibetan_plateau", "altai_grassland",
                "shanghai_coastal", "guangzhou_pearl_river",
            }
            for feat in distinctive:
                if isinstance(feat, str) and feat in compound_keywords:
                    compound_scene = feat
                    break
        except Exception:
            pass

        # Build output record
        record = {
            "image": img_path,
            "lat": true_lat,
            "lng": true_lng,
            "elevation": sensor_elev,
            "compound_scene": compound_scene,
            "macro_output": macro_text,
            "regional_output": regional_text,
            "local_output": local_text,
            "province": province_name,
            "scene_type": scene_type,
            "format": "3stage",
            "quality": quality_flag,
            "quality_reason": quality_reason,
            "inference_time_s": round(dt, 1),
            "teacher_model": "Qwen3-VL-8B-Thinking",
        }
        if quality_flag in ("ok", "low_quality_fusion"):
            # Keep in main output: good data + VLM mistakes (still useful for training)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            stats["success"] += 1
        else:
            # low_quality_vlm: truly bad images (blurry, wall-only, etc.) → delete candidates
            f_lowq.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_lowq.flush()
            stats["filtered"] += 1

    f_out.close()
    if args.filter_low_quality:
        f_lowq.close()

    # ── Final report ─────────────────────────────────────────────────────────
    dt_total = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"Done in {dt_total:.0f}s ({dt_total/60:.1f} min)")
    print(f"  Success: {stats['success']}")
    if args.filter_low_quality:
        print(f"  Filtered (low quality): {stats['filtered']}")
    print(f"  Timeout: {stats['timeout']}")
    print(f"  Error:   {stats['error']}")
    print(f"Output: {output_path}")
    if args.filter_low_quality:
        print(f"Low-quality log: {lowq_path}")


if __name__ == "__main__":
    main()
