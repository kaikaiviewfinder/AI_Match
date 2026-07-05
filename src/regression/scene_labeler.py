"""Scene auto-labeler: assign SceneType labels to geotagged images.

Three-tier strategy:
  Tier 1 (proximity): coordinates within radius of known landmark → auto-label (~85%)
  Tier 2 (VLM): Qwen2-VL visual classification for images not covered by Tier 1 (~10%)
  Tier 3 (manual): spot-check and correction (~5%)

Output: scene_labels.json mapping image_path → scene_type (int 0-3)
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch

from .moe_config import (
    SceneType, SCENE_LANDMARKS, SCENE_TYPE_NAMES,
    URBAN_PROXIMITY_KM, PROXIMITY_CONFIDENCE_THRESHOLD,
)

# Reuse Geocot city DB for urban proximity labeling
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import Geocot.Geocot as _geocot_mod


def _ensure_db():
    """Ensure city DB is initialized (lazy-init in Geocot.py)."""
    _geocot_mod._init_city_db()


def _haversine_km(lat1, lng1, lat2, lng2):
    """Haversine distance in km."""
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def label_by_proximity(lat: float, lng: float) -> tuple[Optional[int], Optional[str], float]:
    """Assign scene type based on proximity to known landmarks.

    Returns:
        (scene_type, landmark_name, confidence) or (None, None, 0.0) if no match.
        Urban check always runs first (largest class, most prone to false positives).
    """
    # Priority 1: Check non-urban scene landmarks first (more specific than city proximity)
    best_type = None
    best_name = None
    best_dist = float("inf")

    for scene_type, landmarks in SCENE_LANDMARKS.items():
        if scene_type == SceneType.URBAN:
            continue
        for lm in landmarks:
            d = _haversine_km(lat, lng, lm[0], lm[1])
            radius = lm[2] * 111.0  # degrees → km (approximate at ~30°N)
            if d <= radius and d < best_dist:
                best_dist = d
                best_type = int(scene_type)
                best_name = f"landmark_r{lm[2]}"

    if best_type is not None:
        conf = max(0.5, 1.0 - best_dist / 100.0)
        return best_type, best_name, conf

    # Priority 2: Urban proximity check (only if no landmark matched)
    _ensure_db()
    min_city_dist = float("inf")
    for name, coords in _geocot_mod._CITY_COORD_DB.items():
        d = _haversine_km(lat, lng, coords[0], coords[1])
        if d < min_city_dist:
            min_city_dist = d

    if min_city_dist <= URBAN_PROXIMITY_KM:
        conf = 1.0 - (1.0 - PROXIMITY_CONFIDENCE_THRESHOLD) * (min_city_dist / URBAN_PROXIMITY_KM)
        return int(SceneType.URBAN), f"city_{min_city_dist:.0f}km", conf

    return None, None, 0.0


def label_dataset(
    data_file: str,
    output_file: Optional[str] = None,
    base_dir: str = "",
) -> dict[str, int]:
    """Label all images in a geotagged JSON file.

    Args:
        data_file: JSON mapping image_path → [lat, lng]
        output_file: optional path to save scene_labels.json
        base_dir: prepended to image paths if they're relative

    Returns:
        dict: {image_path: scene_type (0-3)}
    """
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {}
    stats = {0: 0, 1: 0, 2: 0, 3: 0, "unlabeled": 0}

    for path, coords in data.items():
        lat, lng = coords[0], coords[1]
        scene_type, _, conf = label_by_proximity(lat, lng)

        if scene_type is not None:
            labels[path] = int(scene_type)
            stats[scene_type] += 1
        else:
            stats["unlabeled"] += 1

    # Print summary
    total = len(data)
    print(f"Dataset: {total} images")
    print(f"  Tier 1 (proximity) labeled: {total - stats['unlabeled']} "
          f"({100 * (total - stats['unlabeled']) / total:.1f}%)")
    for st in range(4):
        name = SCENE_TYPE_NAMES.get(st, "unknown")
        print(f"  Expert {st} ({name}): {stats[st]} "
              f"({100 * stats[st] / total:.1f}%)")
    print(f"  Unlabeled: {stats['unlabeled']} "
          f"({100 * stats['unlabeled'] / total:.1f}%)")

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"Saved labels to {output_file}")

    return labels


def label_by_vlm(
    image_paths: list[str],
    model,
    processor,
    model_fn,
    batch_size: int = 4,
) -> dict[str, int]:
    """Tier 2: use Qwen2-VL to classify unlabeled images into scene types.

    Sends a simple 4-way classification prompt to the VLM.
    """
    prompt = (
        "Classify this image into exactly one of these scene types:\n"
        "0: urban (city streets, buildings, roads in towns/cities)\n"
        "1: karst_granite (limestone karst towers, granite peaks, sandstone pillars)\n"
        "2: alpine_snow (snow-capped mountains, high altitude, glaciers, alpine meadows)\n"
        "3: other (canyons, gorges, old towns, grasslands, deserts, lakes)\n\n"
        "Reply with ONLY the number (0, 1, 2, or 3)."
    )

    labels = {}
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        for path in batch:
            try:
                from PIL import Image
                img = Image.open(path).convert("RGB")
                response = model_fn(prompt, img, max_tokens=5)
                # Parse single digit
                for ch in response.strip():
                    if ch in "0123":
                        labels[path] = int(ch)
                        break
                else:
                    labels[path] = 3  # default to OTHER
            except Exception as e:
                print(f"  VLM error for {path}: {e}")
                labels[path] = 3
        if (i // batch_size + 1) % 20 == 0:
            print(f"  VLM labeling: {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    return labels


def label_dataset_full(
    data_file: str,
    output_file: str = "output/regression/scene_labels.json",
    base_dir: str = "",
    model=None,
    processor=None,
    model_fn=None,
) -> dict[str, int]:
    """Full labeling pipeline: Tier 1 (proximity) + Tier 2 (VLM for remainder).

    Returns complete labels dict.
    """
    # Tier 1
    labels = label_dataset(data_file, output_file=None, base_dir=base_dir)

    # Find unlabeled
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    unlabeled = [p for p in data if p not in labels]

    if unlabeled and model is not None:
        print(f"\nTier 2 (VLM): labeling {len(unlabeled)} remaining images...")
        vlm_labels = label_by_vlm(unlabeled, model, processor, model_fn)
        labels.update(vlm_labels)

        # Re-count
        stats = {0: 0, 1: 0, 2: 0, 3: 0, "unlabeled": 0}
        for p in data:
            if p in labels:
                stats[labels[p]] += 1
            else:
                stats["unlabeled"] += 1
        print(f"  After Tier 2: {stats['unlabeled']} still unlabeled")

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")

    return labels
