"""Prepare VLM fine-tuning data: generate (image, instruction, response) triples.

Strategy:
  1. Load geotagged data + scene labels
  2. Select ~125 images per scene type, prioritizing those near KB landmarks
  3. For each image, construct a detailed visual description using nearby KB entries
  4. Output as JSONL with format: {"image": path, "instruction": str, "response": str}

Usage:
    python scripts/prepare_vlm_data.py \
        --data data/merged_geotagged.json \
        --labels output/regression/scene_labels.json \
        --output data/vlm_finetune/train.jsonl \
        --num-per-type 125
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from Geocot.geoknowledge import GEO_KNOWLEDGE, PROVINCE_HINTS
from Geocot.Geocot import _CITY_COORD_DB, _init_city_db


def _haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ── Instructions per scene type ──────────────────────────────────────────

SCENE_INSTRUCTIONS = {
    "karst_granite": (
        "Describe this landscape image with geological, botanical, and cultural details "
        "that would help pinpoint its location in China. Be specific about:\n"
        "1. Rock type and geological features (granite, limestone, sandstone, etc.)\n"
        "2. Vegetation species and distinguishing characteristics\n"
        "3. Any visible cultural markers (architecture, text, infrastructure)\n"
        "4. Weather/climate indicators\n"
        "5. What distinguishes this place from visually similar locations in China"
    ),
    "alpine_snow": (
        "Describe this alpine/mountain image with terrain, vegetation, and cultural details "
        "that would help pinpoint its location in China. Be specific about:\n"
        "1. Mountain morphology (peak shapes, glacier features, valley types)\n"
        "2. Vegetation zones visible (alpine meadow, conifer forest, etc.)\n"
        "3. Snow/ice characteristics\n"
        "4. Any visible cultural elements (temples, prayer flags, trails)\n"
        "5. What distinguishes this mountain range from others in western China"
    ),
    "urban": (
        "Describe this urban street scene with architectural, infrastructural, and botanical "
        "details that would help pinpoint its location in China. Be specific about:\n"
        "1. Architectural style and era (modern, colonial, traditional)\n"
        "2. Street trees and vegetation species\n"
        "3. Visible signage, text, and language clues\n"
        "4. Infrastructure (road type, public transit, vehicles)\n"
        "5. What distinguishes this city from other Chinese cities"
    ),
    "other": (
        "Describe this image with geographical, botanical, and cultural details "
        "that would help pinpoint its location in China. Be specific about:\n"
        "1. Terrain and landscape type (canyon, old town, grassland, desert, lake)\n"
        "2. Vegetation and climate indicators\n"
        "3. Architecture and cultural elements\n"
        "4. Any distinctive regional features\n"
        "5. What distinguishes this place from visually similar locations"
    ),
}


def sample_images_by_scene(
    data: dict,
    scene_labels: dict,
    num_per_type: int = 125,
) -> dict[str, list[str]]:
    """Sample image paths for each scene type, prioritizing those near KB landmarks.

    Returns {scene_type_name: [image_paths]}
    """
    _init_city_db()

    # Group by scene type
    by_type = {"urban": [], "karst_granite": [], "alpine_snow": [], "other": []}
    type_names = {0: "urban", 1: "karst_granite", 2: "alpine_snow", 3: "other"}

    for img_path, st in scene_labels.items():
        name = type_names.get(int(st), "other")
        by_type[name].append(img_path)

    # Score images by proximity to KB landmarks
    def score_by_kb_proximity(img_path):
        if img_path not in data:
            return 0.0
        lat, lng = data[img_path][0], data[img_path][1]
        best_score = 0.0
        for name, entry in GEO_KNOWLEDGE.items():
            coords = entry.get("coordinates")
            if coords is None:
                continue
            d = _haversine_km(lat, lng, coords[0], coords[1])
            score = max(0, 1.0 - d / 500.0)  # 0-1, decay from 0 to 500km
            best_score = max(best_score, score)
        return best_score

    sampled = {}
    for scene_name, paths in by_type.items():
        if not paths:
            sampled[scene_name] = []
            continue

        # Score and sort
        scored = [(p, score_by_kb_proximity(p)) for p in paths]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-scoring + random mix
        n_top = int(num_per_type * 0.7)
        n_random = num_per_type - n_top
        top_paths = [p for p, _ in scored[:n_top]]
        remaining = [p for p, _ in scored[n_top:]]
        random.shuffle(remaining)
        sampled[scene_name] = top_paths + remaining[:n_random]

    return sampled


def build_response_from_kb(lat: float, lng: float) -> str:
    """Build a detailed Chinese response using nearby knowledge base entries.

    This creates a template-based response that teaches the model about
    fine-grained visual details specific to this region.
    """
    # Find closest KB entries
    matches = []
    for name, entry in GEO_KNOWLEDGE.items():
        coords = entry.get("coordinates")
        if coords is None:
            continue
        d = _haversine_km(lat, lng, coords[0], coords[1])
        if d < 500:
            matches.append((d, entry))

    matches.sort(key=lambda x: x[0])

    if not matches:
        return _generic_response(lat, lng)

    # Build response from closest match
    _, entry = matches[0]
    cn = entry.get("name_cn", "该地区")
    province = entry.get("province", "中国")
    geology = entry.get("geology", "")
    vegetation = entry.get("vegetation", "")
    cultural = entry.get("cultural_markers", "")
    distinguish = entry.get("distinguish_from", [])
    mistakes = entry.get("common_mistakes", [])

    lines = [f"综合视觉分析，该图拍摄于{province}{cn}附近。"]

    if geology:
        lines.append(f"地质特征: {geology}。")
    if vegetation:
        lines.append(f"植被特征: {vegetation}。")
    if cultural:
        lines.append(f"文化标志: {cultural}。")

    if distinguish:
        lines.append(f"与相似地点的关键区别: {'; '.join(distinguish)}。")
    if mistakes:
        lines.append(f"常见误判警示: {'; '.join(mistakes)}。")

    # Add a second nearby entry for contrast if available
    if len(matches) > 1:
        _, entry2 = matches[1]
        cn2 = entry2.get("name_cn", "")
        geology2 = entry2.get("geology", "")
        if cn2 and geology2:
            lines.append(f"注意区分附近的{cn2}：{geology2}，与本图有显著不同。")

    return "\n".join(lines)


def _generic_response(lat: float, lng: float) -> str:
    """Fallback generic response when no KB entry nearby."""
    _init_city_db()
    # Find nearest city
    best_city = "中国"
    best_dist = float("inf")
    for (city, country), coords in _CITY_COORD_DB.items():
        d = _haversine_km(lat, lng, coords[0], coords[1])
        if d < best_dist:
            best_dist = d
            best_city = city

    lines = [
        f"该图拍摄于{best_city}附近（约{best_dist:.0f}km）。",
        f"坐标约为({lat:.2f}°N, {lng:.2f}°E)。",
        "请结合图像中的植被、建筑风格、地形特征进行更精确的判断。",
    ]
    return "\n".join(lines)


def generate_dataset(
    data_file: str,
    labels_file: str,
    output_file: str,
    num_per_type: int = 125,
    base_dir: str = "",
):
    """Main entry: generate the VLM fine-tuning dataset."""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if labels_file and os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            scene_labels = json.load(f)
    else:
        # No labels yet — sample uniformly
        scene_labels = {}
        paths = list(data.keys())
        random.shuffle(paths)
        n = min(num_per_type * 4, len(paths))
        for i, p in enumerate(paths[:n]):
            scene_labels[p] = i % 4

    print(f"Data: {len(data)} images total")
    print(f"Labels: {len(scene_labels)} labeled")

    sampled = sample_images_by_scene(data, scene_labels, num_per_type)
    for name, paths in sampled.items():
        print(f"  {name}: {len(paths)} samples")

    # Generate instruction-response pairs
    scene_types = list(sampled.keys())
    records = []

    for scene_name in scene_types:
        instruction = SCENE_INSTRUCTIONS.get(scene_name, SCENE_INSTRUCTIONS["other"])
        for img_path in sampled[scene_name]:
            if img_path not in data:
                continue
            lat, lng = data[img_path][0], data[img_path][1]
            response = build_response_from_kb(lat, lng)

            full_path = os.path.join(base_dir, img_path) if base_dir else img_path
            records.append({
                "image": full_path,
                "instruction": instruction,
                "response": response,
                "scene_type": scene_name,
            })

    # Shuffle
    random.shuffle(records)

    # Save as JSONL
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} records to {output_file}")

    # Print stats
    sc_counts = {}
    for r in records:
        sc = r["scene_type"]
        sc_counts[sc] = sc_counts.get(sc, 0) + 1
    for sc, count in sorted(sc_counts.items()):
        print(f"  {sc}: {count}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare VLM fine-tuning data")
    parser.add_argument("--data", type=str, default="data/merged_geotagged.json")
    parser.add_argument("--labels", type=str, default="output/regression/scene_labels.json")
    parser.add_argument("--output", type=str, default="data/vlm_finetune/train.jsonl")
    parser.add_argument("--num-per-type", type=int, default=125)
    parser.add_argument("--base-dir", type=str, default="")
    args = parser.parse_args()

    generate_dataset(
        data_file=args.data,
        labels_file=args.labels,
        output_file=args.output,
        num_per_type=args.num_per_type,
        base_dir=args.base_dir,
    )


if __name__ == "__main__":
    main()
