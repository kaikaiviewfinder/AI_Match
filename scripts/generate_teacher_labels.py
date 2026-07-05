"""Generate teacher labels using Qwen3-VL-8B-Instruct for GeoVLM distillation.

Three-inference ensemble per image:
  - 3 independent runs (T=0.6, seeds 0/42/99)
  - Majority vote -> hard label
  - Averaged distribution -> soft label (for KL distillation)
  - Sensor consistency filter on elevation

Output: JSONL with per-field hard + soft labels.
"""
import sys, json, time, os, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl, create_qwen2vl_model_fn
from regression.element_fusion import parse_geocot_json, extract_all_elements
from regression.dem_lookup import get_dem
from regression.climate_lookup import get_climate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = str(PROJECT_ROOT / "src" / "Geocot" / "prompts")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", str(PROJECT_ROOT / "models" / "Qwen" / "Qwen3-VL-8B-Instruct"))
IMAGE_LIST = os.environ.get("IMAGE_LIST", str(PROJECT_ROOT / "data" / "mapillary_500" / "image_list.json"))
OUTPUT = os.environ.get("OUTPUT", str(PROJECT_ROOT / "output" / "geovlm_teacher_labels.jsonl"))
NUM_SAMPLES = 500  # target: 1000 total, start with 500
SEEDS = [0, 42, 99]


def run_single(img, pipeline, elev, temp, humid):
    """Run one VLM inference, return parsed elements dict or None."""
    result = pipeline.run(img,
        sensor_elevation_m=elev, sensor_temperature_c=temp, sensor_humidity_pct=humid)
    mj = parse_geocot_json(result.stage_outputs.get("macro", ""))
    rj = parse_geocot_json(result.stage_outputs.get("regional", ""))
    lj = parse_geocot_json(result.stage_outputs.get("local", ""))
    if all([mj, rj, lj]):
        return extract_all_elements(mj, rj, lj)
    return None


def ensemble_vote(elements_list: list[dict]) -> dict:
    """Majority voting across 3 runs per field. Returns hard + soft labels."""
    fields = ["climate_zone", "terrain_type", "vegetation_zone",
              "urbanization", "architecture_style", "pavement_type",
              "language_script", "visible_text"]
    hard_labels = {}
    soft_labels = {}
    vote_quality = {}

    for field in fields:
        values = [e.get(field, "UNKNOWN") for e in elements_list]
        counter = Counter(values)
        most_common, count = counter.most_common(1)[0]

        # Soft label: average distribution across runs
        all_values = set(v for e in elements_list for v in [e.get(field, "UNKNOWN")])
        dist = {v: sum(1 for e in elements_list if e.get(field, "UNKNOWN") == v) / len(elements_list)
                for v in all_values}
        soft_labels[field] = dist
        hard_labels[field] = most_common

        if count == len(elements_list):
            vote_quality[field] = "unanimous"
        elif count >= len(elements_list) // 2 + 1:
            vote_quality[field] = "majority"
        else:
            vote_quality[field] = "split"

    # Elevation: median of 3 estimates
    elev_ests = []
    for e in elements_list:
        val = e.get("elevation_estimate_m", None)
        if isinstance(val, list) and len(val) == 2:
            elev_ests.append((val[0] + val[1]) / 2)
        elif isinstance(val, (int, float)):
            elev_ests.append(val)
    if elev_ests:
        elev_ests.sort()
        hard_labels["elevation_estimate_m"] = elev_ests[len(elev_ests) // 2]
        soft_labels["elevation_estimate_m"] = sum(elev_ests) / len(elev_ests)

    return {
        "hard": hard_labels,
        "soft": soft_labels,
        "quality": vote_quality,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    # Load image list
    with open(IMAGE_LIST, encoding="utf-8") as f:
        base_images = json.load(f)

    # Resumable output
    completed = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["image"])
        print(f"Resume: {len(completed)} done")

    images = [r for r in base_images if r["image"] not in completed]
    limit = args.max_samples - len(completed)
    if limit <= 0:
        print("All done!")
        return
    images = images[:limit]
    print(f"Images to label: {len(images)}")

    # Load DEM + Climate
    dem = get_dem()
    climate = get_climate()

    # Load teacher VLM (4-bit quantized)
    print(f"\nLoading teacher: Qwen3-VL-8B-Instruct (4-bit)...")
    model, processor, _ = load_qwen2vl(
        model_name=TEACHER_MODEL, load_in_4bit=True,
        max_new_tokens=1024, temperature=0.6, enable_thinking=False)
    model_fn = create_qwen2vl_model_fn(model, processor,
        max_new_tokens=1024, temperature=0.6, enable_thinking=False)
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": PROMPTS_DIR})

    out_f = open(OUTPUT, "a", encoding="utf-8")
    month = 7
    total_time = 0.0
    t_all_start = time.time()
    COST_PER_HOUR = 3.6  # V100 machine-hours per wall-clock hour

    for i, sample in enumerate(images):
        name = sample["image"]
        img_path = sample.get("image_path", sample.get("path", ""))
        true_lat = sample.get("true_lat", sample.get("lat", 0))
        true_lng = sample.get("true_lng", sample.get("lng", 0))

        print(f"\n[{i+1}/{len(images)}] {name}", flush=True)
        t_start = time.time()

        # Sensor simulation
        elev = dem.query(true_lat, true_lng) if dem else 500
        temp, humid = None, None
        if climate:
            elev_c = elev if elev else 500
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, elev_c, month)
            temp = (t_lo + t_hi) / 2
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
            humid = (h_lo + h_hi) / 2

        img = Image.open(img_path).convert("RGB")

        # Three-inference ensemble with different seeds
        elements_list = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            elements = run_single(img, pipeline, elev, temp, humid)
            if elements:
                elements_list.append(elements)
            else:
                print(f"  WARN: seed={seed} produced no parseable output")

        if len(elements_list) < 2:
            print(f"  SKIP: <2 valid inferences")
            continue

        # Ensemble voting
        labels = ensemble_vote(elements_list)

        # Sensor consistency check on elevation
        if elev is not None and "elevation_estimate_m" in labels["hard"]:
            est = labels["hard"]["elevation_estimate_m"]
            if isinstance(est, (int, float)):
                deviation = abs(est - elev) / max(elev, 1)
                if deviation > 2.0:  # >2sigma from sensor
                    labels["hard"]["elevation_estimate_m"] = [elev * 0.85, elev * 1.15]
                    labels["quality"]["elevation_estimate_m"] = "sensor_override"
                    print(f"  Sensor override elevation: {est:.0f}m -> [{elev*0.85:.0f}, {elev*1.15:.0f}]")

        quality_counts = Counter(labels["quality"].values())
        dt = time.time() - t_start
        total_time += dt
        avg = total_time / (i + 1)
        remaining = len(images) - (i + 1)
        eta_wall = remaining * avg
        elapsed_mh = total_time / 3600 * COST_PER_HOUR
        total_mh_est = (total_time + remaining * avg) / 3600 * COST_PER_HOUR
        print(f"  {dt:.0f}s | avg={avg:.0f}s | ETA={eta_wall//60:.0f}m{eta_wall%60:.0f}s "
              f"| 机时:已{elapsed_mh:.1f}/估{total_mh_est:.1f}h | {dict(quality_counts)}",
              flush=True)

        # Write JSONL record
        record = {
            "image": name,
            "image_path": img_path,
            "true_lat": true_lat,
            "true_lng": true_lng,
            "sensor_elevation": elev,
            "sensor_temperature": temp,
            "sensor_humidity": humid,
            "hard_labels": labels["hard"],
            "soft_labels": labels["soft"],
            "vote_quality": labels["quality"],
            "n_valid": len(elements_list),
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

        # VRAM cleanup
        del img, elements_list
        torch.cuda.empty_cache()

    out_f.close()
    total_wall = time.time() - t_all_start
    total_mh = total_wall / 3600 * COST_PER_HOUR
    n_done = len(images)
    print(f"\n{'='*50}")
    print(f"Done: {n_done} images | 耗时 {total_wall//60:.0f}m{total_wall%60:.0f}s | 机时 {total_mh:.1f}h")
    print(f"Labels saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
