"""Test v2.2 pipeline on ALL 500 distillation images with full sensor fusion + failure analysis."""
import sys, json, time, re, math, os
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl, create_qwen2vl_model_fn
from regression.element_fusion import fuse_elements_v3, extract_all_elements, parse_geocot_json, score_sensor_consistency, compute_element_agreement, select_best_elements
from regression.dem_lookup import get_dem
from regression.climate_lookup import get_climate

MODEL_PATH = "D:/Geocomp/models/Qwen3-VL-4B-Instruct/qwen/Qwen3-VL-4B-Instruct"
PROMPTS_DIR = str(Path(__file__).resolve().parent.parent / "src" / "Geocot" / "prompts")
OUTPUT = "D:/Geocomp/output/selected_500_test_results_v5.json"
OUTPUT_SUMMARY = "D:/Geocomp/output/selected_500_failure_analysis_v5.txt"


def haversine(lat1, lng1, lat2, lng2):
    if None in (lat1, lng1, lat2, lng2): return float("inf")
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_500_images():
    """Load all 500 images from the distillation selection."""
    jsonl = "D:/Geocomp/data/vlm_finetune/selected_500/selected_500.jsonl"
    images = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec["image"]
            if Path(p).exists():
                images.append((p, rec["lat"], rec["lng"]))
    return images


def run_one_inference(img, pipeline, sensor_elev, sensor_temp, sensor_humid):
    t0 = time.time()
    n_retries = 0
    max_retries = 1
    mj = rj = lj = None

    while True:
        result = pipeline.run(
            img,
            sensor_elevation_m=sensor_elev,
            sensor_temperature_c=sensor_temp,
            sensor_humidity_pct=sensor_humid,
        )
        mj = parse_geocot_json(result.stage_outputs.get("macro", ""))
        rj = parse_geocot_json(result.stage_outputs.get("regional", ""))
        lj = parse_geocot_json(result.stage_outputs.get("local", ""))

        if all([mj, rj, lj]) or n_retries >= max_retries:
            break
        n_retries += 1
        print(f"    JSON parse failed, retry {n_retries}/{max_retries}...", flush=True)

    elements = extract_all_elements(mj or {}, rj or {}, lj or {})
    dt = time.time() - t0
    return elements, dt, n_retries, all([mj, rj, lj]), {"macro": mj, "regional": rj, "local": lj}


def diagnose_failure(r):
    """Analyze root cause of a catastrophic failure (>200km error)."""
    reasons = []

    explanation = " ".join(r.get("explanation", []))
    elements = r.get("elements_summary", {})
    sensor_elev = r.get("sensor_elevation_m")
    vlm_elev = r.get("vlm_elevation_estimate")
    sensor_score = r.get("sensor_score", 1.0)
    n_elements = r.get("n_elements", 0)
    lat = r["true_lat"]
    lng = r["true_lng"]

    # 1. Check if sensor data was completely missing
    if sensor_elev is None:
        reasons.append("NO_SENSOR_ELEV: DEM lookup failed at true location")

    # 2. Check VLM elevation vs sensor
    if sensor_elev is not None and vlm_elev and isinstance(vlm_elev, list) and len(vlm_elev) >= 2:
        vlm_mid = (vlm_elev[0] + vlm_elev[1]) / 2
        elev_deviation = abs(vlm_mid - sensor_elev) / max(1, sensor_elev)
        if elev_deviation > 0.5:
            reasons.append(f"VLM_ELEV_MISMATCH: VLM={vlm_elev} vs sensor={sensor_elev}m (deviation={elev_deviation:.1%})")

    # 3. Low sensor consistency score
    if sensor_score < 0.70:
        reasons.append(f"SENSOR_INCONSISTENCY: score={sensor_score:.2f} — VLM elements conflict with physical sensors")

    # 4. Very few elements extracted
    if n_elements < 5:
        reasons.append(f"SPARSE_ELEMENTS: only {n_elements} elements extracted, insufficient spatial constraints")

    # 5. Check if fusion explanation contains "No search space" or similar
    if "No search space" in explanation or "0 regions" in explanation:
        reasons.append("EMPTY_SEARCH_SPACE: All candidate regions were filtered out")

    # 6. Geographic category
    if lat > 40:
        region = "far_north"
    elif lat > 35:
        region = "north_china"
    elif lat > 30:
        region = "central_china"
    elif lat > 25:
        region = "south_china"
    else:
        region = "far_south"

    # 7. Check GeoCoT raw output quality
    geocot = r.get("geocot_raw", {})
    macro = geocot.get("macro", {})
    regional = geocot.get("regional", {})
    local = geocot.get("local", {})

    # Check if VLM gave completely wrong macro-level info
    if macro:
        climate = macro.get("climate_zone", "")
        terrain = macro.get("terrain_type", "")
        veg = macro.get("vegetation_zone", "")
        urban = macro.get("urbanization_level", "")
        if not climate:
            reasons.append("MISSING_CLIMATE: VLM failed to identify climate zone")
        if not terrain:
            reasons.append("MISSING_TERRAIN: VLM failed to identify terrain type")

    return reasons


def analyze_all(results):
    """Generate comprehensive failure analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("  500-Image Distillation Pool — Catastrophic Failure Analysis")
    lines.append("  Pipeline: version2.2 (Adaptive N=2 + competition mirrors + sensor weighting)")
    lines.append("=" * 70)
    lines.append("")

    valid = [r for r in results if r["error_km"] < 9999]
    errors = sorted([r["error_km"] for r in valid])

    lines.append(f"Total tested: {len(results)}")
    lines.append(f"Valid results: {len(valid)}")
    lines.append(f"JSON parse failures: {sum(1 for r in results if not r['json_ok'])}")
    lines.append("")

    lines.append("── Error Distribution ──")
    lines.append(f"  Mean:   {sum(errors)/len(errors):.0f} km")
    lines.append(f"  Median: {errors[len(errors)//2]:.0f} km")
    lines.append(f"  Best:   {min(errors):.0f} km")
    lines.append(f"  Worst:  {max(errors):.0f} km")
    lines.append("")

    # Buckets
    buckets = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 99999)]
    for lo, hi in buckets:
        n = sum(1 for e in errors if lo <= e < hi)
        label = f"{lo}-{hi}km" if hi < 99999 else f">={lo}km"
        pct = f"({100*n/len(errors):.1f}%)" if len(errors) > 0 else ""
        lines.append(f"  {label:>12s}: {n:4d} {pct}")

    # Adaptive N=2 stats
    n2 = sum(1 for r in valid if r.get("n_samples", 1) > 1)
    lines.append(f"\n── Adaptive N=2 ──")
    lines.append(f"  2nd inference triggered: {n2}/{len(valid)} ({100*n2/len(valid):.1f}%)")
    if n2 > 0:
        avg_score_single = sum(r["sensor_score"] for r in valid if r.get("n_samples", 1) == 1) / max(1, len(valid) - n2)
        avg_score_double = sum(r["sensor_score"] for r in valid if r.get("n_samples", 1) > 1) / max(1, n2)
        lines.append(f"  Avg sensor score (single): {avg_score_single:.2f}")
        lines.append(f"  Avg sensor score (double): {avg_score_double:.2f}")

    lines.append(f"\n── JSON Reliability ──")
    lines.append(f"  All 3 stages OK: {sum(1 for r in results if r['json_ok'])}/{len(results)}")
    n_retried = sum(1 for r in results if r.get("n_retries", 0) > 0)
    lines.append(f"  Needed retry: {n_retried}/{len(results)}")

    # ── Catastrophic failures: error > 200km ──
    catastrophic = [r for r in valid if r["error_km"] > 200]
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  CATASTROPHIC FAILURES (>200km): {len(catastrophic)}/{len(valid)}")
    lines.append(f"{'=' * 70}")

    # Group by root cause
    cause_counts = Counter()
    failure_details = []
    for r in sorted(catastrophic, key=lambda x: -x["error_km"]):
        causes = diagnose_failure(r)
        for c in causes:
            cause_counts[c.split(":")[0]] += 1  # count by category prefix
        failure_details.append((r, causes))

    lines.append("\n── Root Cause Categories ──")
    for cause, count in cause_counts.most_common():
        lines.append(f"  {cause}: {count} occurrences")

    lines.append(f"\n── Top 20 Worst Failures ──")
    for r, causes in failure_details[:20]:
        lines.append(f"\n  {r['image']}")
        lines.append(f"    True: ({r['true_lat']:.2f}, {r['true_lng']:.2f})  Pred: ({r.get('pred_lat','?')}, {r.get('pred_lng','?')})  Error: {r['error_km']:.0f}km")
        lines.append(f"    Sensor elev: {r.get('sensor_elevation_m')}m  VLM elev: {r.get('vlm_elevation_estimate')}")
        lines.append(f"    Sensor score: {r.get('sensor_score',0):.2f}  N elements: {r.get('n_elements',0)}  N samples: {r.get('n_samples',0)}")
        if causes:
            lines.append(f"    Causes: {' | '.join(causes)}")
        else:
            lines.append(f"    Causes: [no specific cause identified — likely GeoKB coverage gap in this region]")

    # ── Good results: error ≤ 20km ──
    good = [r for r in valid if r["error_km"] <= 20]
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  GOOD RESULTS (≤20km): {len(good)}/{len(valid)}")
    lines.append(f"{'=' * 70}")
    for r in sorted(good, key=lambda x: x["error_km"])[:20]:
        lines.append(f"  {r['image']}: {r['error_km']:.0f}km  True=({r['true_lat']:.2f},{r['true_lng']:.2f})  Pred=({r.get('pred_lat','?')},{r.get('pred_lng','?')})")

    # ── Region breakdown ──
    lines.append(f"\n── By Latitude Band ──")
    bands = {">40N": [], "35-40N": [], "30-35N": [], "25-30N": [], "<25N": []}
    for r in valid:
        lat = r["true_lat"]
        if lat > 40: bands[">40N"].append(r)
        elif lat > 35: bands["35-40N"].append(r)
        elif lat > 30: bands["30-35N"].append(r)
        elif lat > 25: bands["25-30N"].append(r)
        else: bands["<25N"].append(r)

    for band, items in bands.items():
        if items:
            errs = [r["error_km"] for r in items]
            catastrophic_band = sum(1 for e in errs if e > 200)
            lines.append(f"  {band}: n={len(items)}  mean={sum(errs)/len(items):.0f}km  median={sorted(errs)[len(errs)//2]:.0f}km  catastrophic={catastrophic_band}")

    return "\n".join(lines)


def main():
    images = load_500_images()

    # ── Resume support: load already-completed image names ──
    completed_names = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            prev = json.load(f)
        completed_names = {r["image"] for r in prev}
        print(f"Resuming: {len(completed_names)} already completed, {len(images) - len(completed_names)} remaining")
    else:
        prev = []

    images = [(p, lat, lng) for p, lat, lng in images if Path(p).name not in completed_names]
    if not images:
        print("All images already completed!")
        # Generate analysis from existing results
        analysis = analyze_all(prev)
        with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
            f.write(analysis)
        print(analysis)
        return

    print(f"Remaining images: {len(images)}")
    for i, (p, lat, lng) in enumerate(images[:5]):
        print(f"  {i+1}. {Path(p).name} ({lat:.2f}, {lng:.2f})")
    if len(images) > 5:
        print(f"  ... ({len(images)} total)")

    # Load models
    print("\nLoading DEM + Climate...")
    dem = get_dem()
    climate = get_climate()
    print(f"  DEM: {'OK' if dem else 'MISSING'}, Climate: {'OK' if climate else 'MISSING'}")

    print("\nLoading VLM...")
    model, processor, _ = load_qwen2vl(
        model_name=MODEL_PATH, load_in_4bit=True,
        max_new_tokens=1024, temperature=0.6, enable_thinking=False)
    model_fn = create_qwen2vl_model_fn(model, processor, max_new_tokens=1024, temperature=0.6, enable_thinking=False)
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": PROMPTS_DIR})

    results = prev  # continue from saved
    total_time = sum(r.get("time_s", 0) for r in prev)
    month = 7  # summer default for offline images without timestamp
    total_done = len(prev)
    total_all = total_done + len(images)

    start_time = time.time()

    for i, (img_path, true_lat, true_lng) in enumerate(images):
        name = Path(img_path).name
        idx = total_done + i + 1
        print(f"\n[{idx:3d}/{total_all}] {name} ({true_lat:.2f}, {true_lng:.2f})", flush=True)

        # ── Sensor simulation ──
        sensor_elev = dem.query(true_lat, true_lng) if dem else None
        sensor_temp = None
        sensor_humid = None
        if climate:
            elev_for_climate = sensor_elev if sensor_elev else 500
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, elev_for_climate, month)
            sensor_temp = (t_lo + t_hi) / 2
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
            sensor_humid = (h_lo + h_hi) / 2

        sensor_str = f"elev={sensor_elev:.0f}m" if sensor_elev else "elev=N/A"
        if sensor_temp: sensor_str += f" T={sensor_temp:.0f}C"
        if sensor_humid: sensor_str += f" RH={sensor_humid:.0f}%"
        print(f"  Sensors: {sensor_str}", flush=True)

        # ── Adaptive N=2/3 inference with element-level stability check ──
        img = Image.open(img_path).convert("RGB")

        elements, dt, n_retries, all_ok, geocot_raw = run_one_inference(
            img, pipeline, sensor_elev, sensor_temp, sensor_humid)
        sensor_score = score_sensor_consistency(
            elements, sensor_elev, sensor_temp, sensor_humid)
        n_samples = 1
        agree_info = {}

        if sensor_score < 0.80:
            print(f"  Score={sensor_score:.2f}<0.80 -> 2nd inference...", flush=True)
            elements2, dt2, n_retries2, all_ok2, geocot_raw2 = run_one_inference(
                img, pipeline, sensor_elev, sensor_temp, sensor_humid)
            sensor_score2 = score_sensor_consistency(
                elements2, sensor_elev, sensor_temp, sensor_humid)
            agree_info = compute_element_agreement(elements, elements2)
            dt += dt2
            n_retries += n_retries2
            n_samples = 2

            if agree_info["disagree_count"] >= 2:
                print(f"  Elements disagree ({agree_info['disagree_count']}/6) -> 3rd inference...", flush=True)
                elements3, dt3, n_retries3, all_ok3, geocot_raw3 = run_one_inference(
                    img, pipeline, sensor_elev, sensor_temp, sensor_humid)
                sensor_score3 = score_sensor_consistency(
                    elements3, sensor_elev, sensor_temp, sensor_humid)
                dt += dt3
                n_retries += n_retries3
                n_samples = 3

                candidates = [
                    (elements, sensor_score, agree_info),
                    (elements2, sensor_score2, compute_element_agreement(elements2, elements)),
                    (elements3, sensor_score3, compute_element_agreement(elements3, elements)),
                ]
                elements, sensor_score = select_best_elements(candidates)
                # geocot_raw doesn't matter downstream for fusion
            else:
                if sensor_score2 > sensor_score:
                    elements = elements2
                    all_ok = all_ok2
                    sensor_score = sensor_score2
                    geocot_raw = geocot_raw2

        total_time += dt
        vlm_elev = elements.get("elevation_estimate_m", "N/A")

        json_status = "OK" if all_ok else "MISSING"
        if n_retries > 0:
            json_status += f" (retry={n_retries})"
        if n_samples > 1:
            json_status += f" N={n_samples}"

        # ── Fusion ──
        fusion = fuse_elements_v3(
            elements,
            sensor_elevation_m=sensor_elev,
            sensor_temperature_c=sensor_temp,
            sensor_humidity_pct=sensor_humid,
        )
        pred_lat = fusion.latitude
        pred_lng = fusion.longitude
        err = haversine(true_lat, true_lng, pred_lat, pred_lng) if pred_lat and pred_lng else 9999

        search_area_str = "N/A"
        for line in fusion.explanation_parts:
            if line.startswith("[SEARCH-AREA]"):
                search_area_str = line.split("Union:")[1].strip() if "Union:" in line else line
                break

        print(f"  {dt:.0f}s  VLM_elev={vlm_elev}  score={sensor_score:.2f}  JSON:{json_status}")
        safe_area = search_area_str.replace("\xb2", "^2").encode("ascii", errors="replace").decode("ascii")
        print(f"  Fusion:({pred_lat:.2f},{pred_lng:.2f}) err={err:.0f}km  {safe_area}", flush=True)

        # Calculate elapsed and ETA
        done_this_session = i + 1
        elapsed = time.time() - start_time
        avg_per = elapsed / done_this_session
        remaining = avg_per * (len(images) - done_this_session)
        eta_h = remaining // 3600
        eta_m = (remaining % 3600) // 60
        print(f"  [ETA: {eta_h:.0f}h{eta_m:.0f}m remaining]", flush=True)

        results.append({
            "image": name,
            "image_path": img_path,
            "true_lat": true_lat, "true_lng": true_lng,
            "sensor_elevation_m": sensor_elev,
            "sensor_temperature_c": sensor_temp,
            "sensor_humidity_pct": sensor_humid,
            "vlm_elevation_estimate": vlm_elev,
            "pred_lat": pred_lat, "pred_lng": pred_lng,
            "error_km": err,
            "uncertainty_km": fusion.uncertainty_km,
            "confidence": fusion.confidence,
            "time_s": dt,
            "json_ok": all_ok,
            "n_retries": n_retries,
            "n_samples": n_samples,
            "sensor_score": sensor_score,
            "n_elements": len(elements),
            "elements_summary": {k: str(v)[:100] for k, v in elements.items()},
            "geocot_raw": {k: v for k, v in geocot_raw.items() if v},
            "explanation": fusion.explanation_parts,
        })

        # Free VRAM after every image (prevents cumulative OOM)
        del img, elements
        if n_samples >= 2:
            del elements2
        torch.cuda.empty_cache()
        import gc; gc.collect()

        # Save incremental results every 10 images
        if done_this_session % 10 == 0:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            print(f"  [Checkpoint: {total_done + done_this_session}/{total_all}] VRAM={used:.1f}GB", flush=True)

    # ── Final save ──
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    total_all = len(results)
    errors = [r["error_km"] for r in results if r["error_km"] < 9999]
    errors.sort()
    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.0f}s ({total_time/total_all:.0f}s/image)" if total_all else "")
    print(f"Wall clock: {(time.time()-start_time):.0f}s")
    print(f"JSON valid: {sum(1 for r in results if r['json_ok'])}/{total_all}")
    print(f"Errors (km):")
    print(f"  Mean:  {sum(errors)/len(errors):.0f}")
    print(f"  Median:{errors[len(errors)//2]:.0f}")
    print(f"  Best:  {min(errors):.0f}")
    print(f"  Worst: {max(errors):.0f}")
    print(f"  <=20km:  {sum(1 for e in errors if e<=20)}/{len(errors)}")
    print(f"  <=50km:  {sum(1 for e in errors if e<=50)}/{len(errors)}")
    print(f"  <=100km: {sum(1 for e in errors if e<=100)}/{len(errors)}")
    print(f"  <=200km: {sum(1 for e in errors if e<=200)}/{len(errors)}")

    # ── Failure analysis ──
    analysis = analyze_all(results)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(analysis)
    print(f"\nFailure analysis saved to {OUTPUT_SUMMARY}")
    print(f"Results saved to {OUTPUT}")

    # ── Done ──
    print("\nTest complete.")


if __name__ == "__main__":
    main()
