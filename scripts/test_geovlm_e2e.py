"""End-to-end test: GeoVLM vs 4B-Instruct baseline on 20 stratified images.

Compares:
  - GeoVLM (430M params, <1s/image)
  - 4B-Instruct + GeoCoT 3-stage (4B params, ~76s/image GPU / ~200s CPU)

Metrics: per-field accuracy, fusion error (km), inference time.
"""
import sys, json, time, math, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from Geocot.Geocot import load_geovlm, GeoVLMPipeline, GeoCoTPipeline, \
    load_qwen2vl, create_qwen2vl_model_fn
from regression.element_fusion import fuse_elements_v3
from regression.dem_lookup import get_dem
from regression.climate_lookup import get_climate

GEOVLM_CHECKPOINT = "D:/Geocomp/output/geovlm_checkpoints/geovlm_final.pt"
BASELINE_MODEL = "D:/Geocomp/models/Qwen/Qwen3-VL-4B-Instruct"
PROMPTS_DIR = str(Path(__file__).resolve().parent.parent / "src" / "Geocot" / "prompts")
SELECTED = "D:/Geocomp/output/test_50_selected.json"
OUTPUT = "D:/Geocomp/output/geovlm_e2e_results.json"


def haversine(lat1, lng1, lat2, lng2):
    if None in (lat1, lng1, lat2, lng2):
        return float("inf")
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def run_baseline_inference(img, pipeline, elev, temp, humid):
    """Run 4B-Instruct 3-stage GeoCoT -> elements dict."""
    from regression.element_fusion import parse_geocot_json, extract_all_elements
    result = pipeline.run(img, sensor_elevation_m=elev,
                          sensor_temperature_c=temp, sensor_humidity_pct=humid)
    mj = parse_geocot_json(result.stage_outputs.get("macro", ""))
    rj = parse_geocot_json(result.stage_outputs.get("regional", ""))
    lj = parse_geocot_json(result.stage_outputs.get("local", ""))
    return extract_all_elements(mj or {}, rj or {}, lj or {})


def main():
    with open(SELECTED, encoding="utf-8") as f:
        images = json.load(f)

    test_images = images[:20]
    print(f"E2E test on {len(test_images)} images")
    print(f"  GeoVLM checkpoint: {GEOVLM_CHECKPOINT}")
    print(f"  Baseline: Qwen3-VL-4B-Instruct")

    # Load DEM + Climate
    dem = get_dem()
    climate = get_climate()

    # ---- GeoVLM path ----
    print("\nLoading GeoVLM...")
    use_geovlm = os.path.exists(GEOVLM_CHECKPOINT)
    if use_geovlm:
        geovlm_model, geovlm_fn = load_geovlm(checkpoint_path=GEOVLM_CHECKPOINT, device="cpu")
        geovlm_pipe = GeoVLMPipeline(geovlm_fn)
        counts = geovlm_model.count_parameters()
        print(f"  Params: {counts['total']/1e6:.1f}M (ViT={counts['vision_encoder']/1e6:.1f}M "
              f"QFormer={counts['qformer']/1e6:.1f}M)")
    else:
        print(f"  [SKIP] Checkpoint not found. GeoVLM not tested.")
        geovlm_pipe = None

    # ---- 4B-Instruct baseline path ----
    print("\nLoading 4B-Instruct baseline...")
    use_baseline = os.path.exists(BASELINE_MODEL.replace("/", "/"))  # local path
    if use_baseline:
        try:
            bl_model, bl_proc, _ = load_qwen2vl(
                model_name=BASELINE_MODEL, load_in_4bit=True,
                max_new_tokens=1024, temperature=0.6, enable_thinking=False)
            bl_fn = create_qwen2vl_model_fn(bl_model, bl_proc, max_new_tokens=1024,
                                             temperature=0.6, enable_thinking=False)
            bl_pipe = GeoCoTPipeline(bl_fn, {"prompts_dir": PROMPTS_DIR})
            print("  4B-Instruct loaded (4-bit)")
        except Exception as e:
            print(f"  [SKIP] Cannot load baseline: {e}")
            use_baseline = False
            bl_pipe = None
    else:
        print(f"  [SKIP] Model not found at {BASELINE_MODEL}")
        bl_pipe = None

    month = 7
    results = []

    for i, sample in enumerate(test_images):
        name = sample["name"]
        true_lat = sample["lat"]
        true_lng = sample["lng"]
        img_path = sample["path"]
        group = sample.get("group", "unknown")

        img = Image.open(img_path).convert("RGB")

        # Sensor simulation
        elev = dem.query(true_lat, true_lng) if dem else 500
        temp, humid = None, None
        if climate:
            ec = elev if elev else 500
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, ec, month)
            temp = (t_lo + t_hi) / 2
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
            humid = (h_lo + h_hi) / 2

        print(f"\n[{i+1}/{len(test_images)}] {name} ({true_lat:.2f},{true_lng:.2f}) [{group}]")
        row = {"image": name, "true_lat": true_lat, "true_lng": true_lng, "group": group}

        # ---- GeoVLM ----
        if geovlm_pipe:
            t0 = time.time()
            try:
                elements = geovlm_pipe.extract(img, elev, temp, humid)
                fusion = fuse_elements_v3(elements,
                    sensor_elevation_m=elev, sensor_temperature_c=temp,
                    sensor_humidity_pct=humid)
                dt = time.time() - t0
                err = haversine(true_lat, true_lng, fusion.latitude, fusion.longitude)
                row["geovlm"] = {
                    "time_s": round(dt, 2),
                    "error_km": round(err, 1),
                    "pred_lat": fusion.latitude,
                    "pred_lng": fusion.longitude,
                    "confidence": fusion.confidence,
                    "climate": elements.get("climate_zone_pred", "N/A"),
                    "terrain": elements.get("terrain_type_pred", "N/A"),
                    "elevation": elements.get("elevation_estimate_m", "N/A"),
                    "urbanization": elements.get("urbanization_pred", "N/A"),
                }
                print(f"  GeoVLM: {dt:.1f}s err={err:.0f}km "
                      f"climate={elements.get('climate_zone_pred','?')} "
                      f"elev={elements.get('elevation_estimate_m','?')}")
            except Exception as e:
                import traceback
                print(f"  GeoVLM: ERROR — {e}")
                traceback.print_exc()
                row["geovlm"] = {"error": str(e)}

        # ---- 4B-Instruct baseline ----
        if bl_pipe:
            t0 = time.time()
            try:
                elements_bl = run_baseline_inference(img, bl_pipe, elev, temp, humid)
                fusion_bl = fuse_elements_v3(elements_bl,
                    sensor_elevation_m=elev, sensor_temperature_c=temp,
                    sensor_humidity_pct=humid)
                dt_bl = time.time() - t0
                err_bl = haversine(true_lat, true_lng, fusion_bl.latitude, fusion_bl.longitude)
                row["4b_instruct"] = {
                    "time_s": round(dt_bl, 2),
                    "error_km": round(err_bl, 1),
                    "pred_lat": fusion_bl.latitude,
                    "pred_lng": fusion_bl.longitude,
                    "confidence": fusion_bl.confidence,
                    "climate": elements_bl.get("climate_zone_pred", "N/A"),
                    "terrain": elements_bl.get("terrain_type_pred", "N/A"),
                    "elevation": elements_bl.get("elevation_estimate_m", "N/A"),
                    "urbanization": elements_bl.get("urbanization_pred", "N/A"),
                }
                print(f"  4B:      {dt_bl:.1f}s err={err_bl:.0f}km "
                      f"climate={elements_bl.get('climate_zone_pred','?')}")
            except Exception as e:
                import traceback
                print(f"  4B: ERROR — {e}")
                traceback.print_exc()
                row["4b_instruct"] = {"error": str(e)}

        results.append(row)
        torch.cuda.empty_cache()

    # ---- Summary ----
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    geovlm_errors = [r["geovlm"]["error_km"] for r in results
                     if "geovlm" in r and "error_km" in r["geovlm"]]
    bl_errors = [r["4b_instruct"]["error_km"] for r in results
                 if "4b_instruct" in r and "error_km" in r["4b_instruct"]]
    geovlm_times = [r["geovlm"]["time_s"] for r in results
                    if "geovlm" in r and "time_s" in r["geovlm"]]
    bl_times = [r["4b_instruct"]["time_s"] for r in results
                if "4b_instruct" in r and "time_s" in r["4b_instruct"]]

    print(f"\n{'='*60}")
    print(f"GeoVLM vs 4B-Instruct E2E Results")
    print(f"{'='*60}")

    if geovlm_errors:
        geovlm_errors.sort()
        geovlm_times.sort()
        print(f"\nGeoVLM (n={len(geovlm_errors)}):")
        print(f"  Error:  mean={sum(geovlm_errors)/len(geovlm_errors):.0f}km "
              f"median={geovlm_errors[len(geovlm_errors)//2]:.0f}km "
              f"best={min(geovlm_errors):.0f}km")
        print(f"  Time:   mean={sum(geovlm_times)/len(geovlm_times):.1f}s "
              f"median={geovlm_times[len(geovlm_times)//2]:.1f}s")

    if bl_errors:
        bl_errors.sort()
        bl_times.sort()
        print(f"\n4B-Instruct (n={len(bl_errors)}):")
        print(f"  Error:  mean={sum(bl_errors)/len(bl_errors):.0f}km "
              f"median={bl_errors[len(bl_errors)//2]:.0f}km "
              f"best={min(bl_errors):.0f}km")
        print(f"  Time:   mean={sum(bl_times)/len(bl_times):.1f}s "
              f"median={bl_times[len(bl_times)//2]:.1f}s")

    if geovlm_errors and bl_errors:
        delta = sum(geovlm_errors)/len(geovlm_errors) - sum(bl_errors)/len(bl_errors)
        speedup = sum(bl_times)/len(bl_times) / (sum(geovlm_times)/len(geovlm_times)) if geovlm_times and bl_times else 0
        print(f"\nDelta:  Error {delta:+.0f}km, Speedup {speedup:.1f}x")

    # Per-group breakdown
    if geovlm_errors:
        print(f"\nGeoVLM by group:")
        for g in ["wilderness_rural", "small_town_village", "medium_city", "metropolis"]:
            grp = [r for r in results if r.get("group") == g and "geovlm" in r and "error_km" in r["geovlm"]]
            if grp:
                ge = [r["geovlm"]["error_km"] for r in grp]
                print(f"  {g} (n={len(grp)}): mean={sum(ge)/len(ge):.0f}km median={sorted(ge)[len(ge)//2]:.0f}km")

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
