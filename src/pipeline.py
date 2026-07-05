"""End-to-end pipeline: input image -> GeoCoT reasoning -> prediction -> evaluation -> report.

Usage:
    python src/pipeline.py --image-dir data/panoramas --config dk2500
    python src/pipeline.py --image-dir data/panoramas --config default --eval
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.Geocot.Geocot import GeoCoTPipeline, GeoCoTResult, create_hf_model_fn
from src.Geocot.prediction_extractor import extract_prediction_enhanced, format_prediction
from src.utils.config import load_config, get_path


def run_inference(pipeline: GeoCoTPipeline, image_dir: str, output_dir: str):
    """Run GeoCoT inference on all images in a directory."""
    reasoning_dir = os.path.join(output_dir, "reasoning")
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(reasoning_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    images = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(root, f))

    print(f"Processing {len(images)} image(s)...")
    results = []
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  {name}...", end=" ", flush=True)

        image = Image.open(img_path).convert("RGB")
        result = pipeline.run(image)

        with open(os.path.join(reasoning_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(result.reasoning_chain)

        pred_str = format_prediction(result.final_prediction)
        with open(os.path.join(prediction_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(pred_str)

        results.append({"name": name, "prediction": pred_str})
        print(f"-> {pred_str}")

    return results


def run_evaluation(output_dir: str, ground_truth_dir: str):
    """Run evaluation metrics on GeoCoT output vs ground truth."""
    from src.Geoeval.GeoClassificationMetrics import evaluate_classification
    from src.Geoeval.GeoDistanceChecker import check_distances

    prediction_dir = os.path.join(output_dir, "predictions")

    print("\n=== Classification Metrics ===")
    try:
        cls_metrics = evaluate_classification(ground_truth_dir, prediction_dir)
        for level, metrics in cls_metrics.items():
            print(f"  {level}: accuracy={metrics.get('accuracy', 0):.3f}, "
                  f"F1={metrics.get('f1_macro', 0):.3f}")
    except Exception as e:
        print(f"  Classification eval failed: {e}")

    print("\n=== Distance Metrics ===")
    latlng_dir = os.path.join(output_dir, "latlng")
    gt_latlng_dir = os.path.join(ground_truth_dir, "latlng")
    if os.path.exists(latlng_dir) and os.path.exists(gt_latlng_dir):
        try:
            check_distances(gt_latlng_dir, latlng_dir)
        except Exception as e:
            print(f"  Distance eval failed: {e}")
    else:
        print("  Skipped (lat/lng files not available)")


def main():
    parser = argparse.ArgumentParser(description="TuXun end-to-end pipeline")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--output-dir", type=str, default="output/geocot")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after inference")
    parser.add_argument("--ground-truth-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load model
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from accelerate import Accelerator

    accelerator = Accelerator()
    device = accelerator.device
    model_name = cfg["model"]["name"]

    print(f"Loading model: {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
        device_map=device,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)

    model_fn = create_hf_model_fn(
        model, processor, device,
        temperature=cfg["inference"]["temperature"],
        top_p=cfg["inference"]["top_p"],
        max_new_tokens=cfg["model"]["max_output_tokens"],
    )

    prompt_config = {}
    prompts_dir = os.path.join(os.path.dirname(__file__), "Geocot", "prompts")
    if os.path.exists(prompts_dir):
        prompt_config["prompts_dir"] = prompts_dir

    pipeline = GeoCoTPipeline(model_fn, prompt_config)

    # Run inference
    results = run_inference(pipeline, args.image_dir, args.output_dir)

    # Run evaluation if requested
    if args.eval:
        gt_dir = args.ground_truth_dir or get_path(cfg, "paths", "ground_truth")
        run_evaluation(args.output_dir, gt_dir)

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(results), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
