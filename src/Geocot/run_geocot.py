"""CLI entry point for running the GeoCoT pipeline.

Usage:
    python -m src.Geocot.run_geocot --image path/to/image.jpg
    python -m src.Geocot.run_geocot --image-dir path/to/images/ --output-dir output/
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image
from src.Geocot.Geocot import GeoCoTPipeline, load_qwen2vl
from src.Geocot.prediction_extractor import format_prediction


def main():
    parser = argparse.ArgumentParser(description="Run GeoCoT geolocation pipeline")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, help="Directory of images to process")
    parser.add_argument("--output-dir", type=str, default="output/geocot", help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir is required")

    # Load model
    print(f"Loading model: {args.model}")
    t0 = time.time()
    model, processor, model_fn = load_qwen2vl(args.model)
    print(f"Loaded in {time.time()-t0:.0f}s")

    # Build pipeline
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": prompts_dir})

    # Prepare output
    os.makedirs(os.path.join(args.output_dir, "reasoning"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)

    # Collect images
    images = []
    if args.image:
        images.append(args.image)
    elif args.image_dir:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(args.image_dir, f))

    # Process
    print(f"Processing {len(images)} image(s)...")
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n--- {name} ---")
        image = Image.open(img_path).convert("RGB")

        result = pipeline.run(image)

        # Print results (full output, no truncation)
        for stage_key, output in result.stage_outputs.items():
            print(f"\n  [{stage_key.upper()}]")
            print(f"  {output}")
        print(f"\n  PREDICTION: {format_prediction(result.final_prediction)}")

        # Save
        with open(os.path.join(args.output_dir, "reasoning", f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(result.reasoning_chain)
        with open(os.path.join(args.output_dir, "predictions", f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(format_prediction(result.final_prediction))

    print("\nDone.")


if __name__ == "__main__":
    main()
