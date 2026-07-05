"""Quantize an OpenVINO model using NNCF.

Usage:
    python scripts/quantize_model.py --model-dir models/qwen2-vl-7b-ov --output-dir models/qwen2-vl-7b-ov-int4
"""

import argparse
import os


def quantize_model(model_dir: str, output_dir: str, vision_precision: str = "int8", llm_precision: str = "int4"):
    """Quantize OpenVINO model with mixed precision.

    Args:
        model_dir: Path to OpenVINO IR model directory.
        output_dir: Path to save quantized model.
        vision_precision: Precision for vision encoder (int8 or fp16).
        llm_precision: Precision for language model (int4 or int8).
    """
    import nncf
    import openvino as ov

    os.makedirs(output_dir, exist_ok=True)
    core = ov.Core()

    print(f"Loading model from {model_dir}")
    # Load vision encoder
    vit_path = os.path.join(model_dir, "openvino_vision_encoder_model.xml")
    if os.path.exists(vit_path):
        print(f"Quantizing vision encoder to {vision_precision}")
        vit_model = core.read_model(vit_path)
        if vision_precision == "int8":
            vit_model = nncf.compress_weights(vit_model, mode=nncf.CompressWeightsMode.INT8)
        ov.save_model(vit_model, os.path.join(output_dir, "openvino_vision_encoder_model.xml"))
    else:
        print("Warning: Vision encoder not found, skipping")

    # Load language model
    llm_path = os.path.join(model_dir, "openvino_language_model.xml")
    if os.path.exists(llm_path):
        print(f"Quantizing language model to {llm_precision}")
        llm_model = core.read_model(llm_path)
        if llm_precision == "int4":
            llm_model = nncf.compress_weights(llm_model, mode=nncf.CompressWeightsMode.INT4_SYM)
        elif llm_precision == "int8":
            llm_model = nncf.compress_weights(llm_model, mode=nncf.CompressWeightsMode.INT8)
        ov.save_model(llm_model, os.path.join(output_dir, "openvino_language_model.xml"))
    else:
        print("Warning: Language model not found, skipping")

    # Copy other model files (embeddings, etc.)
    for fname in os.listdir(model_dir):
        if fname.endswith((".xml", ".bin")) and "vision_encoder" not in fname and "language_model" not in fname:
            src = os.path.join(model_dir, fname)
            dst = os.path.join(output_dir, fname)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

    print(f"Quantization complete. Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Quantize OpenVINO model with NNCF")
    parser.add_argument("--model-dir", type=str, required=True, help="Input OpenVINO model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--vision-precision", choices=["int8", "fp16"], default="int8")
    parser.add_argument("--llm-precision", choices=["int4", "int8"], default="int4")
    args = parser.parse_args()

    quantize_model(args.model_dir, args.output_dir, args.vision_precision, args.llm_precision)


if __name__ == "__main__":
    main()
