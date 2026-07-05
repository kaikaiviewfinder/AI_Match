"""Export HuggingFace VLM to OpenVINO IR format.

Usage:
    python scripts/export_openvino.py --model qwen2-vl-7b --output-dir models/qwen2-vl-7b-ov
    python scripts/export_openvino.py --model llama-3.2-11b-vision --output-dir models/llama-3.2-11b-ov
"""

import argparse
import os

SUPPORTED_MODELS = {
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "llama-3.2-11b-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


def export_model(model_name: str, output_dir: str, token: str = None):
    """Export a HuggingFace VLM to OpenVINO IR format."""
    from optimum.intel import OVModelForVisualCausalLM

    print(f"Exporting {model_name} to OpenVINO IR at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model = OVModelForVisualCausalLM.from_pretrained(
        model_name,
        export=True,
        compile=False,
        token=token,
        local_files_only=True,
    )
    model.save_pretrained(output_dir)
    print(f"Export complete. Files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export VLM to OpenVINO IR")
    parser.add_argument(
        "model",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to export",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()

    model_id = SUPPORTED_MODELS[args.model]
    token = args.token or os.getenv("HUGGINGFACE_TOKEN")
    export_model(model_id, args.output_dir, token)


if __name__ == "__main__":
    main()
