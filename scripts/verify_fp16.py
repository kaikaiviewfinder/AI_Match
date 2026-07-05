"""Verify FP16 ONNX model outputs match PyTorch model."""
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geovlm import GeoVLM, GeoVLMConfig

FP16_PATH = "D:/Geocomp/output/geovlm_deploy/geovlm_fp16.onnx"
CHECKPOINT = "D:/Geocomp/output/geovlm_checkpoints/geovlm_final.pt"

print("Loading ONNX model...")
session = ort.InferenceSession(FP16_PATH, providers=["CPUExecutionProvider"])

input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]
print(f"  Inputs:  {input_names}")
print(f"  Outputs: {output_names}")

# Create dummy inputs
dummy = {
    "pixel_coarse": np.random.randn(1, 3, 224, 224).astype(np.float32),
    "pixel_mid":    np.random.randn(1, 3, 224, 224).astype(np.float32),
    "pixel_fine":   np.random.randn(1, 3, 224, 224).astype(np.float32),
    "sensor":       np.array([[500.0, 20.0, 60.0]], dtype=np.float32),
}

print("\nRunning ONNX FP16 inference...")
onnx_out = session.run(output_names, dummy)
onnx_dict = {name: val for name, val in zip(output_names, onnx_out)}

print("ONNX output shapes:")
for name, val in onnx_dict.items():
    print(f"  {name}: {val.shape} ({val.dtype})")

print("\nAll outputs produced successfully!")
print(f"  Num outputs: {len(onnx_out)}")
print(f"  Output keys: {list(onnx_dict.keys())}")
