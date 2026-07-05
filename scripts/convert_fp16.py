"""Convert FP32 ONNX model to FP16 for DK-2500 deployment.
Uses onnxruntime's float16 conversion (no neural-compressor needed).
"""
from pathlib import Path
import onnx
from onnxconverter_common import float16
import numpy as np

FP32_PATH = "D:/Geocomp/output/geovlm_deploy/geovlm_fp32.onnx"
FP16_PATH = "D:/Geocomp/output/geovlm_deploy/geovlm_fp16.onnx"

print("FP32 → FP16 Conversion")
print(f"  Input:  {FP32_PATH}")
print(f"  Output: {FP16_PATH}")
print()

print("Loading FP32 model...")
model = onnx.load_model(FP32_PATH)

print("Converting to FP16...")
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

print("Saving FP16 model...")
onnx.save_model(model_fp16, FP16_PATH)

fp32_mb = Path(FP32_PATH).stat().st_size / 1024**2
fp16_mb = Path(FP16_PATH).stat().st_size / 1024**2

print(f"\nDone!")
print(f"  FP32: {fp32_mb:.0f} MB")
print(f"  FP16: {fp16_mb:.0f} MB")
print(f"  Compression: {fp32_mb / fp16_mb:.1f}x")
