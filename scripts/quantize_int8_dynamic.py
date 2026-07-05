"""INT8 dynamic quantization for GeoVLM ONNX model.
Uses onnxruntime's built-in dynamic quantization (no neural-compressor needed).
"""
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

FP32_PATH = "D:/Geocomp/output/geovlm_deploy/geovlm_fp32.onnx"
INT8_PATH = "D:/Geocomp/output/geovlm_deploy/geovlm_int8.onnx"

print("INT8 Dynamic Quantization")
print(f"  Input:  {FP32_PATH}")
print(f"  Output: {INT8_PATH}")
print()

print("Quantizing (this may take a few minutes)...")
quantize_dynamic(
    model_input=FP32_PATH,
    model_output=INT8_PATH,
    weight_type=QuantType.QInt8,
    extra_options={
        "DefaultTensorType": onnx.TensorProto.FLOAT,
    },
)

fp32_mb = Path(FP32_PATH).stat().st_size / 1024**2
int8_mb = Path(INT8_PATH).stat().st_size / 1024**2

print(f"\nDone!")
print(f"  FP32: {fp32_mb:.0f} MB")
print(f"  INT8: {int8_mb:.0f} MB")
print(f"  Compression: {fp32_mb / int8_mb:.1f}x")
