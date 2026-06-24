from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantType
import os

ONNX_MODEL = "outputs/onnx/vit5_encoder.onnx"

OUTPUT_MODEL = "outputs/onnx/headline_generator_int8.onnx"

os.makedirs("outputs/onnx", exist_ok=True)

print("=" * 60)
print("ĐANG QUANTIZATION...")
print("=" * 60)

quantize_dynamic(

    model_input=ONNX_MODEL, 

    model_output=OUTPUT_MODEL,

    weight_type=QuantType.QInt8

)

print("=" * 60)
print("HOÀN THÀNH!")
print("=" * 60)

print("Đã lưu tại:")
print(OUTPUT_MODEL)