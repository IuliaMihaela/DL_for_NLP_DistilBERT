import os
from onnxruntime.quantization import quantize_dynamic, QuantType

os.makedirs("onnxfiles", exist_ok=True)

pairs = [
    ("onnxfiles/distilbert_baseline.onnx", "onnxfiles/distilbert_baseline_int8.onnx"),
    ("onnxfiles/distilbert_pruned.onnx",   "onnxfiles/distilbert_pruned_int8.onnx"),
]

for inp, outp in pairs:
    quantize_dynamic(
        model_input=inp,
        model_output=outp,
        weight_type=QuantType.QInt8,
    )
    print("wrote:", outp)
