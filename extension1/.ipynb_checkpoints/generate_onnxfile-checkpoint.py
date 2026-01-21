import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "distilbert-base-uncased"
PT_DIR = "optimized_models"
ONNX_DIR = "onnxfiles"

os.makedirs(ONNX_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dummy input
text = "This is a test sentence for ONNX export."
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128
)
inputs.pop("token_type_ids", None)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

pt_files = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))

if len(pt_files) == 0:
    print(f"No .pt files found in: {PT_DIR}")
    exit()

for pt_path in pt_files:
    base_name = os.path.splitext(os.path.basename(pt_path))[0]
    onnx_path = os.path.join(ONNX_DIR, f"{base_name}.onnx")

    # Load model structure
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    # Load weights
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # Export
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )

    print("Exported:", onnx_path)

print("\nAll exports completed.")
