import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# =====================================================
# BEST MODEL
# =====================================================

MODEL_PATH = "models/best_model"

OUTPUT_DIR = "outputs/onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("ĐANG LOAD BEST MODEL...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

print("Device:", device)

# =====================================================
# EXPORT ENCODER
# =====================================================

dummy = tokenizer(
    "bóng đá việt nam aff cup",
    return_tensors="pt",
    max_length=64,
    truncation=True,
    padding="max_length",
)

input_ids = dummy["input_ids"].to(device)
attention_mask = dummy["attention_mask"].to(device)

onnx_path = os.path.join(
    OUTPUT_DIR,
    "vit5_encoder.onnx"
)

print("=" * 60)
print("ĐANG EXPORT ENCODER...")
print("=" * 60)

torch.onnx.export(

    model.encoder,

    (input_ids, attention_mask),

    onnx_path,

    export_params=True,

    opset_version=18,

    do_constant_folding=True,

    input_names=[
        "input_ids",
        "attention_mask",
    ],

    output_names=[
        "last_hidden_state",
    ],

    dynamic_axes={

        "input_ids": {
            0: "batch",
            1: "sequence",
        },

        "attention_mask": {
            0: "batch",
            1: "sequence",
        },

        "last_hidden_state": {
            0: "batch",
            1: "sequence",
        },

    },

)

print("=" * 60)
print("EXPORT THÀNH CÔNG")
print("=" * 60)

print("Đã lưu:")
print(onnx_path)