import torch

# Nạp các DLL CUDA mà PyTorch mang theo
import os
os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

import onnxruntime as ort
import os
import time
import pandas as pd
import torch
import onnxruntime as ort

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# =====================================================
# ĐƯỜNG DẪN
# =====================================================

MODEL_PATH = "models/best_model"

ONNX_PATH = "outputs/onnx/vit5_encoder.onnx"

ONNX_INT8_PATH = "outputs/onnx/headline_generator_int8.onnx"

CSV_OUTPUT = "outputs/benchmarks/inference_benchmark.csv"

os.makedirs("outputs/benchmarks", exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 60)
print("ĐANG LOAD MODEL...")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

model.to(device)

model.eval()

# =====================================================
# LOAD ONNX
# =====================================================

print("=" * 60)
print("ĐANG LOAD ONNX...")
print("=" * 60)

providers = (
    ["CUDAExecutionProvider"]
    if torch.cuda.is_available()
    else ["CPUExecutionProvider"]
)

ort_session = ort.InferenceSession(
    ONNX_PATH,
    providers=providers
)

ort_int8_session = ort.InferenceSession(
    ONNX_INT8_PATH,
    providers=providers
)

# =====================================================
# SAMPLE INPUT
# =====================================================

text = "bóng đá việt nam aff cup"

inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=64,
    truncation=True,
    padding="max_length"
)

input_ids = inputs["input_ids"].to(device)

attention_mask = inputs["attention_mask"].to(device)

onnx_inputs = {
    "input_ids": input_ids.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy()
}

# =====================================================
# WARMUP
# =====================================================

print("=" * 60)
print("WARMUP...")
print("=" * 60)

with torch.no_grad():

    for _ in range(10):

        model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

for _ in range(10):

    ort_session.run(None, onnx_inputs)

    ort_int8_session.run(None, onnx_inputs)

# =====================================================
# BENCHMARK
# =====================================================

print("=" * 60)
print("ĐANG BENCHMARK...")
print("=" * 60)

N = 100

# -----------------------------
# PyTorch
# -----------------------------

start = time.perf_counter()

with torch.no_grad():

    for _ in range(N):

        model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

if torch.cuda.is_available():
    torch.cuda.synchronize()

pytorch_time = (time.perf_counter() - start) / N * 1000

# -----------------------------
# ONNX
# -----------------------------

start = time.perf_counter()

for _ in range(N):

    ort_session.run(None, onnx_inputs)

onnx_time = (time.perf_counter() - start) / N * 1000

# -----------------------------
# ONNX INT8
# -----------------------------

start = time.perf_counter()

for _ in range(N):

    ort_int8_session.run(None, onnx_inputs)

int8_time = (time.perf_counter() - start) / N * 1000

# =====================================================
# MODEL SIZE
# =====================================================

def folder_size(folder):

    total = 0

    for root, _, files in os.walk(folder):

        for file in files:

            total += os.path.getsize(
                os.path.join(root, file)
            )

    return total / 1024 / 1024


pytorch_size = folder_size(MODEL_PATH)

onnx_size = (
    os.path.getsize(ONNX_PATH)
    + os.path.getsize(ONNX_PATH + ".data")
) / 1024 / 1024
int8_size = os.path.getsize(ONNX_INT8_PATH) / 1024 / 1024

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame({

    "Model": [
        "PyTorch",
        "ONNX",
        "ONNX INT8"
    ],

    "Size(MB)": [
        round(pytorch_size, 2),
        round(onnx_size, 2),
        round(int8_size, 2)
    ],

    "Inference(ms/sample)": [
        round(pytorch_time, 3),
        round(onnx_time, 3),
        round(int8_time, 3)
    ]

})

df.to_csv(
    CSV_OUTPUT,
    index=False
)

# =====================================================
# RESULT
# =====================================================

print()

print("=" * 60)
print(df)
print("=" * 60)

print()

print("Đã lưu:")

print(CSV_OUTPUT)