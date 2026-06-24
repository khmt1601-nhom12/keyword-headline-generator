import pandas as pd
import torch
import evaluate
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# ==========================
# Load model
# ==========================

MODEL_PATH = "models/run_04_batch4_epoch5"

print("=" * 60)
print("ĐANG LOAD MODEL...")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

model.to(device)

model.eval()

print("Device:", device)

# ==========================
# Load ROUGE
# ==========================

rouge = evaluate.load("rouge")

# ==========================
# Đọc dữ liệu test
# ==========================

df = pd.read_csv("data/cleaned/test_clean.csv")

predictions = []

references = []

print("=" * 60)
print("BẮT ĐẦU ĐÁNH GIÁ")
print("=" * 60)

for i, row in tqdm(df.iterrows(), total=len(df)):

    keywords = row["keywords"]

    reference = row["title"]

    inputs = tokenizer(
        keywords,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    predictions.append(prediction)

    references.append(reference)

    if i < 10:
        print("-" * 60)
        print("Keywords:")
        print(keywords)
        print()
        print("Prediction:")
        print(prediction)
        print()
        print("Reference:")
        print(reference)

# ==========================
# ROUGE
# ==========================

results = rouge.compute(
    predictions=predictions,
    references=references
)

print("\n")
print("=" * 60)
print("KẾT QUẢ ROUGE")
print("=" * 60)

for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")