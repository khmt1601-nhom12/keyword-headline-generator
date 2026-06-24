import pandas as pd
import os
from datasets import Dataset
from transformers import AutoTokenizer

# ==========================
# Model sử dụng
# ==========================
MODEL_NAME = "VietAI/vit5-base"

# ==========================
# Khởi tạo tokenizer
# ==========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

# ==========================
# Đọc dữ liệu
# ==========================
def load_data(csv_path):
    """
    Đọc file csv và chuyển thành HuggingFace Dataset
    """
    df = pd.read_csv(csv_path)

    dataset = Dataset.from_pandas(df)

    return dataset


# ==========================
# Tokenize
# ==========================
def preprocess_function(examples):

    model_inputs = tokenizer(
        examples["keywords"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=examples["title"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    # model_inputs["labels"] = labels["input_ids"]
    labels_ids = labels["input_ids"]

    labels_ids = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in labels_ids
    ]

    model_inputs["labels"] = labels_ids

    return model_inputs


# ==========================
# Chuẩn bị Dataset
# ==========================
def prepare_dataset(csv_path):

    dataset = load_data(csv_path)

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    os.makedirs("data/tokenized", exist_ok=True)

    tokenized_dataset.save_to_disk("data/tokenized/train_dataset")

    print("\nĐã lưu dataset tokenized vào:")
    print("data/tokenized/train_dataset")

    return tokenized_dataset


# ==========================
# Test
# ==========================
if __name__ == "__main__":

    train_dataset = prepare_dataset(
        "data/cleaned/train_clean.csv"
    )

    print(train_dataset)

    print(train_dataset[0])