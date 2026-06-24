import os
import re
import unicodedata
import pandas as pd

# ==========================
# Tạo thư mục lưu dữ liệu sạch
# ==========================
os.makedirs("data/cleaned", exist_ok=True)


def clean_text(text):
    """
    Hàm làm sạch văn bản.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFC", text)

    # Xóa xuống dòng
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Xóa tab
    text = text.replace("\t", " ")

    # Xóa khoảng trắng dư
    text = re.sub(r"\s+", " ", text)

    # Xóa khoảng trắng đầu cuối
    text = text.strip()

    return text


def preprocess(input_path, output_path):

    print("=" * 60)
    print(f"Đang xử lý: {input_path}")

    df = pd.read_csv(input_path)

    before = len(df)

    # Xóa dòng thiếu dữ liệu
    df = df.dropna()

    # Xóa dòng trùng
    df = df.drop_duplicates()

    # Làm sạch các cột văn bản
    text_columns = [
        "document",
        "title",
        "summary",
        "keywords"
    ]

    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    after = len(df)

    print(f"Số dòng trước xử lý : {before}")
    print(f"Số dòng sau xử lý   : {after}")
    print(f"Đã loại bỏ          : {before-after}")

    df.to_csv(output_path, index=False, encoding="utf-8")

    print("Đã lưu:", output_path)


preprocess(
    "data/processed/train_processed.csv",
    "data/cleaned/train_clean.csv"
)

preprocess(
    "data/processed/dev_processed.csv",
    "data/cleaned/dev_clean.csv"
)

preprocess(
    "data/processed/test_processed.csv",
    "data/cleaned/test_clean.csv"
)

print("\nHoàn thành tiền xử lý dữ liệu.")