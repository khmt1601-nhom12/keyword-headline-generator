import os
import pandas as pd

# Tạo thư mục output
os.makedirs("data/processed", exist_ok=True)


def prepare(input_file, output_file, split_name):

    print(f"\nĐang xử lý {input_file}")

    df = pd.read_csv(input_file)

    # Đổi tên cột index -> id
    df = df.rename(columns={
        "index": "id"
    })

    # Tạo title từ summary
    df["title"] = df["summary"]

    # Thêm source
    df["source"] = "8Opt/vietnamese-summarization-dataset-0003"

    # Chưa có chủ đề
    df["topic"] = "unknown"

    # train / dev / test
    df["split"] = split_name

    # Sắp xếp lại thứ tự cột
    df = df[
        [
            "id",
            "document",
            "title",
            "keywords",
            "summary",
            "source",
            "topic",
            "split"
        ]
    ]

    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Đã lưu {output_file}")
    print(df.head())


prepare(
    "data/splits/train.csv",
    "data/processed/train_processed.csv",
    "train"
)

prepare(
    "data/splits/dev.csv",
    "data/processed/dev_processed.csv",
    "dev"
)

prepare(
    "data/splits/test.csv",
    "data/processed/test_processed.csv",
    "test"
)

print("\nHoàn thành chuẩn hóa dataset.")