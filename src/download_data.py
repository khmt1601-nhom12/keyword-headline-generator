import os
from datasets import load_dataset
import pandas as pd

# Tự động tạo thư mục chứa nếu chưa có để tránh lỗi hoàn toàn trên Windows
os.makedirs(os.path.join("data", "splits"), exist_ok=True)

print("🔄 Đang tải dataset từ Hugging Face... Vui lòng đợi...")
# Tải tập dữ liệu chính xác của đề tài
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0003")

print("\n📊 Các tập dữ liệu tìm thấy trên hệ thống:", dataset.keys())

# Chuyển đổi dữ liệu và chia tập Train/Dev/Test (80/10/10)
if 'train' in dataset:
    df_all = pd.DataFrame(dataset['train'])
    
    print("✂️ Đang phân chia dữ liệu theo tỷ lệ 80% Train / 10% Dev / 10% Test...")
    train_df = df_all.sample(frac=0.8, random_state=42)
    rest_df = df_all.drop(train_df.index)
    dev_df = rest_df.sample(frac=0.5, random_state=42)
    test_df = rest_df.drop(dev_df.index)
    
    # Lưu vào đúng thư mục splits bằng đường dẫn an toàn trên mọi hệ điều hành
    train_df.to_csv(os.path.join("data", "splits", "train.csv"), index=False, encoding="utf-8")
    dev_df.to_csv(os.path.join("data", "splits", "dev.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join("data", "splits", "test.csv"), index=False, encoding="utf-8")
    
    print("✅ Đã lưu thành công các file tại 'data/splits/'!")
    print(f"   - Train: {len(train_df)} dòng | Dev: {len(dev_df)} dòng | Test: {len(test_df)} dòng")
