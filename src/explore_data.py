import pandas as pd

# Đọc tập train
df = pd.read_csv("data/splits/train.csv")

print("=" * 60)
print("THÔNG TIN DATASET")
print("=" * 60)

# Kích thước
print(f"Số dòng: {df.shape[0]}")
print(f"Số cột : {df.shape[1]}")

print("\nTên các cột:")
print(df.columns.tolist())

print("\n")

print("=" * 60)
print("KIỂM TRA THIẾU DỮ LIỆU")
print("=" * 60)

print(df.isnull().sum())

print("\n")

print("=" * 60)
print("KIỂM TRA DỮ LIỆU TRÙNG")
print("=" * 60)

print(f"Số dòng trùng: {df.duplicated().sum()}")

print("\n")

print("=" * 60)
print("5 DÒNG ĐẦU")
print("=" * 60)

print(df.head())