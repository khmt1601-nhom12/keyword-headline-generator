# ==========================================
# CẤU HÌNH CHUNG CHO TOÀN BỘ DỰ ÁN
# ==========================================

import os

# ==========================
# Thư mục dữ liệu
# ==========================

DATA_DIR = "data"

CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")

TRAIN_FILE = os.path.join(CLEAN_DIR, "train_clean.csv")
DEV_FILE = os.path.join(CLEAN_DIR, "dev_clean.csv")
TEST_FILE = os.path.join(CLEAN_DIR, "test_clean.csv")

# ==========================
# Thư mục output
# ==========================

OUTPUT_DIR = "outputs"

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ==========================
# Keyword Extraction
# ==========================

TOP_K = 5

# ==========================
# Random Seed
# ==========================

RANDOM_STATE = 42