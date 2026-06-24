import pandas as pd
import re
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from config import *
with open("data/stopwords_vi.txt", encoding="utf-8") as f:
    stopwords = [line.strip() for line in f]
print("=" * 60)
print("ĐỌC DỮ LIỆU")

df = pd.read_csv(TRAIN_FILE)

print(df.head())

print(f"\nSố lượng bài báo: {len(df)}")
print("\n" + "=" * 60)
print("TÁCH TỪ TIẾNG VIỆT")


def tokenize(text):
    """
    Tách từ tiếng Việt bằng Underthesea.
    format='text' sẽ nối từ nhiều âm tiết bằng dấu _
    """
    return word_tokenize(str(text), format="text")


print("Đang tách từ... (có thể mất vài phút)")

df["document_tokenized"] = df["document"].apply(tokenize)

print("\nVí dụ sau khi tách từ:\n")

for i in range(3):
    print(f"--- Văn bản {i+1} ---")
    print(df["document_tokenized"].iloc[i][:500])  # chỉ in 500 ký tự đầu
    print()

print("\n" + "=" * 60)
print("HUẤN LUYỆN TF-IDF")

vectorizer = TfidfVectorizer(
    lowercase=False,
    stop_words=stopwords,
    max_features=8000,
    min_df=2,
    max_df=0.8,
    token_pattern=r"(?u)\b[\w_]{2,}\b"
)

tfidf_matrix = vectorizer.fit_transform(df["document_tokenized"])

feature_names = vectorizer.get_feature_names_out()

print(f"Số lượng document : {tfidf_matrix.shape[0]}")
print(f"Kích thước từ vựng: {tfidf_matrix.shape[1]}")

print("\n10 từ đầu tiên trong Vocabulary:")

print(feature_names[:10])

print("\n" + "=" * 60)
print("TRÍCH XUẤT TOP KEYWORDS")


def extract_top_keywords(tfidf_row, feature_names, top_k=TOP_K):
    """
    Lấy Top-K từ có điểm TF-IDF cao nhất của một document.
    """

    # Chuyển sparse matrix -> numpy array
    scores = tfidf_row.toarray().flatten()

    # Sắp xếp giảm dần theo điểm TF-IDF
    sorted_indices = scores.argsort()[::-1]

    keywords = []

    for idx in sorted_indices:

        # Bỏ qua nếu điểm TF-IDF bằng 0
        if scores[idx] == 0:
            continue

        word = feature_names[idx]

        # Bỏ token chỉ chứa số
        if word.isdigit():
            continue

        # Bỏ token quá ngắn
        if len(word) < 2:
            continue

        # Bỏ token bắt đầu bằng số (ví dụ: 2020_năm)
        if word[0].isdigit():
            continue

        keywords.append(word)

        # Đủ số lượng keyword thì dừng
        if len(keywords) >= top_k:
            break

    return keywords

predicted_keywords = []

print("\nĐang trích xuất từ khóa...")

for i in range(len(df)):
    kw = extract_top_keywords(
        tfidf_matrix[i],
        feature_names
    )

    predicted_keywords.append("; ".join(kw))

df["predicted_keywords"] = predicted_keywords

print("\nVí dụ kết quả:\n")

for i in range(5):

    print("=" * 60)

    print("Keyword thật:")

    print(df["keywords"].iloc[i])

    print()

    print("Keyword dự đoán:")

    print(df["predicted_keywords"].iloc[i])

    print()