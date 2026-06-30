# Data Card: Bộ dữ liệu sinh tiêu đề báo tiếng Việt

## 1. Tóm tắt bộ dữ liệu (Dataset Summary)

Bộ dữ liệu được sử dụng trong đề tài "Keyword → Headline Generator" là tập dữ liệu tiếng Việt do giảng viên cung cấp phục vụ học phần Xử lý Ngôn ngữ Tự nhiên (CSC4005 & CSC4007).

Mỗi mẫu dữ liệu bao gồm:

- Từ khóa (Keywords)
- Tiêu đề bài báo (Headline)

Mục tiêu của bộ dữ liệu là huấn luyện mô hình học sâu sinh tiêu đề báo tiếng Việt từ tập từ khóa đầu vào.

Sau khi tiền xử lý và chia dữ liệu:

| Tập dữ liệu | Số lượng |
|------------|---------:|
| Train | 24.721 |
| Validation | 3.090 |
| Test | 3.090 |

Tổng số mẫu: **30.901**

---

## 2. Mục đích và phạm vi sử dụng (Motivation and Intended Use)

Bộ dữ liệu được xây dựng nhằm phục vụ:

- Huấn luyện mô hình sinh tiêu đề tiếng Việt.
- Đánh giá các mô hình Seq2Seq.
- Thực hành Fine-tuning mô hình Transformer.
- So sánh hiệu năng giữa các mô hình học sâu.

Không hướng đến sử dụng như một nguồn dữ liệu báo chí chính thức.

---

## 3. Nguồn dữ liệu (Dataset Sources)

Bộ dữ liệu được cung cấp bởi giảng viên môn học CSC4005 & CSC4007.

Nguồn dữ liệu ban đầu được tổng hợp từ các bài báo tiếng Việt trên Internet, sau đó được chuẩn hóa và phát hành phục vụ mục đích học tập.

Nhóm không trực tiếp thu thập dữ liệu gốc.

---

## 4. Thành phần dữ liệu (Dataset Composition)

Mỗi mẫu dữ liệu gồm hai trường:

| Trường | Ý nghĩa |
|---------|---------|
| keywords | Danh sách từ khóa đầu vào |
| headline | Tiêu đề bài báo tương ứng |

Ví dụ:

Keywords

```
giá vàng, hôm nay
```

Headline

```
Giá vàng hôm nay tăng mạnh trong phiên giao dịch sáng.
```

---

## 5. Quá trình thu thập dữ liệu (Data Collection Process)

Nhóm không trực tiếp thu thập dữ liệu.

Bộ dữ liệu được giảng viên chuẩn bị trước và phát cho sinh viên phục vụ học phần.

Do đó nhóm chỉ thực hiện các bước:

- Kiểm tra dữ liệu.
- Làm sạch dữ liệu.
- Chia train/validation/test.
- Huấn luyện mô hình.

---

## 6. Quá trình gán nhãn (Annotation Process)

Không tồn tại quá trình gán nhãn thủ công.

Tiêu đề bài báo chính là nhãn (label) của từng mẫu.

Input:

```
Keywords
```

Output:

```
Headline
```

---

## 7. Tiền xử lý dữ liệu (Preprocessing)

Nhóm thực hiện các bước tiền xử lý sau:

- Loại bỏ các dòng dữ liệu bị thiếu.
- Chuẩn hóa khoảng trắng.
- Chuẩn hóa encoding UTF-8.
- Tokenize bằng tokenizer của mô hình ViT5.
- Padding chuỗi đầu vào đến 128 token.
- Padding chuỗi đầu ra đến 64 token.
- Cắt bỏ chuỗi vượt quá chiều dài tối đa.

---

## 8. Chia tập dữ liệu (Train/Validation/Test Split)

Dữ liệu được chia thành ba tập:

| Tập | Số lượng |
|------|---------:|
| Train | 24.721 |
| Validation | 3.090 |
| Test | 3.090 |

Tỷ lệ xấp xỉ:

- Train: 80%
- Validation: 10%
- Test: 10%

---

## 9. Phân bố nhãn (Label Distribution)

Do đây là bài toán sinh văn bản (Text Generation), mỗi tiêu đề gần như là duy nhất.

Không tồn tại các lớp (classes) cố định như bài toán phân loại.

Do đó không có biểu đồ phân bố nhãn truyền thống.

---

## 10. Kiểm tra chất lượng dữ liệu (Data Quality Checks)

Nhóm đã thực hiện:

- Kiểm tra dữ liệu thiếu.
- Kiểm tra dòng rỗng.
- Kiểm tra dữ liệu trùng lặp.
- Kiểm tra lỗi encoding.
- Kiểm tra số lượng mẫu sau khi chia dữ liệu.

Các tập train, validation và test đều đọc thành công trước khi huấn luyện.

---

## 11. Các vấn đề đạo đức (Ethical Considerations)

Bộ dữ liệu được sử dụng hoàn toàn cho mục đích học tập và nghiên cứu.

Nhóm không sử dụng dữ liệu để:

- phát tán tin giả;
- tạo nội dung độc hại;
- xâm phạm quyền riêng tư.

---

## 12. Sai lệch và hạn chế (Biases and Limitations)

Một số hạn chế của bộ dữ liệu:

- Chỉ gồm dữ liệu tiếng Việt.
- Chủ yếu là tin tức báo chí.
- Có thể thiên lệch về các chủ đề phổ biến như:
  - thời sự;
  - thể thao;
  - kinh tế;
  - giải trí.
- Không phản ánh đầy đủ mọi lĩnh vực của tiếng Việt.

---

## 13. Giấy phép và quyền truy cập (License and Access)

Bộ dữ liệu được cung cấp cho sinh viên trong phạm vi học phần.

Việc sử dụng ngoài mục đích học tập cần tuân theo quy định của giảng viên và nhà trường.

---

## 14. Mục đích sử dụng khuyến nghị (Recommended Uses)

Bộ dữ liệu phù hợp cho:

- Fine-tuning ViT5.
- Huấn luyện Transformer.
- Nghiên cứu Text Generation.
- Sinh tiêu đề báo.
- Thực hành NLP tiếng Việt.

---

## 15. Các mục đích không nên sử dụng (Prohibited or Risky Uses)

Không nên sử dụng bộ dữ liệu cho:

- Sinh tin giả.
- Tạo nội dung gây hiểu nhầm.
- Phát tán thông tin sai lệch.
- Các ứng dụng thương mại khi chưa được cấp phép.
- Các hệ thống yêu cầu độ chính xác tuyệt đối.