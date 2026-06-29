# Giải thích chi tiết: `extract_global.py`

**Đường dẫn:** `C:/Users/ezycloudx-admin/Desktop/seg/main/src/offline/stage2_global_extract/extract_global.py`


## 📌 Nhiệm vụ (What it is)
File Trích xuất Đặc trưng Toàn cảnh (Global Feature Extraction) bằng mô hình CVNet/SuperGlobal. Nhiệm vụ là nén toàn bộ hình ảnh thành 1 vector duy nhất.

## ⚙️ Cách hoạt động (How it works)
1. Khởi tạo mạng CVNet (Dựa trên ResNet101).
2. Tải trọng số Pre-trained từ file `.pyth`.
3. Cho ảnh đi qua mạng CNN để lấy Feature Map tổng.
4. Áp dụng GeM Pooling (Generalized Mean Pooling) để gộp toàn bộ ma trận thành 1 chuỗi dài 2048 con số.
5. L2 Normalize để chuẩn hóa chuỗi này.
6. Lưu lại.

## 📥 Đầu vào (Input)
- Ảnh `.jpg`.

## 📤 Đầu ra (Output)
- File `(1, 2048) .npy` mang ý nghĩa bối cảnh toàn cục.
