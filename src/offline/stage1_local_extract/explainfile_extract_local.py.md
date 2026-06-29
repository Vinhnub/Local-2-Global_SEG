# Giải thích chi tiết: `extract_local.py`

**Đường dẫn:** `C:/Users/ezycloudx-admin/Desktop/seg/main/src/offline/stage1_local_extract/extract_local.py`


## 📌 Nhiệm vụ (What it is)
File này đóng vai trò Trích xuất Đặc trưng Cục bộ (Local Feature Extraction) sử dụng mô hình FIRe. Nó là trái tim của Stage 1 (Offline).

## ⚙️ Cách hoạt động (How it works)
1. Khởi tạo mô hình FIRe (SfM 120k).
2. Lặp qua toàn bộ hình ảnh trong Dataset (Cả tập Database và tập Query).
3. Đưa từng ảnh vào mô hình: `net.forward_local()`.
4. Trích xuất ra $N=600$ điểm nổi bật nhất (Keypoints). Mỗi điểm là 1 Vector 128 chiều.
5. Lưu kết quả thành các ma trận Numpy `.npy`.

## 📥 Đầu vào (Input)
- Các file ảnh gốc định dạng `.jpg`.
- File Config/Ground Truth `.pkl` để xác định đường dẫn ảnh.

## 📤 Đầu ra (Output)
- File `(600, 128) .npy` tương ứng với mỗi bức ảnh.
