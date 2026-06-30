# Stage 1: Local Feature Extraction (FIRe)

## 📌 Chức năng
Stage 1 chịu trách nhiệm trích xuất **Đặc trưng Cục bộ (Local Features)** từ tất cả các hình ảnh trong Database và Query.
Hệ thống sử dụng mô hình **FIRe** (phát triển bởi Facebook AI). Đối với mỗi bức ảnh, mô hình sẽ quét qua và trích xuất ra 600 vector cục bộ (mỗi vector dài 128 chiều) đại diện cho các vùng chi tiết khác nhau trên bức ảnh.

## 📥 Dữ liệu Đầu vào (Input)
*   Thư mục chứa ảnh gốc (ví dụ: `data/datasets/roxford5k/jpg/`).
*   Trọng số mô hình FIRe: `fire/model/model_best.pth.tar`.

## 📤 Dữ liệu Đầu ra (Output)
*   Thư mục lưu trữ: `output/stage1/features/roxford5k/`
*   Các file định dạng numpy `.npy`, mỗi file tương ứng với một bức ảnh.
*   Shape của mỗi file `.npy` là `(600, 128)`.

## 🚀 Cách chạy (How to run)

Di chuyển vào thư mục gốc của dự án và chạy lệnh sau:

```bash
python src/offline/stage1_local_extract/extract_local.py
```
