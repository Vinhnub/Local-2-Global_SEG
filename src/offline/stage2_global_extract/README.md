# Stage 2: Global Feature Extraction (CVNet)

## 📌 Chức năng
Stage 2 đảm nhiệm việc trích xuất **Đặc trưng Toàn cục (Global Features)** cho tất cả các bức ảnh.
Sử dụng mô hình **CVNet-R101** (ResNet-101), mỗi bức ảnh sẽ được nén toàn bộ thông tin (bối cảnh, màu sắc, hình khối lớn) thành một vector duy nhất. 

## 📥 Dữ liệu Đầu vào (Input)
*   Thư mục chứa ảnh gốc (`data/datasets/roxford5k/jpg/`).
*   Trọng số mô hình CVNet: `CVNet/weights/CVNet-R101.pth`.

## 📤 Dữ liệu Đầu ra (Output)
*   Thư mục lưu trữ: `output/stage2/features/roxford5k/`
*   Các file định dạng numpy `.npy`.
*   Shape của mỗi file `.npy` là `(2048,)` (Một vector 2048 chiều).

## 🚀 Cách chạy (How to run)

Từ thư mục gốc dự án, chạy lệnh:

```bash
python src/offline/stage2_global_extract/extract_global.py
```
