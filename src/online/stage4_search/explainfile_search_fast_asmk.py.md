# Giải thích chi tiết: `search_fast_asmk.py`

**Đường dẫn:** `C:/Users/ezycloudx-admin/Desktop/seg/main/src/online/stage4_search/search_fast_asmk.py`


## 📌 Nhiệm vụ (What it is)
Script tìm kiếm Siêu tốc để đánh giá điểm mAP trên bộ Dataset tiêu chuẩn bằng hệ thống kết hợp L2G (Local-to-Global).

## ⚙️ Cách hoạt động (How it works)
1. Đưa 70 ảnh Query vào ASMK để lọc nhanh ra 100 ảnh giống nhất.
2. Áp dụng Exact Chamfer Distance để đo khoảng cách chính xác từng điểm với 100 ảnh này.
3. Dùng thuật toán MDS (Multi-Dimensional Scaling) hạ chiều ma trận khoảng cách thành Vector 128D.
4. Gộp vector MDS này với vector Global của CVNet.
5. Đưa vào Đồ thị lan truyền (SuperGlobal Graph) để các ảnh tự gánh điểm cho nhau.
6. Tính mAP bằng Ground Truth.

## 📥 Đầu vào (Input)
- Vector Local của Query và Database.
- Từ điển ASMK.
- Vector Global.

## 📤 Đầu ra (Output)
- Điểm đánh giá mAP (Medium, Hard).
