# Giải thích chi tiết: `build_index.py`

**Đường dẫn:** `C:/Users/ezycloudx-admin/Desktop/seg/main/src/offline/stage3_build_index/build_index.py`


## 📌 Nhiệm vụ (What it is)
Xây dựng Từ điển ASMK (Aggregated Selective Match Kernel) để phục vụ cho việc tìm kiếm Siêu tốc ở Stage 4, 5.

## ⚙️ Cách hoạt động (How it works)
1. Đọc toàn bộ hàng triệu vector `(600, 128)` từ thư mục Output của Stage 1.
2. Ép chúng thành một khối khổng lồ `np.vstack`.
3. Gọi thư viện Faiss trên GPU chạy thuật toán K-Means Clustering để tạo ra $65,536$ tâm cụm (Visual Words).
4. Ánh xạ toàn bộ ảnh vào các cụm này và tính phần dư (Residuals).
5. Xây dựng Inverted File (Bảng băm ngược) để tra cứu.

## 📥 Đầu vào (Input)
- Toàn bộ các file `.npy` từ Stage 1.

## 📤 Đầu ra (Output)
- `_codebook.pkl` (65,536 Từ vựng).
- `_ivf.pkl` (Danh mục tra cứu ảnh).
