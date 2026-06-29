# 🧩 Stage 1: FIRe Local Feature Extraction (Offline)

## 📌 Chức năng (What it does)
Quét qua toàn bộ hình ảnh trong Database và trích xuất các **đặc trưng cục bộ (Local Features)** bằng mô hình FIRe.
Bạn sẽ chạy lệnh này MỘT LẦN DUY NHẤT để số hóa (Digitize) toàn bộ hình ảnh thô thành các Ma trận số học `.npy`.

## 🛠 Lệnh thực thi:
```bash
python src/offline/stage1_local_extract/extract_local.py
```

## 🎯 Mô tả & Ví dụ đầu ra (What you get):
Khi chạy lệnh, hệ thống sẽ tiến hành khởi tạo mô hình FIRe trên GPU và bắt đầu trích xuất tuần tự.
Màn hình Terminal sẽ xuất hiện:
```text
Loading FIRe model...
Model loaded successfully!
============================================================
DEBUG: Checking configuration
============================================================
Dataset name: roxford5k
Number of queries: 70
Number of database: 5063

Processing 70 QUERY images...
Query   1/70: all_souls_1 -> (600, 128)
Query   2/70: all_souls_2 -> (600, 128)
...
Query processing complete: 70 success, 0 failed

Processing 5063 DATABASE images...
DB      1/5063: all_souls_000000 -> (600, 128)
DB      2/5063: all_souls_000001 -> (600, 128)
...
Progress: 100/5063
...
Database processing complete: 5063 success, 0 failed
```

### 🧠 Phân tích kết quả:
- **`-> (600, 128)`**: Có nghĩa là từ một bức ảnh khổng lồ ban đầu, AI (ResNet50) đã tìm ra đúng **600 điểm nổi bật nhất** (ví dụ: góc cửa sổ, mái vòm, hoa văn tòa nhà). Mỗi điểm được biểu diễn bằng một dãy số gồm 128 con số (Vector 128D).
- Các ma trận này được lưu thẳng vào ổ cứng dưới định dạng file Numpy siêu nhẹ.
- **Vị trí lưu:** `output/stage1/features/roxford5k/database/`

---

## 🧮 Nguyên lý toán học (Mathematical Logic)
Mỗi bức ảnh $I$ được cho qua mạng ResNet50 (SFM). 
Thay vì lấy 1 vector duy nhất, mạng xuất ra một ma trận Tensor $\mathcal{X} \in \mathbb{R}^{C \times H \times W}$.
Các điểm ảnh (pixels) được chọn lọc qua Threshold và nén lại thành $N$ vector $v_i \in \mathbb{R}^{128}$ (thường chọn $N=600$ điểm nổi bật nhất).
- Ký hiệu tập hợp điểm của bức ảnh: $V(I) = \{v_1, v_2, ..., v_N\}$
