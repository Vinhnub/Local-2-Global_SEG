# Stage 4: Search & Evaluate mAP

## 📌 Chức năng
Stage 4 là vòng kiểm thử hàng loạt (Evaluation). Nó sẽ lấy toàn bộ tập ảnh Query (ví dụ 55 ảnh truy vấn của tập Oxford5k), đưa vào hệ thống L2G để tìm kiếm. 
Mục tiêu là mô phỏng luồng chạy L2G thực tế:
1.  **Base Search:** So sánh Query với Database để lấy top 1600.
2.  **MDS Embedding:** Nén từ ma trận Chamfer (có sử dụng Sparse Lookup siêu tốc từ Stage 3) thành vector.
3.  **L2G Fusion & Graph Diffusion:** Hợp nhất Local + Global và chạy mô hình Đồ thị.
Sau đó, hệ thống sẽ tự động so sánh danh sách kết quả với Ground Truth (đáp án chuẩn do con người gán) để chấm điểm mAP (Mean Average Precision) cho 3 độ khó: Easy, Medium, Hard.

## 📥 Dữ liệu Đầu vào (Input)
*   Đặc trưng Local & Global của toàn bộ Query và Database (Stage 1 & 2).
*   Từ điển Sparse Lookup: `output/stage3/roxford5k_sparse_sim.pkl` (Bắt buộc phải có, nếu chưa chạy Stage 3 hệ thống sẽ báo lỗi).

## 📤 Dữ liệu Đầu ra (Output)
*   Bảng điểm mAP in ra màn hình.
*   File Text log kết quả: `output/stage3/roxford5k_final_results.txt`.
*   File Ranking: `output/stage3/roxford5k_ranks.npy`.

## 🚀 Cách chạy (How to run)

Từ thư mục gốc dự án, chạy lệnh:

```bash
# Chạy với Backend Tự động (Ưu tiên CANN nếu có, không có dùng PyTorch)
python src/online/stage4_search/search_exact_chamfer.py --backend auto

# Hoặc ép buộc chạy bằng PyTorch GPU 
python src/online/stage4_search/search_exact_chamfer.py --backend pytorch
```
