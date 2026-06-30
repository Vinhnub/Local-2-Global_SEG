# Stage 5: Real-time Test Query

## 📌 Chức năng
Stage 5 là một ứng dụng "chạy thực tế" của hệ thống L2G. 
Thay vì lấy ảnh Query trong tập Dataset có sẵn, Stage 5 cho phép bạn đưa vào một bức ảnh định dạng `jpg` (hoặc `png`) bất kỳ lấy từ bên ngoài.
Khi bạn đưa ảnh vào, Stage 5 sẽ:
1.  Khởi tạo tức thời 2 mô hình (FIRe và CVNet) để trích xuất Local & Global cho ảnh đó.
2.  Chạy pipeline L2G siêu tốc sử dụng từ điển Sparse Lookup.
3.  Vẽ (Visualize) trực quan bức ảnh Query của bạn bên cạnh Top 20 ảnh giống nhất trong kho dữ liệu (Database).

## 📥 Dữ liệu Đầu vào (Input)
*   Đường dẫn tới 1 file ảnh (ví dụ: `my_test_image.jpg`).
*   Các file model weights (FIRe và CVNet).
*   Từ điển Sparse Lookup: `output/stage3/roxford5k_sparse_sim.pkl` (Bắt buộc phải có, nếu chưa chạy Stage 3 hệ thống sẽ báo lỗi).

## 📤 Dữ liệu Đầu ra (Output)
*   Thời gian tìm kiếm (tốc độ xử lý tính bằng giây) in ra terminal.
*   Một file hình ảnh trực quan thể hiện kết quả tìm kiếm được lưu tại: `output_test/result_<tên_ảnh_query>.png`.

## 🚀 Cách chạy (How to run)

Từ thư mục gốc dự án, chuẩn bị sẵn 1 tấm ảnh và chạy lệnh:

```bash
# Cú pháp
python src/online/stage5_test_query/test_query.py --image <đường_dẫn_tới_ảnh> --backend <auto|pytorch|cann>

# Ví dụ chạy thực tế
python src/online/stage5_test_query/test_query.py --image test.jpg --backend pytorch
```
