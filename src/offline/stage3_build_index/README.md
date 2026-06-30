# Stage 3: Build Index (Xây dựng Từ điển Khoảng cách)

## 📌 Chức năng
Stage 3 là bước chuẩn bị cuối cùng (Offline) trước khi hệ thống có thể tìm kiếm.
Vì việc tính toán Chamfer Distance (khoảng cách giữa 600 vector của 2 bức ảnh) là cực kỳ tốn kém, Stage 3 sẽ tính toán trước toàn bộ khoảng cách nội bộ giữa các ảnh trong Database với nhau và lưu vào một từ điển (Sparse Lookup).
Khi có một ảnh Query từ ngoài vào (ở Stage 4 và 5), chúng ta chỉ cần tính Chamfer cho ảnh Query, còn khoảng cách giữa các ảnh Database với nhau sẽ được lôi ra từ Từ điển này, giúp giảm thời gian chạy từ "Vài tiếng" xuống còn "Vài giây".

## 📥 Dữ liệu Đầu vào (Input)
*   Thư mục đặc trưng cục bộ (Stage 1): `output/stage1/features/.../database/`

## 📤 Dữ liệu Đầu ra (Output)
*   Từ điển dạng pickle: `output/stage3/roxford5k_sparse_sim.pkl` (hoặc rparis6k_sparse_sim.pkl).

## 🚀 Cách chạy (How to run)

Từ thư mục gốc dự án, chạy lệnh:

```bash
# Chạy với Backend Tự động (Ưu tiên CANN nếu có, không có dùng PyTorch)
python src/offline/stage3_build_index/build_index.py --dataset roxford5k --backend auto

# Hoặc ép buộc chạy bằng PyTorch GPU (Mini-batching 200 ảnh/lần)
python src/offline/stage3_build_index/build_index.py --dataset roxford5k --backend pytorch
```
