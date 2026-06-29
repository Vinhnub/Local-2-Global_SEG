# 🧪 Stage 5: Real-time API Inference (Test Query)

## 📌 Chức năng (What it does)
Đây là môi trường thử nghiệm **End-to-End** sát với thực tế nhất. Tưởng tượng bạn đang code 1 Website hoặc 1 App Mobile, khi User bấm nút "Tải ảnh lên và Tìm kiếm", đây chính xác là Script mà hệ thống Backend sẽ chạy.

Nó lấy bức ảnh của bạn đi xuyên qua tất cả các mô hình, so sánh trong 0.5s và tự động lưu 20 bức ảnh giống nhất vào thư mục để bạn xem.

## 🛠 Lệnh thực thi:
```bash
python src/online/stage5_test_query/test_query.py --image data/datasets/roxford5k/jpg/all_souls_000000.jpg
```

## 🎯 Mô tả & Ví dụ đầu ra (What you get):
Terminal sẽ hiển thị quy trình của API:
```text
==================================================
INITIALIZING REAL-TIME SEARCH ENGINE (L2G)
==================================================
Loading FIRe Model (Local)...
Loading CVNet Model (Global)...
Loading ASMK Index & Database...
Engine Ready!
==================================================
PROCESSING QUERY IMAGE: data\datasets\roxford5k\jpg\all_souls_000000.jpg
  [ASMK] Found top 100 candidates in 0.05s
  [Chamfer + MDS] Local distance matrix computed in 0.12s
  [SuperGlobal] Reranking graph convergence in 0.08s
SEARCH COMPLETED IN 0.25s!

Top 20 Matches:
  01. all_souls_000072
  02. all_souls_000045
  03. all_souls_000102
...
Results saved to C:\Users\ezycloudx-admin\Desktop\seg\main\output_test
```

### 🧠 Phân tích kết quả:
- **`Engine Ready!`**: Website/API đã khởi động xong, nạp 3 Model khổng lồ vào RAM, sẵn sàng chiến đấu. (Lần chạy đầu tiên sẽ tốn 5-10s nạp Model, ở hệ thống thật, Model luôn nằm trong RAM chờ).
- **`0.25s`**: Tổng thời gian từ lúc ném ảnh vào mạng tới lúc trả về danh sách xếp hạng. Quá nhanh!
- **`output_test/`**: Bạn chỉ việc mở thư mục này ra, nó chứa bức ảnh truy vấn (mang tên `00_QUERY.jpg`) và 20 bức ảnh kết quả (mang tên `rank_01.jpg` đến `rank_20.jpg`). Việc đối chiếu bằng mắt thường chưa bao giờ dễ dàng như thế.

---

## 🧮 Cấu hình siêu tham số (Hyperparameters)
- $K = 1600$: Lọc ra 1600 ứng viên ban đầu bằng ASMK (Theo đúng cấu hình chuẩn của Paper để đạt mAP cao nhất).
- Lưới trọng số Fusion: Trộn $w_{local} = 0.19$ và $w_{global} = 0.81$ (Theo chuẩn của Paper SuperGlobal) để đưa vào đồ thị.
- Tham số lan truyền (Diffusion): $\beta = 0.31$ và kích thước đồ thị $k=6$ (Mỗi nút ảnh sẽ truyền điểm cho 6 hàng xóm gần nó nhất).
