# Local-to-Global (L2G) Image Retrieval Pipeline

Đây là mã nguồn chính thức cho hệ thống Image Retrieval sử dụng phương pháp **Local-to-Global (L2G)**. Hệ thống này kết hợp sức mạnh của đặc trưng cục bộ (Local Features) và đặc trưng toàn cục (Global Features), sau đó tối ưu hóa kết quả bằng thuật toán **Graph Diffusion** trên đồ thị K-Nearest Neighbors (KNN).

---

## 📖 Giải thích các thuật ngữ & Ký hiệu trong dự án

Để bạn dễ hiểu luồng đi của mã nguồn, dưới đây là các thuật ngữ và ký hiệu toán học được sử dụng trong code:

*   **L2G (Local-to-Global):** Là chiến lược tìm kiếm 2 bước. Đầu tiên, dùng đặc trưng cục bộ (Local) để lấy ra một danh sách ngắn các ảnh ứng viên tiềm năng. Sau đó, dùng đặc trưng toàn cục (Global) kết hợp với Local để xếp hạng lại (Re-ranking) danh sách đó.
*   **FIRe:** Mạng nơ-ron trích xuất đặc trưng cục bộ. Mỗi bức ảnh sẽ được FIRe trích xuất ra 600 vector nhỏ, mỗi vector có kích thước 128 chiều.
*   **CVNet:** Mạng nơ-ron trích xuất đặc trưng toàn cục. Đại diện cho toàn bộ nội dung của bức ảnh bằng 1 vector duy nhất có kích thước 2048 chiều.
*   **Exact Chamfer Distance:** Thuật toán tính khoảng cách giữa 2 tập hợp các đặc trưng cục bộ (so sánh 600 vector của ảnh A với 600 vector của ảnh B).
*   **MDS (Multidimensional Scaling):** Thuật toán nén khoảng cách. Từ ma trận khoảng cách Chamfer của các ảnh, MDS ép chúng lại thành những vector có số chiều cố định (ở đây là 128 chiều) để dễ dàng ghép với các đặc trưng khác.
*   **Graph Diffusion (MDescAug):** Thuật toán lan truyền đồ thị. Cập nhật đặc trưng của một bức ảnh bằng cách "hấp thụ" đặc trưng từ các hàng xóm gần nhất của nó, giúp khử nhiễu.
*   **K (số hàng xóm):** Số lượng hàng xóm gần nhất trên đồ thị KNN (thường K = 6).
*   **$\beta$ (beta):** Trọng số lan truyền. Quyết định việc bức ảnh sẽ "tin tưởng" hàng xóm bao nhiêu phần trăm (thường $\beta = 0.31$).
*   **$M$:** Số lượng ảnh ứng viên được đưa vào bước Re-ranking (thường M = 1600).
*   **$w_{local}$, $w_{global}$:** Trọng số khi hợp nhất đặc trưng Local (0.19) và Global (0.81).
*   **$F_{concat}$:** Vector đặc trưng cuối cùng, được nối (concatenate) từ MDS (Local) và CVNet (Global).

---

## ⚙️ Kiến trúc Pipeline (3 Stages)

Dự án được chia làm 3 giai đoạn (Stage) độc lập để dễ dàng theo dõi và gỡ lỗi:

### Stage 1: Trích xuất đặc trưng cục bộ (Local Features)
*   **Mã nguồn:** `src/stage1_extract_local.py`
*   **Đầu vào:** Thư mục ảnh (ví dụ: `roxford5k/jpg/`).
*   **Xử lý:** Chạy mô hình FIRe.
*   **Đầu ra:** Các tệp `.npy` chứa mảng kích thước `(600, 128)`.

### Stage 2: Trích xuất đặc trưng toàn cục (Global Features)
*   **Mã nguồn:** `src/stage2_extract_global.py`
*   **Đầu vào:** Thư mục ảnh.
*   **Xử lý:** Chạy mô hình CVNet-R101.
*   **Đầu ra:** Các tệp `.npy` chứa mảng kích thước `(2048,)`.

### Stage 3: Hợp nhất (Fusion) & Lan truyền đồ thị (Graph Diffusion)
*   **Mã nguồn:** `src/stage3_rerank.py`
*   **Quy trình chi tiết:**
    1.  **Base Search:** So sánh Query với toàn bộ Database bằng Exact Chamfer Distance để lấy ra top `M` ảnh ứng viên.
    2.  **MDS Embedding:** Biến đổi ma trận khoảng cách Chamfer của `M` ảnh này thành các vector cục bộ $F_{mds}$ có độ dài 128.
    3.  **L2G Fusion:** Nối $F_{mds}$ với đặc trưng toàn cục $F_{global}$ (độ dài 2048) theo trọng số $w$ để tạo ra $F_{concat}$ (độ dài 2176).
    4.  **Graph Diffusion:** Xây dựng đồ thị KNN trên tập $F_{concat}$. Cập nhật lại đặc trưng của mỗi bức ảnh bằng cách lấy trung bình có trọng số với K hàng xóm của nó.
    5.  **Final Ranking:** Dùng đặc trưng sau khi Diffusion để tính độ tương đồng (Dot-product) với Query và xếp hạng lần cuối.

---

## 🚀 Hướng dẫn chạy thử nghiệm

### 1. Tải Dữ liệu & Trọng số Mô hình (Models)
Do giới hạn dung lượng của GitHub, toàn bộ dữ liệu ảnh và trọng số model đã được đóng gói và lưu trữ trên Cloud. Bạn cần tải về trước khi chạy:
- **Tải Datasets (roxford5k, rparis6k):** [Link Google Drive của bạn] -> Giải nén vào thư mục `data/datasets/`
- **Tải Weights (FIRe & CVNet):** [Link Google Drive của bạn] -> Giải nén tương ứng vào `fire/model/` và `CVNet/`

### 2. Chạy Pipeline
Dự án có sẵn một script tự động hóa chạy lần lượt cả 3 stage và xuất ra kết quả mAP.

```bash
# Mở terminal (sử dụng môi trường conda chứa PyTorch)
python src/run_pipeline.py
```

Kết quả (mAP) sẽ được in trực tiếp ra màn hình và lưu chi tiết tại `output/stage3/`.

---

## 📁 Cấu trúc thư mục mã nguồn

```text
main/
├── src/                          
│   ├── run_pipeline.py           # Script tự động chạy 3 stage
│   ├── stage1_extract_local.py   # Code trích xuất FIRe
│   ├── stage2_extract_global.py  # Code trích xuất CVNet
│   └── stage3_rerank.py          # Code thuật toán L2G & Graph Diffusion
├── fire/                         # Model và code của mạng FIRe
├── CVNet/                        # Model và code của mạng CVNet 
├── SuperGlobal-main/             # Các mô-đun Re-ranking bổ trợ
├── data/                         # Nơi chứa dataset (roxford5k, rparis6k)
└── output/                       # Thư mục lưu file .npy và kết quả đánh giá
```
