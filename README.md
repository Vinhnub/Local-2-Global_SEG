# Local-to-Global (L2G) Image Retrieval Pipeline

Đây là mã nguồn chính thức cho hệ thống Image Retrieval sử dụng phương pháp **Local-to-Global (L2G)**. Hệ thống kết hợp đặc trưng cục bộ (Local Features) và đặc trưng toàn cục (Global Features), sau đó tối ưu hóa kết quả bằng thuật toán **Graph Diffusion** trên đồ thị K-Nearest Neighbors (KNN).

Đặc biệt, hệ thống đã được tối ưu hóa để chạy **hoàn toàn bằng Python/PyTorch trên GPU** với kỹ thuật Mini-batching siêu nhẹ (<1GB VRAM), giúp bạn không cần phải tốn công cài đặt các môi trường C++ phức tạp như phiên bản gốc.

---

## 🚀 Hướng dẫn chạy từng Stage (Step-by-Step)

Dự án này được thiết kế theo dạng module, bạn cần chạy lần lượt từng Stage từ 1 đến 5 để có kết quả cuối cùng.

### 🗄️ OFFLINE (Chuẩn bị Dữ liệu)

**Stage 1: Trích xuất đặc trưng cục bộ (Local Features)**
*   Chạy mô hình **FIRe**. Mỗi bức ảnh được trích xuất thành 600 vector x 128 chiều.
*   **Lệnh chạy:**
    ```bash
    python src/offline/stage1_local_extract/extract_local.py
    ```

**Stage 2: Trích xuất đặc trưng toàn cục (Global Features)**
*   Chạy mô hình **CVNet-R101**. Mỗi bức ảnh được nén thành 1 vector duy nhất x 2048 chiều.
*   **Lệnh chạy:**
    ```bash
    python src/offline/stage2_global_extract/extract_global.py
    ```

**Stage 3: Xây dựng Từ điển (Build Index)**
*   Hệ thống tính toán trước ma trận khoảng cách Chamfer (Sparse Lookup) giữa các ảnh trong Database để tiết kiệm thời gian tra cứu sau này.
*   **Lệnh chạy:** 
    ```bash
    python src/offline/stage3_build_index/build_index.py --dataset roxford5k --backend auto
    ```

### ⚡ ONLINE (Tìm kiếm Thực tế)

**Stage 4: Chấm điểm độ chính xác (mAP Evaluation)**
*   Hệ thống sẽ chạy qua toàn bộ tập Query để tính toán và chấm điểm mAP (Easy, Medium, Hard).
*   **Lệnh chạy:**
    ```bash
    python src/online/stage4_search/search_exact_chamfer.py --backend auto
    ```

**Stage 5: Truy vấn Ảnh Thực Tế (Test Query)**
*   Đưa vào 1 bức ảnh bất kỳ từ ngoài đời thực, hệ thống sẽ trả về top 20 ảnh giống nhất trong kho dữ liệu và lưu ảnh trực quan ra thư mục `output_test/`.
*   **Lệnh chạy:**
    ```bash
    python src/online/stage5_test_query/test_query.py --image <đường_dẫn_tới_ảnh.jpg> --backend auto
    ```

---

## 🛠️ Tùy chọn Công cụ Tìm kiếm (Backend)

Khâu nặng nề nhất của hệ thống là **Base Search** (Lọc thô). Ở Stage 3, Stage 4 và Stage 5, bạn có thể truyền thêm cờ `--backend`:

1. `--backend auto` (Mặc định): Tự động thông minh. Nếu tìm thấy CANN (.exe) thì dùng CANN, nếu không có thì tự động dùng PyTorch GPU.
2. `--backend pytorch`: **Khuyên dùng**. Sử dụng Pytorch với kỹ thuật **Mini-batching**. Chạy chính xác tuyệt đối 100% bằng GPU, siêu nhẹ (< 1GB VRAM). Hoạt động hoàn hảo cho dataset 5000-6000 ảnh.
3. `--backend cann`: Chạy bằng mã nguồn gốc C++ của Google. Yêu cầu bạn phải cài đặt **Microsoft C++ Build Tools** và **Bazel** để tự compile file `.exe`.

---

## 📁 Cấu trúc thư mục mã nguồn

```text
Local-2-Global_SEG/
├── src/                          
│   ├── offline/                  # Các bước chuẩn bị Database
│   │   ├── stage1_local_extract/
│   │   ├── stage2_global_extract/
│   │   └── stage3_build_index/
│   ├── online/                   # Các bước Tìm kiếm thực tế
│   │   ├── stage4_search/        
│   │   └── stage5_test_query/    
├── fire/                         # Model và code của mạng FIRe
├── CVNet/                        # Model và code của mạng CVNet 
├── google-research/              # Chứa mã nguồn gốc của Google (CANN)
├── data/                         # Nơi chứa dataset ảnh gốc
└── output/                       # Thư mục lưu file đặc trưng và kết quả
```
