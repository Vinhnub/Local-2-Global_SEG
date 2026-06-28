# 📂 Cấu Trúc Mã Nguồn Dự Án L2G (Local To Global)

Tài liệu này giải thích chi tiết toàn bộ các thư mục và file có trong hệ thống mã nguồn `main`. Nó cung cấp cái nhìn tổng quan về nguồn gốc, nội dung bên trong, và mục đích của từng thành phần để ông dễ dàng bảo trì và mở rộng trong tương lai.

---

## 1. Thư mục `src/` (Trái tim của dự án)
Đây là thư mục do chúng ta tự viết 100%, chứa luồng chạy chính (pipeline) kết nối tất cả các thuật toán AI lại với nhau.

* **`stage1_extract_local.py`**
  * **Nguồn gốc:** Tự lập trình.
  * **Bên trong có gì:** Mã nguồn load mô hình mạng Neural **FIRe** (`fire_SfM_120k.pth`), đọc hàng ngàn tấm ảnh từ thư mục `data/`, cắt bớt kích thước ảnh nếu quá lớn, và trích xuất đặc trưng cục bộ (Local Features).
  * **Mục đích:** Tạo ra các file `.npy` chứa 600 vector x 128 chiều cho mỗi ảnh. Bước này chuẩn bị nguyên liệu đầu vào cho CANN.
* **`stage2_extract_global.py`**
  * **Nguồn gốc:** Tự lập trình (gọi hàm từ thư mục SuperGlobal).
  * **Bên trong có gì:** Mã nguồn load mô hình mạng **CVNet_R101/SuperGlobal**, chạy ảnh qua mạng ResNet-101 để trích xuất đặc trưng toàn cục (Global Features).
  * **Mục đích:** Tạo ra file `.npy` chứa 1 vector x 2048 chiều cho mỗi ảnh. Dùng cho bước hợp nhất (Fusion) ở cuối Stage 3.
* **`stage3_rerank.py`**
  * **Nguồn gốc:** Tự lập trình (logic lõi bám sát L2G paper).
  * **Bên trong có gì:** Mã nguồn mô phỏng lại luồng Benchmarking cực kỳ đồ sộ: gọi `cann_search` để lấy Top 1600, dùng PyTorch để tính toán Exact Chamfer Distance trên GPU, dùng Scikit-learn để chạy giảm chiều MDS, và tính toán công thức gộp điểm số với Global features. Cuối cùng tự động chấm điểm mAP cho 3 mức Easy/Medium/Hard.
  * **Mục đích:** Chấm điểm độ chính xác (Benchmarking) và xác thực toàn bộ pipeline L2G.
* **`run_cann_search.py`**
  * **Nguồn gốc:** Tự lập trình (làm cầu nối Python - C++).
  * **Bên trong có gì:** Code chuyển đổi file đặc trưng `.npy` thành file nhị phân `raw bytes (.desc)` mà C++ có thể đọc được. Kế tiếp, nó gọi file `colored_c_nn_random_grids_index_main.exe` để chạy thuật toán tìm kiếm xấp xỉ Random Grids, sau đó đọc file kết quả `pairs.txt` trả về.
  * **Mục đích:** Bao bọc (Wrapper) mã nguồn C++ khó nhằn của Google thành một hàm Python đơn giản dễ dùng cho `stage3`.

## 2. Thư mục `google-research/`
Kho chứa mã nguồn nguyên thủy của Google Research.

* **`cann/`** (Constrained Approximate Nearest Neighbors)
  * **Nguồn gốc:** Clone trực tiếp từ Github chính thức của Google Research (đính kèm trong paper L2G).
  * **Bên trong có gì:** Hàng loạt file C++ (`cann_rg.cc`, `colored_c_nn_random_grids_index_main.cc`,...).
  * **Mục đích:** Đây là thuật toán Indexing & Base Search tốc độ cao. Dùng thuật toán "Random Grids" để lọc nhanh Top 1600 bức ảnh từ Database 5000+ bức ảnh.
* **`asmk/`** (Aggregated Selective Match Kernel)
  * **Nguồn gốc:** Code cũ dùng để thay thế CANN trước đó.
  * **Mục đích:** (Hiện tại đã bị thay thế bởi CANN, giữ lại để phòng hờ dự phòng/so sánh).

## 3. Thư mục `fire/`
* **Nguồn gốc:** Clone từ Github của thuật toán FIRe (Learning Super-Features for Image Retrieval - ICLR 2022).
* **Bên trong có gì:** Mã nguồn PyTorch định nghĩa mạng nơ-ron FIRe, các hàm lấy features, và file trọng số mô hình `fire_SfM_120k.pth`.
* **Mục đích:** Mạng AI chuyên biệt để trích xuất đặc trưng cục bộ (Local Features) chuẩn xác nhất thế giới hiện tại.

## 4. Các file gốc ở ngoài `main/`
* **`CVPR2022_CVNet_R101.pyth` & `CVPR2022_CVNet_R50.pyth`**
  * **Nguồn gốc:** Trọng số (Weights) tải về từ dự án CVNet / SuperGlobal.
  * **Mục đích:** Chứa tri thức (trọng số Neural Network) đã được train trên hàng triệu bức ảnh để dùng cho `stage2_extract_global.py`.
* **`bazel.exe` & `vs_buildtools.exe`**
  * **Nguồn gốc:** Tải về từ Internet trong lúc cài đặt.
  * **Bên trong có gì:** File cài đặt trình biên dịch C++ (MSVC) và hệ thống build của Google (Bazel).
  * **Mục đích:** Công cụ trung gian bắt buộc phải có trên Windows để có thể compile được cái thư mục `google-research/cann/` thành file chạy `.exe`.
* **`METHOD.md` / `sg_readme.md`**
  * **Nguồn gốc:** File tài liệu.
  * **Mục đích:** Chứa các lý thuyết, hướng dẫn sử dụng và toán học liên quan đến SuperGlobal.

## 5. Các Thư Mục Dữ Liệu
* **`data/datasets/`**
  * **Bên trong có gì:** Chứa toàn bộ file ảnh `.jpg` của các tập ROxford5k và RParis6k, cộng với các file `.pkl` chứa nhãn (Ground Truth).
* **`output/`**
  * **Bên trong có gì:** Nơi lưu trữ toàn bộ các file `.npy` đặc trưng đã trích xuất từ FIRe (nằm trong `output/stage1/`) và CVNet (nằm trong `output/stage2/`). Đặc biệt thư mục `output/stage3/` chứa các file text `.txt` báo cáo kết quả điểm số mAP cuối cùng.
* **`scratch/`**
  * **Bên trong có gì:** Chứa các file script Python gỡ lỗi tạm thời (VD: `edit_stage3.py`, `fix_imports.py`) được tạo ra trong quá trình tinh chỉnh mã nguồn.

---
**Tổng Kết Lại:** 
Luồng dữ liệu (Data Flow) sẽ đi từ `data/` ➡️ qua `stage1` (dùng `fire/`) và `stage2` (dùng CVNet) để sinh ra file `.npy` vào `output/` ➡️ Cuối cùng, `stage3_rerank.py` ném các file `.npy` này vào `google-research/cann` (thông qua `run_cann_search.py`), tính toán trên GPU và chốt kết quả!
