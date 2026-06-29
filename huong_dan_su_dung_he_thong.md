# 📖 Bách Khoa Toàn Thư Hệ Thống L2G (Local-to-Global Image Retrieval)

Tài liệu này cung cấp cái nhìn toàn cảnh (Bird-eye view), luồng luân chuyển dữ liệu (Data Flow) và hướng dẫn từ A-Z để cài đặt, cấu hình và vận hành hệ thống truy xuất hình ảnh siêu tốc (L2G).

---

## 🏗️ 1. Cấu trúc Kiến trúc (Architecture Overview)

Hệ thống được chia thành 2 chu trình làm việc tách biệt và chuyên nghiệp: **Offline** (Chuẩn bị Database) và **Online** (Đón nhận tìm kiếm từ User).
Cấu trúc cây thư mục chuẩn:
```text
C:\Users\ezycloudx-admin\Desktop\seg\main\
├── output/                  # Chứa Features (.npy) và Index (.pkl) sau khi chạy Offline
├── output_test/             # Chứa Top 20 ảnh trực quan xuất ra khi chạy Test Query
├── data/datasets/           # Chứa ảnh thô (jpg) và file Label (pkl)
├── src/
│   ├── offline/             # Mã nguồn chạy Offline (Chỉ chạy 1 lần)
│   ├── online/              # Mã nguồn chạy Online (Phục vụ Search Engine)
│   ├── core/                # Thư viện gốc của FIRe và SuperGlobal
│   └── utils/               # Công cụ đánh giá điểm mAP
└── huong_dan_su_dung_he_thong.md
```

---

## 🔄 2. Chu trình OFFLINE (Xây dựng Database)

Tưởng tượng bạn vừa có được 1 Triệu bức ảnh. Bạn không thể tìm kiếm chay trên 1 triệu tấm này. Bạn phải số hóa chúng.
👉 **Chu trình Offline CHỈ CẦN chạy 1 lần duy nhất!**

### Bước 2.1: FIRe Local Feature
Số hóa các điểm nhỏ nhặt của bức ảnh.
- **Lệnh:** `python src/offline/stage1_local_extract/extract_local.py`
- **Đầu ra:** Mỗi tấm ảnh biến thành 1 file Numpy kích thước `600x128`. Thể hiện 600 đặc điểm của vật thể (ví dụ: góc bàn, logo, cái ly).

### Bước 2.2: CVNet Global Feature
Số hóa toàn cảnh bức ảnh.
- **Lệnh:** `python src/offline/stage2_global_extract/extract_global.py`
- **Đầu ra:** Mỗi tấm ảnh biến thành 1 chuỗi dài `1x2048`. Thể hiện bối cảnh chung, màu sắc, không khí của bức ảnh.

### Bước 2.3: Xây dựng Từ điển ASMK
Nếu có 1 triệu ảnh, bạn sẽ có 600 triệu đặc điểm cục bộ. Máy tính sẽ bị tràn RAM nếu tính ma trận. ASMK giải quyết vấn đề này.
- **Lệnh:** `python src/offline/stage3_build_index/build_index.py`
- **Hoạt động:** Gom 600 triệu điểm thành `65,536` Từ Vựng (Centroids). Sau đó nhét các bức ảnh vào hệ thống tra cứu IVF.
- **Đầu ra:** File `_codebook.pkl` và `_ivf.pkl`. Từ giờ việc tìm kiếm sẽ siêu tốc như lật từ điển.

---

## ⚡ 3. Chu trình ONLINE (Tìm kiếm Thực tế)

Sau khi có "Từ điển" và "Database", hệ thống đóng vai trò như một bộ máy Google Hình Ảnh. Đã sẵn sàng đón nhận truy vấn từ người dùng.

### Tính năng 3.1: Test API End-to-End (Thử bằng Mắt Thường)
Nhét 1 tấm ảnh Query vào, máy sẽ tìm ra 20 tấm giống nhất.
- **Lệnh:**
  ```bash
  python src/online/stage5_test_query/test_query.py --image "đường_dẫn_tới_ảnh.jpg"
  ```
- **Kết quả:** Xử lý mất **0.2s**. Hãy mở thư mục `output_test/`, bạn sẽ thấy ảnh Truy vấn của mình và 20 bức ảnh kết quả nằm kề nhau. Quá tuyệt vời để thuyết trình demo cho sếp/khách hàng.

### Tính năng 3.2: Đánh giá mAP (Dành cho nhà nghiên cứu viết Báo cáo)
Chạy toàn bộ 70 truy vấn chuẩn thế giới cùng 1 lúc và tính điểm số. Có 2 hệ phái:

**A. Hệ phái Thực tiễn (Dành cho Website - Siêu Tốc)**
Tốc độ $0.5s$/ảnh. Dùng ASMK + SuperGlobal. mAP khoảng 88%.
- **Lệnh:** `python src/online/stage4_search/search_fast_asmk.py`

**B. Hệ phái Hàm Lâm (Dành cho Paper - Vét Cạn)**
Tốc độ $30s$/ảnh. Tính toán mọi điểm ma trận. mAP đạt đỉnh 92.2%.
- **Lệnh:** `python src/online/stage4_search/search_exact_chamfer.py`

---

## 🧠 4. Giải phẫu thuật toán Reranking (The Mathematics)

Sức mạnh ma thuật của hệ thống nằm ở khâu kết hợp Stage 4 / Stage 5 (Fusion).

1. **Exact Chamfer Distance (Khoảng cách Bề Mặt):**
   Hệ thống chiếu các tập hợp 600 điểm ảnh (Của bức Query $Q$ và Database $X$) và tìm độ giống nhau lớn nhất (Max Pool):
   $$ S_{local} = \frac{1}{2|Q|} \sum_{q \in Q} \max_{x \in X}(q \cdot x) + \frac{1}{2|X|} \sum_{x \in X} \max_{q \in Q}(q \cdot x) $$

2. **Dung hợp (Fusion) bằng Đa Hướng (MDS):**
   Ma trận $S_{local}$ khá rối rắm, nên toán học áp dụng MDS (Multi-Dimensional Scaling) ép ma trận đó về lại vector 128 chiều (Gọi là $F_{mds}$).
   Nối nó với vector Toàn cảnh của CVNet ($F_{global}$):
   $$ F_{concat} = [ \sqrt{0.19} \times F_{mds}, \sqrt{0.81} \times F_{global} ] $$

3. **Truyền Lan Đồ Thị (SuperGlobal Graph):**
   Thay vì chỉ sắp xếp kết quả, hệ thống xây một mạng lưới.
   - Nếu bạn tìm cái Tháp Eiffel, ảnh A là chân tháp, ảnh B là ngọn tháp.
   - Query rất giống ảnh A.
   - Ảnh A lại rất giống ảnh B trong DB.
   $\Rightarrow$ Hệ thống sẽ lấy điểm của A truyền bớt sang B với hệ số hao hụt $\beta = 0.31$. Nhờ vậy B cũng được lọt vào top tìm kiếm!
