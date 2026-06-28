# 🌊 Luồng Hoạt Động (Data Flow) Của Kiến Trúc L2G

Tài liệu này trình bày luồng hoạt động từ Tổng quan đến Chi tiết của hệ thống truy xuất hình ảnh (Image Retrieval) dựa trên bài báo L2G. Luồng hệ thống được chia làm 2 pha chính: **Offline** (Xây dựng dữ liệu) và **Online** (Truy vấn ảnh).

---

## 🛑 PHẦN 1: TỔNG QUAN (HIGH-LEVEL OVERVIEW)

Hệ thống hoạt động theo nguyên lý "Phễu Lọc":
1. **Offline Indexing:** Chuyển đổi hàng triệu bức ảnh thô trong kho dữ liệu thành các vector toán học (Features) để máy tính hiểu được. Sau đó sắp xếp các vector này thành một mạng lưới "đường hầm" (Indexing) để truy tìm cực nhanh.
2. **Online Search:** Khi có 1 bức ảnh mới được tải lên, hệ thống sẽ số hóa nó thành vector, thả vào "đường hầm" để lấy ra Top 1600 ảnh giống nhất. Cuối cùng dùng công thức toán học ma trận (Re-ranking) để soi kỹ 1600 ảnh này và xếp hạng chuẩn xác nhất.

---

## 🛠️ PHẦN 2: CHI TIẾT GIAI ĐOẠN OFFLINE (Chuẩn Bị Cơ Sở Dữ Liệu)

Ở giai đoạn này, hệ thống sẽ xử lý toàn bộ Database (Ví dụ 5062 ảnh của tập ROxford5k).

### Bước 1: Trích xuất đặc trưng Cục Bộ (Local Features)
* **Code thực thi:** `src/stage1_extract_local.py`
* **Nguồn Input:** Đọc trực tiếp các file ảnh gốc `.jpg` từ thư mục `data/datasets/roxford5k/jpg/`.
* **Cách hoạt động:** 
  Từng bức ảnh sẽ được đưa vào mạng Neural Network **FIRe** (`fire/`). Mạng này dùng thuật toán Super-Features quét qua bức ảnh, tìm ra các điểm nổi bật nhất (keypoints) và mã hóa thành một ma trận đặc trưng.
* **Đầu ra (Output):** Các ma trận có kích thước `[600, 128]` (600 vector 128 chiều). 
* **Nơi lưu trữ:** Lưu thành các file `.npy` tại thư mục `output/stage1/features/roxford5k/database/`.

### Bước 2: Trích xuất đặc trưng Toàn Cục (Global Features)
* **Code thực thi:** `src/stage2_extract_global.py`
* **Nguồn Input:** Cũng là file ảnh gốc `.jpg` từ thư mục `data/datasets/...`.
* **Cách hoạt động:** 
  Đưa bức ảnh vào mạng **CVNet ResNet-101** để nhìn bức ảnh dưới góc độ bao quát (tổng thể cảnh vật xung quanh) thay vì từng chi tiết nhỏ. 
* **Đầu ra (Output):** Trả về 1 vector duy nhất `[2048]` chiều.
* **Nơi lưu trữ:** Lưu thành các file `.npy` tại `output/stage2/features/roxford5k/database/`.

### Bước 3: Đóng Gói (Indexing) Base Search bằng CANN
*(Trong code hiện tại, quá trình này nằm ẩn bên trong hàm `cann_search` của `stage3_rerank.py` khi nó nạp Database)*
* **Cách hoạt động:** 
  Toàn bộ 5062 file `.npy` (đặc trưng Cục bộ) từ Bước 1 sẽ được gỡ bỏ đuôi Numpy, biến thành file **nhị phân nguyên thủy `.desc`** (Raw Binary Float32). Các file `.desc` này sẽ được đẩy cho file chạy `colored_c_nn_random_grids_index_main.exe` của Google. 
* **Đầu ra (Output):** Một cấu trúc tra cứu nhị phân trong bộ nhớ gọi là *Random Grids Tree*. Nhờ cấu trúc này, việc dò tìm sau này chỉ tốn vài mili-giây.

---

## 🔎 PHẦN 3: CHI TIẾT GIAI ĐOẠN ONLINE (Truy Vấn - Inference)

Khi người dùng Upload 1 tấm ảnh (Query Image), đây là cách dữ liệu chảy từ đầu đến cuối:

### Bước 1: Tiền xử lý ảnh Query
* Nó sẽ phải chạy qua mạng **FIRe** (giống hệt Offline Bước 1) để lấy ra File ma trận `[600, 128]` cục bộ. Lại chạy tiếp qua **CVNet** (giống hệt Offline Bước 2) để lấy ra Vector toàn cục `[2048]`.

### Bước 2: Tìm kiếm Thô (Base Search)
* **Code thực thi:** Hàm `cann_search()` trong `src/run_cann_search.py`
* **Cách hoạt động:** Lấy ma trận `[600, 128]` của ảnh Query, đẩy cho file `colored_c_nn_random_grids_index_main.exe` để nó quăng vào cái *Random Grids Tree* đã xây dựng sẵn ở phần Offline. 
* **Đầu ra (Output):** CANN sẽ nhả ra 1 file text tên là `pairs.txt` lưu trên ổ đĩa nháp (temp). File này chứa tên của **Top 1600** bức ảnh Database giống nhất kèm theo điểm số xấp xỉ của nó. 

### Bước 3: Soi Chi Tiết (Exact Chamfer Distance)
* **Code thực thi:** Nằm bên trong vòng lặp for của `src/stage3_rerank.py`
* **Cách hoạt động:** 
  Hệ thống bỏ qua 3462 ảnh kia, chỉ tập trung vào 1600 ảnh do Bước 2 trả về (cộng thêm 1 ảnh Query = 1601 ảnh). Nó gọi PyTorch tạo ra một ma trận GPU khổng lồ `1601 x 1601` và thực hiện hàm `compute_chamfer_matrix_pytorch()`. Nó so khớp từng điểm trong 600 điểm của ảnh A với 600 điểm của ảnh B.
* **Đầu ra (Output):** Đưa ra một Ma Trận Khoảng Cách cực kỳ chính xác gọi là `D_mod` kích thước `[1601, 1601]`.

### Bước 4: Giảm Chiều (MDS)
* **Code thực thi:** Gọi thư viện `MDS` của `scikit-learn` trong `stage3_rerank.py`.
* **Cách hoạt động:** Thuật toán SMACOF sẽ cố gắng ép cái ma trận `[1601, 1601]` cồng kềnh kia xuống thành một dạng biểu diễn nhỏ gọn hơn.
* **Đầu ra (Output):** Thu được ma trận giảm chiều `F_mds` kích thước `[1601, 128]`.

### Bước 5: Hợp Nhất (Fusion)
* **Cách hoạt động:** Kết hợp góc nhìn "Chi tiết" (MDS) và góc nhìn "Tổng quát" (CVNet).
  Hệ thống lấy đặc trưng Global `[2048]` của 1601 tấm ảnh này (đã cất ở thư mục `output/stage2/` từ lúc Offline) và **Nối (Concatenate)** nó chung với cái `F_mds [128]`.
* **Đầu ra (Output):** Tạo ra 1601 vector siêu bự kích thước `[2176]` (vì 2048 + 128 = 2176). Cái này gọi là `F_concat`.

### Bước 6: Xếp Hạng Lại (SuperGlobal Re-ranking)
* **Code thực thi:** Hàm `superglobal_reranking_full()` trong `stage3_rerank.py`.
* **Cách hoạt động:** Đưa 1601 cái vector siêu bự kia vào một mạng đồ thị (Database-side Augmentation/Graph). Tại đây, điểm số của các ảnh lân cận sẽ bổ trợ cho nhau để chốt hạ vị trí cuối cùng.
* **Đầu ra (Output):** Một danh sách `final_ranks` xếp hạng từ số 1 đến 5000. 
* **Lưu Trữ Cuối Cùng:** Kết quả bảng xếp hạng được nén thành file `roxford5k_ranks.npy`. Hệ thống đọc danh sách này, mang đi đối chiếu với Đáp án (Ground Truth) để tính ra tỷ lệ chính xác mAP rồi ghi vào file text `roxford5k_final_results.txt`. Đã xong!
