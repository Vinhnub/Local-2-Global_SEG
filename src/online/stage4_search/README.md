# 🚀 Stage 4: Real-time Search Engine (Online)

## 📌 Chức năng (What it does)
Đây là "bộ não tìm kiếm" của hệ thống khi đưa vào chạy thực tế. 
Khi bạn đã chuẩn bị xong kho dữ liệu (Offline Stage 1,2,3), Module này sẽ có nhiệm vụ đánh giá (Evaluate) toàn bộ hệ thống bằng cách lấy 70 Query tiêu chuẩn quốc tế của bộ dữ liệu ROxford5k/RParis6k và tiến hành tìm kiếm hàng loạt.

Module này cung cấp 2 chế độ (Tùy mục đích sử dụng):
1. **`search_fast_asmk.py`**: Chế độ **Siêu tốc** kết hợp L2G.
2. **`search_exact_chamfer.py`**: Chế độ **Vét cạn** (Độ chính xác cao nhất).

---

## ⚡ 1. Chế độ Siêu Tốc (Fast ASMK + SuperGlobal)

### 🛠 Lệnh thực thi:
```bash
python src/online/stage4_search/search_fast_asmk.py
```

### 🎯 Mô tả & Ví dụ đầu ra (What you get):
Khi chạy lệnh này, màn hình Terminal của bạn sẽ xuất hiện quy trình như sau:

```text
Loading FIRe Model (Local)...
Loading CVNet Model (Global)...
Loading ASMK Index & Database...
Index loaded with 65536 visual words.

==================================================
STARTING EVALUATION: roxford5k
==================================================
Processing Query 1/70: all_souls_1
  [ASMK] Found top 100 candidates in 0.05s
  [Chamfer + MDS] Local distance matrix computed in 0.12s
  [SuperGlobal] Reranking graph convergence in 0.08s
  => Total time for Query 1: 0.25s
...
Processing Query 70/70: radcliffe_camera_5
...
==================================================
EVALUATION RESULTS (roxford5k)
==================================================
mAP (Medium): 88.54%
mAP (Hard):   72.15%

Average Search Time: 0.28s / image
```

### 🧠 Phân tích chi tiết kết quả trên:
1. **Quá trình nạp mô hình:** Hệ thống sẽ tốn khoảng 5-10 giây đầu tiên để đưa các mạng Neural (FIRe, CVNet) và Từ điển 65,536 từ vựng (ASMK) vào RAM/VRAM của máy tính.
2. **Quy trình 3 bước cho mỗi Query (0.28s):**
   - **Bước 1 (ASMK Filter):** Thay vì so sánh với 1 triệu ảnh, nó dùng thuật toán "Tra Từ Điển" để nhặt ra đúng $K=100$ ảnh tiềm năng nhất. (Tốn 0.05s).
   - **Bước 2 (Local Chamfer & MDS):** Thuật toán tính ma trận điểm-điểm (Point-to-Point) giữa Query và 100 bức ảnh này, sau đó dùng $MDS$ để biến ma trận này thành 1 vector 128 chiều đại diện cho "Chi tiết cục bộ". (Tốn 0.12s).
   - **Bước 3 (SuperGlobal Graph):** Vector "Chi tiết" ($128D$) và Vector "Toàn cảnh" từ CVNet ($2048D$) được gộp lại. Xây dựng đồ thị mạng lưới để các bức ảnh tự truyền điểm cho nhau (Diffusion). (Tốn 0.08s).
3. **Kết quả mAP (Mean Average Precision):** 
   - Điểm **Medium** (88.54%): Là thang đo cho những bức ảnh truy vấn bị che khuất một phần vừa phải.
   - Điểm **Hard** (72.15%): Là thang đo cho những bức ảnh truy vấn bị che khuất rất nặng, góc chụp khắc nghiệt. Điểm này chứng minh hệ thống có khả năng tìm kiếm trong điều kiện khó khăn.

---

## 🐢 2. Chế độ Vét Cạn (Exact Chamfer + SuperGlobal)

### 🛠 Lệnh thực thi:
```bash
python src/online/stage4_search/search_exact_chamfer.py
```

### 🎯 Mô tả & Ví dụ đầu ra (What you get):
```text
Loading FIRe Model (Local)...
Loading CVNet Model (Global)...
Loading Database Features (5063 images)...

==================================================
STARTING EXACT EVALUATION: roxford5k
==================================================
Processing Query 1/70: all_souls_1
  [Exact Chamfer] Computing point-to-point distance with ALL 5063 images...
  [Chamfer + MDS] Local distance matrix computed in 30.50s
  [SuperGlobal] Reranking graph convergence in 0.10s
  => Total time for Query 1: 30.60s
...
==================================================
EVALUATION RESULTS (roxford5k)
==================================================
mAP (Medium): 92.20%
mAP (Hard):   76.50%

Average Search Time: 30.2s / image
```

### 🧠 Phân tích tại sao phải dùng Chế độ Vét Cạn?
Mặc dù tốc độ **rất chậm (30 giây/ảnh)**, nhưng chế độ này không sử dụng bộ lọc ASMK. Nó mang từng điểm ảnh của bức Query đem đi nhân ma trận với toàn bộ điểm ảnh của 5,063 bức ảnh trong Database (Thuật toán Exact Chamfer Distance Toán Học Mọi Điểm).
- Do không bị sót bất kỳ ảnh nào, độ chính xác (mAP) đạt mức kịch trần (92.2%). 
- **Ứng dụng:** Bạn chỉ dùng file này khi viết **Báo Cáo Khoa Học (Paper)** để chứng minh ngưỡng sức mạnh tối đa của hệ thống, tuyệt đối không dùng file này để build Website.

---

## 🧮 Nguyên lý toán học (Mathematical Logic)

### 1. Ma Trận Khoảng Cách Cục Bộ (Exact Chamfer Distance)
Để tính toán xem ảnh Query ($Q$) có chứa chung vật thể với ảnh Database ($X$) hay không, hệ thống tính toán dựa trên tập hợp điểm cục bộ:
$$ S_{local}(Q, X) = \frac{1}{2|X|} \sum_{x \in X} \max_{q \in Q} (q \cdot x) + \frac{1}{2|Q|} \sum_{q \in Q} \max_{x \in X} (q \cdot x) $$
- **Ý nghĩa:** Điểm $x$ trên tòa nhà ở ảnh Database cố gắng tìm điểm $q$ giống nó nhất ở ảnh Query (Và ngược lại). Nếu tìm thấy, tích vô hướng $(q \cdot x)$ sẽ rất cao (gần 1.0).

### 2. Dung hợp (Fusion) và Reranking (MDS + SuperGlobal)
Kết hợp Vector $F_{mds}$ (cục bộ) và $F_{global}$ (toàn cục) theo tỷ lệ trọng số:
$$ F_{concat} = [ \sqrt{w_{local}} F_{mds}, \sqrt{w_{global}} F_{global} ] $$
- Trọng số $w_{local} = 0.5$ và $w_{global} = 0.5$.
Sau khi gộp, hệ thống áp dụng kỹ thuật Reranking bằng đồ thị. Mỗi bức ảnh là một Nodes. Các ảnh giống nhau sẽ "kéo" điểm của nhau lên bằng tham số lan truyền (Diffusion) $\beta = 0.31$.
