# 📚 Stage 3: ASMK Index Building (Offline)

## 📌 Chức năng (What it does)
Xây dựng "Từ điển Hình ảnh" bằng thuật toán ASMK (Aggregated Selective Match Kernel) từ hàng triệu vector cục bộ thu được ở Stage 1.
Tại sao phải làm bước này? Vì nếu Query so sánh trực tiếp với 1 triệu ảnh, nó phải tính $1,000,000 \times 600 \times 600$ phép nhân, mất hàng tháng trời. ASMK sẽ biến việc tìm kiếm giống như bạn "tra từ điển tiếng anh", tìm 1 phát ra ngay đáp án trong 0.05 giây!

## 🛠 Lệnh thực thi:
```bash
python src/offline/stage3_build_index/build_index.py
```

## 🎯 Mô tả & Ví dụ đầu ra (What you get):
Màn hình Terminal sẽ xuất hiện:
```text
Loading 5063 database local features...
Total vectors collected: 3,037,800 vectors of size 128.
--------------------------------------------------
[ASMK] Training Codebook with Faiss (GPU)...
  => Running KMeans to find 65,536 centroids from 3M vectors.
  => Training completed in 45.2s. Codebook saved!
--------------------------------------------------
[ASMK] Building IVF Index (Quantization)...
  => Pushing 5063 images into IVF buckets.
  => Binarizing residuals...
  => IVF Build completed in 12.5s!
--------------------------------------------------
DONE. Index saved to: output/stage3/asmk/roxford5k_ivf.pkl
```

### 🧠 Phân tích kết quả:
- **`3,037,800 vectors`**: Hệ thống đã lấy 5063 ảnh, mỗi ảnh có 600 điểm nổi bật $\Rightarrow 5063 \times 600 = 3,037,800$ điểm chấm trên không gian 128 chiều.
- **`65,536 centroids`**: Máy GPU dùng thuật toán K-Means gom 3 triệu điểm đó thành 65,536 cụm (Clusters). Mỗi cụm đóng vai trò là 1 "Từ Vựng" (Visual Word).
- **`IVF (Inverted File)`**: Lập bảng chỉ mục ngược. Giống như trang phụ lục cuối quyển sách: "Từ vựng số 4522 xuất hiện ở các bức ảnh số [12, 456, 1024]".
- **Vị trí lưu:** `output/stage3/asmk/` (2 file `.pkl` là linh hồn của tốc độ tìm kiếm).

---

## 🧮 Nguyên lý toán học (Mathematical Logic)
Khi một vector điểm ảnh $v$ được phân vào cụm (Từ vựng) số $c$, ASMK không ném bỏ $v$, mà giữ lại **Phần Dư (Residual)** để chống sai số:
$$ r(v) = v - c $$
Sau đó nhị phân hóa (Binarization) $r(v)$ thành chuỗi bit $010101$ để tối ưu dung lượng RAM và tính toán khoảng cách bằng phép toán `XOR/Hamming` ở tốc độ phần cứng máy tính (cực kỳ nhanh).
