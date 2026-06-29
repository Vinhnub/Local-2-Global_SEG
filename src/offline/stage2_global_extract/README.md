# 🌍 Stage 2: CVNet Global Feature Extraction (Offline)

## 📌 Chức năng (What it does)
Dùng mạng Neural Network sâu (CVNet_R101 hoặc SuperGlobal) để tóm gọn toàn bộ ý nghĩa (Bối cảnh, màu sắc, khung cảnh) của bức ảnh vào **1 Vector duy nhất (Global Feature)**.
Vector này sẽ làm nhiệm vụ Reranking (Sắp xếp lại) ở Stage 4 để gỡ điểm cho những bức ảnh bị ASMK xếp hạng thấp.

## 🛠 Lệnh thực thi:
```bash
python src/offline/stage2_global_extract/extract_global.py
```

## 🎯 Mô tả & Ví dụ đầu ra (What you get):
Màn hình Terminal sẽ xuất hiện:
```text
Loading CVNet (SuperGlobal R101) model...
Model loaded successfully!

Processing 70 QUERY images...
Query   1/70: all_souls_1 -> (1, 2048)
Query   2/70: all_souls_2 -> (1, 2048)
...
Query processing complete: 70 success

Processing 5063 DATABASE images...
DB      1/5063: all_souls_000000 -> (1, 2048)
DB      2/5063: all_souls_000001 -> (1, 2048)
...
Database processing complete: 5063 success
```

### 🧠 Phân tích kết quả:
- **`-> (1, 2048)`**: Toàn bộ bức ảnh được AI "nén" lại thành một chuỗi duy nhất gồm đúng 2048 con số. Chuỗi số này chứa tất cả "Tâm hồn" của bức ảnh đó.
- Cực kỳ tiết kiệm dung lượng và giúp thuật toán tính khoảng cách (Cosine Similarity) chạy trong 0.0001 giây.
- **Vị trí lưu:** `output/stage2/features/roxford5k/database/`

---

## 🧮 Nguyên lý toán học (Mathematical Logic)
Với 1 ảnh $I$, mạng CVNet trích xuất ra Feature Map $X \in \mathbb{R}^{2048 \times H \times W}$.
Thay vì giữ nguyên ma trận to, ta dùng hàm **GeM Pooling (Generalized Mean Pooling)** để ép nó thành 1 vector duy nhất $g \in \mathbb{R}^{2048}$:
$$ g_k = \left( \frac{1}{|X_k|} \sum_{x \in X_k} (ReLU(x))^p \right)^{\frac{1}{p}} $$
Sau đó chuẩn hóa độ dài về 1 (L2 Normalize) để đưa vector lên bề mặt hình cầu đa chiều, giúp đo lường dễ dàng hơn:
$$ \hat{g} = \frac{g}{\|g\|_2} $$
