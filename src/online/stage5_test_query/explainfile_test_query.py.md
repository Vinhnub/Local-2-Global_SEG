# Giải thích chi tiết: `test_query.py`

**Đường dẫn:** `C:/Users/ezycloudx-admin/Desktop/seg/main/src/online/stage5_test_query/test_query.py`


## 📌 Nhiệm vụ (What it is)
File API dùng thử thực tế. Bạn cung cấp 1 đường dẫn ảnh, nó sẽ tìm và xuất ra 20 ảnh tương đồng nhất.

## ⚙️ Cách hoạt động (How it works)
1. Tự động nạp đồng thời FIRe, CVNet và ASMK lên RAM.
2. Rút trích Feature Cục bộ và Toàn cục NGAY LẬP TỨC từ bức ảnh bạn đưa vào.
3. Query qua ASMK.
4. Rerank bằng Graph.
5. Sao chép và đánh số 20 bức ảnh đạt top vào thư mục `output_test/`.

## 📤 Đầu ra (Output)
- Ảnh xếp hạng từ 1 đến 20 trực quan hóa.
