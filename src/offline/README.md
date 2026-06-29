# 🔄 Offline Modules Directory

## 📌 Chức năng (What it does)
Nơi chứa các Script dùng để Số hóa (Digitize) và Xây dựng chỉ mục (Indexing) cho Database.
Quá trình này cực kỳ mất thời gian, nhưng **CHỈ CẦN CHẠY 1 LẦN DUY NHẤT**.

## 🗂 3 Giai đoạn:
1. `stage1_local_extract`: Rút trích điểm đặc trưng nhỏ (FIRe).
2. `stage2_global_extract`: Rút trích điểm đặc trưng lớn (CVNet).
3. `stage3_build_index`: Nhào nặn dữ liệu vào Từ điển (ASMK).
