# L2G Pipeline - Source Code
Đây là thư mục lõi chứa toàn bộ mã nguồn của hệ thống Tìm kiếm Hình ảnh Local-to-Global (L2G). Mã nguồn đã được cấu trúc lại theo chuẩn công nghiệp, tách biệt rõ ràng giữa môi trường Offline và Online.

## Cấu trúc thư mục (Directory Structure)
- `offline/`: Chứa các Script chạy **1 lần duy nhất** để trích xuất đặc trưng và lập chỉ mục (Indexing) cho kho dữ liệu ảnh (Database).
- `online/`: Chứa các Script chạy **Thời gian thực (Real-time)** để phục vụ User khi họ upload ảnh lên tìm kiếm (Search).
- `core/`: Chứa mã nguồn của các thư viện/thuật toán gốc (ví dụ: SuperGlobal).
- `utils/`: Chứa các công cụ hỗ trợ như đánh giá mAP (Evaluation).

**Lưu ý:** Luôn chạy các lệnh Python từ thư mục gốc của dự án (`main/`), ví dụ: `python src/offline/stage1_local_extract/extract_local.py`.
