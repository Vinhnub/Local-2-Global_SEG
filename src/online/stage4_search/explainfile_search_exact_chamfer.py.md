# Giải thích chi tiết: `search_exact_chamfer.py`

**Đường dẫn:** `C:/Users/ezycloudx-admin/Desktop/seg/main/src/online/stage4_search/search_exact_chamfer.py`


## 📌 Nhiệm vụ (What it is)
Script tìm kiếm Vét Cạn (Exhaustive Search). Nó bỏ qua bộ lọc nhanh ASMK để đạt độ chính xác tối đa.

## ⚙️ Cách hoạt động (How it works)
- Lấy từng ảnh Query đi so khớp trực tiếp (Exact Chamfer) với TẤT CẢ hàng ngàn ảnh trong Database.
- Tốc độ cực chậm (30 giây/ảnh) nhưng không bỏ sót bất kỳ manh mối nào.
- Sau đó vẫn áp dụng SuperGlobal Graph.

## 📤 Đầu ra (Output)
- Điểm mAP cao nhất tuyệt đối (Dùng để viết báo cáo khoa học).
