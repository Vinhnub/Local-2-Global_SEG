import torch

def superglobal_reranking_gpu(features_tensor, k_sg=6, beta=0.31):
    """
    Thực hiện thuật toán SuperGlobal Re-ranking (Khuếch tán đồ thị - Graph Diffusion) trên GPU bằng PyTorch.
    Thuật toán này giúp tinh chỉnh (refine) lại các đặc trưng dựa trên cấu trúc đồ thị của các láng giềng gần nhất,
    từ đó làm nổi bật các bức ảnh có ngữ nghĩa giống nhau.
    
    Tham số (Args):
        features_tensor: Tensor chứa các đặc trưng đã ghép nối (ví dụ M ảnh, mỗi ảnh 2176 chiều). Kích thước (M, D).
        k_sg: Số lượng láng giềng gần nhất để xây dựng đồ thị k-NN (mặc định k_sg = 6 theo paper).
        beta: Tham số kiểm soát mức độ khuếch tán đồ thị (mặc định beta = 0.31 theo paper).
        
    Kết quả trả về (Returns):
        F_new_norm: Tensor chứa các đặc trưng MỚI sau khi đã được khuếch tán và chuẩn hóa chiều dài (L2-Norm).
    """
    device = features_tensor.device
    M = features_tensor.shape[0]
    
    # BƯỚC 0: Đảm bảo dữ liệu đầu vào đã được chuẩn hóa L2
    # Việc chuẩn hóa L2 (độ dài vector = 1) là cực kỳ quan trọng để phép nhân ma trận X * X^T
    # tương đương chính xác với phép tính Cosine Similarity (Khoảng cách góc).
    features_tensor = features_tensor / torch.norm(features_tensor, dim=1, keepdim=True).clamp(min=1e-6)
    
    # BƯỚC 1: Tính ma trận tương đồng (Cosine Similarity Matrix)
    # Nhân ma trận với ma trận chuyển vị của chính nó để tính độ tương đồng giữa mọi cặp ảnh.
    # Ma trận A sẽ có kích thước (M, M), trong đó A[i, j] là độ giống nhau giữa ảnh i và ảnh j.
    A = torch.matmul(features_tensor, features_tensor.T)
    
    # BƯỚC 2: Xây dựng đồ thị k-NN (K-Nearest Neighbors)
    # Tạo một bản sao của A và loại bỏ đường chéo (chính là độ tương đồng của một ảnh với chính nó)
    # Bằng cách gán đường chéo bằng âm vô cực (-inf), ta đảm bảo thuật toán không tự chọn lại chính nó làm láng giềng.
    A_no_self = A.clone()
    A_no_self.fill_diagonal_(-float('inf'))
    
    # Trích xuất k_sg láng giềng có độ tương đồng lớn nhất cho mỗi bức ảnh (theo từng hàng)
    topk_vals, topk_indices = torch.topk(A_no_self, k_sg, dim=1)
    
    # Khởi tạo ma trận trọng số W (chứa các cạnh của đồ thị) ban đầu rỗng (toàn số 0)
    W = torch.zeros((M, M), device=device, dtype=features_tensor.dtype)
    # Đổ các giá trị tương đồng (topk_vals) vào đúng vị trí tọa độ (topk_indices) trên ma trận W
    W.scatter_(1, topk_indices, topk_vals)
    
    # BƯỚC 3: Đối xứng hóa đồ thị
    # K-NN là đồ thị có hướng (A thích B không có nghĩa là B thích A).
    # Để thuật toán khuếch tán hoạt động ổn định, ta biến nó thành đồ thị vô hướng bằng cách lấy giá trị lớn nhất: W_sym = max(W, W^T)
    W_sym = torch.maximum(W, W.T)
    
    # BƯỚC 4: Chuẩn hóa đồ thị theo hàng (Row-normalization)
    # Tính tổng trọng số của từng hàng để chia đều (đảm bảo tổng mỗi hàng = 1)
    row_sums = W_sym.sum(dim=1, keepdim=True)
    # Kỹ thuật an toàn: Nếu có hàng nào tổng = 0 (node bị cô lập), ta set tạm thành 1.0 để tránh lỗi chia cho 0 (Divide by Zero)
    row_sums = torch.where(row_sums == 0, torch.tensor(1.0, device=device, dtype=features_tensor.dtype), row_sums)
    W_norm = W_sym / row_sums
    
    # BƯỚC 5: Thực hiện Khuếch tán đồ thị (Graph Diffusion)
    # Cập nhật đặc trưng mới = Đặc trưng cũ + (beta * Lan truyền thông tin từ các láng giềng)
    # Phép nhân W_norm * features_tensor chính là bước lấy trung bình đặc trưng của các láng giềng.
    F_new = features_tensor + beta * torch.matmul(W_norm, features_tensor)
    
    # BƯỚC 6: Chuẩn hóa L2-Norm bước cuối cùng
    # Đặc trưng sau khi được cộng gộp sẽ bị thay đổi độ dài, cần phải ép lại về mặt cầu đơn vị (L2-norm = 1)
    # để chuẩn bị cho bước xếp hạng (Re-ranking) bằng Cosine Similarity ở bên ngoài.
    F_new_norm = F_new / torch.norm(F_new, dim=1, keepdim=True).clamp(min=1e-6)
    
    return F_new_norm
