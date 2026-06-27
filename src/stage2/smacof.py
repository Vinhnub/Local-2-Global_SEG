import torch

def smacof(D_tensor, epsilon=0.01, max_iter=20, n_components=2048, device='cuda'):
    """
    Triển khai thuật toán SMACOF chạy trực tiếp trên GPU bằng PyTorch.
    
    Args:
        D_tensor: Ma trận bất đồng điệu kích thước (N, N), kiểu torch.Tensor, đã nằm trên GPU
        epsilon: Ngưỡng dừng hội tụ
        max_iter: Số vòng lặp tối đa
        n_components: Số chiều của vector nhúng (VD: 2048 để khớp với SuperGlobal)
        device: 'cuda' hoặc 'cpu'
    """
    # Nếu D_tensor vẫn đang là numpy array, chuyển nó thành Tensor và đẩy lên GPU
    if not isinstance(D_tensor, torch.Tensor):
        D_tensor = torch.tensor(D_tensor, dtype=torch.float32, device=device)
    else:
        # Nếu đã là Tensor nhưng nằm ở CPU, đẩy nó lên thiết bị tương ứng (GPU)
        D_tensor = D_tensor.to(dtype=torch.float32, device=device)
        
    n = D_tensor.shape[0]
    
    # 1. Khởi tạo tọa độ ngẫu nhiên trên GPU
    X = torch.rand((n, n_components), device=device, dtype=torch.float32)
    
    # 2. Khởi tạo ma trận V và tính nghịch đảo giả (Pseudoinverse)
    V = torch.full((n, n), -1.0, device=device, dtype=torch.float32)
    V.fill_diagonal_(n - 1)
    
    # torch.linalg.pinv chạy cực nhanh trên GPU
    V_pinv = torch.linalg.pinv(V) 
    
    prev_stress = float('inf')
    
    for iteration in range(max_iter):
        # Tính khoảng cách Euclidean: ||X_i - X_j||
        # Sử dụng cdist của PyTorch để tối ưu hóa bộ nhớ và tốc độ
        D_current = torch.cdist(X, X, p=2)
        
        # Tránh lỗi chia cho 0
        D_current = torch.where(D_current == 0, torch.tensor(1e-9, device=device), D_current)
        
        # 3. Tính ma trận B(X)
        B = - (D_tensor / D_current)
        B.fill_diagonal_(0)
        
        # SỬA LỖI: Lấy tổng từng hàng và copy vào đường chéo
        row_sums = -torch.sum(B, dim=1)
        B.diagonal().copy_(row_sums)
        
        # 4. Cập nhật tọa độ X mới (Phép nhân ma trận song song)
        X = torch.matmul(V_pinv, torch.matmul(B, X))
        
        # 5. Đánh giá hội tụ
        stress = torch.sum((D_tensor - D_current)**2) / 2
        #print(f"Iteration {iteration + 1}/{max_iter}, Stress: {stress.item():.4f}")
        if prev_stress - stress < epsilon:
            # Uncomment dòng dưới nếu muốn in log debug
            print(f"Hội tụ tại vòng lặp {iteration} với Stress = {stress.item():.4f}")
            break
            
        prev_stress = stress
        
    return X

# if __name__ == "__main__":
#     # Kiểm tra xem máy có GPU (CUDA) không
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Đang chạy trên thiết bị: {device}")
    
#     k = 700 
#     n_points = k + 1
    
#     # Khởi tạo ma trận khoảng cách giả lập trực tiếp trên GPU
#     D = torch.ones((n_points, n_points), device=device, dtype=torch.float32)
#     D.fill_diagonal_(0)
    
#     # Chạy thuật toán lấy đặc trưng 2048 chiều
#     global_embeddings_gpu = smacof(D, n_components=2048, device=device)
    
#     print("Kích thước tensor đầu ra:", global_embeddings_gpu.shape)