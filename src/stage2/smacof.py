import torch

def smacof(D_tensor, epsilon=0.1, max_iter=20, n_components=2048, device='cuda'):
    """
    Triển khai thuật toán SMACOF chạy trực tiếp trên GPU bằng PyTorch.
    
    Args:
        D_tensor: Ma trận bất đồng điệu kích thước (N, N), kiểu torch.Tensor, đã nằm trên GPU
        epsilon: Ngưỡng dừng hội tụ
        max_iter: Số vòng lặp tối đa
        n_components: Số chiều của vector nhúng (VD: 2048 để khớp với SuperGlobal)
        device: 'cuda' hoặc 'cpu'
    """
    if not isinstance(D_tensor, torch.Tensor):
        D_tensor = torch.tensor(D_tensor, dtype=torch.float32, device=device)
    else:
        D_tensor = D_tensor.to(dtype=torch.float32, device=device)
        
    n = D_tensor.shape[0]
    
    X = torch.rand((n, n_components), device=device, dtype=torch.float32)
    
    V = torch.full((n, n), -1.0, device=device, dtype=torch.float32)
    V.fill_diagonal_(n - 1)
    
    V_pinv = torch.linalg.pinv(V) 
    
    prev_stress = float('inf')
    
    for iteration in range(max_iter):
        D_current = torch.cdist(X, X, p=2)
        
        D_current = torch.where(D_current == 0, torch.tensor(1e-9, device=device), D_current)
        
        B = - (D_tensor / D_current)
        B.fill_diagonal_(0)
        
        row_sums = -torch.sum(B, dim=1)
        B.diagonal().copy_(row_sums)
        
        X = torch.matmul(V_pinv, torch.matmul(B, X))
        
        stress = torch.sum((D_tensor - D_current)**2) / 2
        if prev_stress - stress < epsilon:
            print(f"Hội tụ tại vòng lặp {iteration} với Stress = {stress.item():.4f}")
            break
            
        prev_stress = stress
        
    return X