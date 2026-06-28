import subprocess
import sys
import os
import re

def run_stage1(dataset):
    """
    Stage 1: Trích xuất đặc trưng cục bộ (Local Features) bằng mô hình FIRe.
    Mỗi ảnh sẽ được chuyển thành 600 vector x 128 chiều.
    """
    script_path = os.path.join('src', 'stage1_extract_local.py')
    print(f"Running Stage 1 (Local Features) for {dataset}...")
    subprocess.run([sys.executable, script_path, '--dataset', dataset], check=True)

def run_stage2(dataset):
    """
    Stage 2: Trích xuất đặc trưng toàn cục (Global Features) bằng mô hình CVNet-R101.
    Mỗi ảnh sẽ được nén thành 1 vector duy nhất x 2048 chiều.
    """
    script_path = os.path.join('src', 'stage2_extract_global.py')
    print(f"Running Stage 2 (Global Features) for {dataset}...")
    subprocess.run([sys.executable, script_path, '--dataset', dataset], check=True)

def run_stage3():
    """
    Stage 3: Kết hợp L2G (Local-to-Global) và Graph Diffusion.
    - So sánh Local features bằng Exact Chamfer Distance.
    - Ép kiểu MDS (từ M x M thành 128-d).
    - Hợp nhất với Global feature thành 2176-d (F_concat).
    - Dùng thuật toán Graph Diffusion trên đồ thị KNN để Re-ranking.
    """
    script_path = os.path.join('src', 'stage3_rerank.py')
    print("Running Stage 3 (Re-ranking and Evaluation)...")
    subprocess.run([sys.executable, script_path], check=True)

def parse_results(dataset):
    """ Hàm đọc kết quả mAP từ file txt xuất ra sau khi chạy Stage 3 """
    result_file = os.path.join('output', 'stage3', f"{dataset}_final_results.txt")
    if not os.path.exists(result_file):
        return None
    with open(result_file, 'r') as f:
        text = f.read()
    
    # Tìm kiếm các dòng chứa kết quả (Ví dụ: "Medium mAP: 80.31%")
    m = re.search(r"Easy mAP:\s*([0-9.]+)%", text)
    easy = float(m.group(1)) if m else None
    m = re.search(r"Medium mAP:\s*([0-9.]+)%", text)
    medium = float(m.group(1)) if m else None
    m = re.search(r"Hard mAP:\s*([0-9.]+)%", text)
    hard = float(m.group(1)) if m else None
    
    return {'easy': easy, 'medium': medium, 'hard': hard}

def main():
    datasets = ['roxford5k', 'rparis6k']
    
    # Chạy lần lượt các bước cho cả 2 bộ dữ liệu (Oxford và Paris)
    print("=== STARTING L2G PIPELINE ===")
    
    # Bước 1: Trích xuất Local
    for ds in datasets:
        run_stage1(ds)
        
    # Bước 2: Trích xuất Global
    for ds in datasets:
        run_stage2(ds)
        
    # Bước 3: Re-ranking
    run_stage3()
    
    # In báo cáo kết quả cuối cùng ra màn hình
    print("\n=== FINAL mAP RESULTS ===")
    for ds in datasets:
        scores = parse_results(ds)
        if scores:
            print(f"{ds.upper():<12} | Easy: {scores['easy']:.2f}% | Medium: {scores['medium']:.2f}% | Hard: {scores['hard']:.2f}%")
        else:
            print(f"{ds.upper():<12} | No results found")

if __name__ == "__main__":
    main()
