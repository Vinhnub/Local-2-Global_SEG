"""
Giai đoạn 3 (Stage 3): Graph Diffusion cho một ảnh Query duy nhất

Script này chạy thử nghiệm độc lập Stage 2 (Fusion) và Stage 3 (Graph Diffusion) 
lên một bức ảnh truy vấn (query) để in ra danh sách Top 10 ứng viên tốt nhất sau khi Re-ranking.
Phù hợp để test nhanh hoặc debug kết quả của một ảnh cụ thể thay vì chạy toàn bộ mAP.
"""
import torch
import json
import time
import sys
import os
import numpy as np

# Đảm bảo import được module từ thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.stage2.embedding_query_candidate import generate_distance_matrix, embedding_query_candidate
from src.stage3.graph_diffusion import superglobal_reranking_gpu


def run_pipeline_for_query(query_image, dataset_name='roxford5k'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")
    
    # Hyperparameters chuẩn bài báo
    k_local = 700
    k_sg = 6
    beta_sg = 0.31
    M_sg = 1600 # 1600 candidates

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(os.path.dirname(project_root), "data_stage2")
    
    # Load dữ liệu Stage 2 (từ thư mục data_stage2)
    json_path = os.path.join(data_dir, f"{dataset_name}_self_pairs.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(os.path.dirname(script_dir), "stage2", f"{dataset_name}_self_pairs.json")
            
    print(f"Loading JSON distances: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    database = {
        q["query_name"]: q["results"]
        for q in data["queries"]
    }
    
    pth_path = os.path.join(data_dir, f"feats_{dataset_name}_RN50.pth")
    if not os.path.exists(pth_path):
        pth_path = os.path.join(os.path.dirname(script_dir), "stage2", f"feats_{dataset_name}_RN50.pth")
            
    print(f"Loading SuperGlobal features: {pth_path}")
    super_global_features = torch.load(pth_path)
    names = super_global_features["names"]
    feats = super_global_features["feats"]
    name2idx = {name: i for i, name in enumerate(names)}
    
    start_time = time.time()
    
    # -------------------------------------------------------------
    # STAGE 2: LOCAL TO GLOBAL FUSION (MDS + SuperGlobal)
    # -------------------------------------------------------------
    print(f"\n[STAGE 2] Xử lý ảnh query: {query_image}")
    matrix, candidates = generate_distance_matrix(query_image, database, dataset_name, k=k_local)
    
    # fused_embeddings: (1601, 2048) chứa query ở index 0
    fused_embeddings = embedding_query_candidate(matrix, query_image, candidates, feats, name2idx, device=device)
    
    # -------------------------------------------------------------
    # STAGE 3: SUPERGLOBAL RE-RANKING (Graph Diffusion)
    # -------------------------------------------------------------
    print(f"\n[STAGE 3] Bắt đầu SuperGlobal Graph Diffusion (M={M_sg}, k_sg={k_sg}, beta={beta_sg})...")
    
    # Đưa tensor vào hàm diffusion
    refined_embeddings = superglobal_reranking_gpu(fused_embeddings, k_sg=k_sg, beta=beta_sg)
    print(f"Refined Embeddings shape: {refined_embeddings.shape}")
    
    # Tính điểm số (Dùng hàm F.cosine_similarity cho chuẩn học thuật)
    import torch.nn.functional as F
    # Query ở vị trí 0, ứng viên từ 1 -> 1600
    query_vector = refined_embeddings[0].unsqueeze(0)
    candidate_vectors = refined_embeddings[1:]
    
    # Tính Cosine Similarity
    scores = F.cosine_similarity(candidate_vectors, query_vector, dim=1).cpu().numpy()
    
    # Sắp xếp lại danh sách candidates dựa trên điểm số mới giảm dần
    sorted_indices = scores.argsort()[::-1]
    
    top10_names = []
    top10_scores = []
    print(f"\n[KẾT QUẢ] Top 10 ứng viên sau khi Re-ranking cho ảnh '{query_image}':")
    for i in range(10):
        orig_idx = sorted_indices[i]
        cand_name = candidates[orig_idx]["db_name"]
        score = scores[orig_idx]
        top10_names.append(cand_name)
        top10_scores.append(score)
        print(f"  {i+1:2d}. {cand_name} (Score: {score:.4f})")
        
    print(f"\nTổng thời gian (Stage 2 + 3): {time.time() - start_time:.4f} giây")
    


if __name__ == "__main__":
    # Test thử với 1 ảnh trong ROxford5k
    run_pipeline_for_query("all_souls_000021")
