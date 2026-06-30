from src.stage2.query_candidate import query_candidate
import numpy as np
import json
from src.stage2.smacof import smacof
import torch
import numpy as np
import time

def generate_distance_matrix(query_image, database, dataset_name='roxford5k', k=700):
    # Get candidate images for the input image
    
    candidates = query_candidate(query_image, f"{dataset_name}_query_results", num_candidates=1600)

    matrix = np.ones((k+1, k+1), dtype=np.float32)
    matrix[0, 0] = 0.0  # Distance from the query
    for i in range(1, k+1):
        matrix[0, i] = candidates[i-1]["score"]  # Distance from the query to candidate i

    for i in range(1, k+1):
        if database[candidates[i-1]["db_name"]].get(query_image) is not None:
            matrix[i, 0] = database[candidates[i-1]["db_name"]][query_image]  # Distance from candidate i to query
        else:
            matrix[i, 0] = 1.0  # If no distance is found, set to infinity
        for j in range(1, k+1):
            if i == j: matrix[i, j] = 0.0  # Distance from candidate i to itself
            if database[candidates[i-1]["db_name"]].get(candidates[j-1]["db_name"]) is not None:
                matrix[i, j] = database[candidates[i-1]["db_name"]][candidates[j-1]["db_name"]]["score"]  # Distance from candidate i to candidate j
            else:
                matrix[i, j] = 1.0  # If no distance is found, set to infinity
    return matrix, candidates

def embedding_query_candidate(matrix, query_image, candidates, feats, name2idx, device='cuda'):
    print("Distance matrix shape:", matrix.shape)
    
    mds_embeddings = smacof(
        D_tensor=matrix,
        epsilon=0.1,  
        max_iter=15,  
        n_components=128,
        device=device
    )
    print("MDS Embeddings shape:", mds_embeddings.shape)
    
    # Lấy k từ kích thước ma trận sinh ra MDS
    k = matrix.shape[0] - 1 
    
    all_names = [query_image] + [c["db_name"] for c in candidates]
    
    sg_list = []
    for name in all_names:
        lookup_name = name if name in name2idx else f"{name}.jpg"
        
        if lookup_name in name2idx:
            idx = name2idx[lookup_name]
            sg_list.append(feats[idx])
        else:
            print(f"Cảnh báo: Không tìm thấy {name} trong đặc trưng SG. Dùng vector 0.")
            sg_list.append(torch.zeros(2048, dtype=torch.float32))
            
    sg_tensor = torch.stack(sg_list).to(device)
    print("All SuperGlobal Tensor shape:", sg_tensor.shape)
    
    w_local = 0.19 
    w_global = 0.81

    # Create full MDS embedding tensor
    mds_full = torch.zeros((sg_tensor.shape[0], mds_embeddings.shape[1]), dtype=sg_tensor.dtype, device=device)
    mds_full[:k+1] = mds_embeddings

    # Combine using horizontal concatenation (dim=1) like in stage 4
    final_embeddings = torch.cat([
        float(np.sqrt(w_local)) * mds_full,
        float(np.sqrt(w_global)) * sg_tensor
    ], dim=1)
    print("Final Embeddings shape:", final_embeddings.shape)
    
    return final_embeddings

if __name__ == "__main__":
    dataset_name = 'roxford5k'
    with open(f"src/stage2/{dataset_name}_self_pairs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    database = {
        q["query_name"]: q["results"]
        for q in data["queries"]
    }
    super_global_features = torch.load("src/stage2/feats_roxford5k_RN50.pth")

    names = super_global_features["names"]      # list[str]
    feats = super_global_features["feats"]      # Tensor (N, D)
    name2idx = {name: i for i, name in enumerate(names)}


    start = time.time()
    matrix, candidates = generate_distance_matrix("all_souls_000021", database, dataset_name, k=700)
    embedding_query_candidate(matrix, "all_souls_000021", candidates, feats, name2idx)

    matrix, candidates = generate_distance_matrix("all_souls_000026", database, dataset_name, k=700)
    embedding_query_candidate(matrix, "all_souls_000026", candidates, feats, name2idx)

    print(time.time() - start)