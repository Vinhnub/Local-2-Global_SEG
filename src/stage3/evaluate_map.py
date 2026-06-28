"""
Giai đoạn 3 (Stage 3): SuperGlobal Re-ranking & Đánh giá toàn bộ Pipeline

Script này đóng vai trò chạy toàn bộ quy trình Local-to-Global và đánh giá điểm mAP cuối cùng.
Quy trình hoạt động:
1. Đọc dữ liệu khoảng cách ứng viên từ thuật toán CANN (Stage 1).
2. Xây dựng ma trận khoảng cách và thực hiện ghép nối đặc trưng MDS + SuperGlobal (Stage 2).
3. Đưa qua đồ thị khuếch tán Graph Diffusion để xếp hạng lại (Stage 3).
4. Tính toán và in ra điểm số mAP chuẩn cho các bộ giao thức Easy, Medium, Hard của Oxford/Paris.
"""
import os
import sys
import pickle
import json
import time
import torch
import numpy as np

# Đảm bảo import được module từ thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.stage2.embedding_query_candidate import generate_distance_matrix, embedding_query_candidate
from src.stage3.graph_diffusion import superglobal_reranking_gpu

# =======================================================
# Các hàm đánh giá mAP chuẩn RevisitOp (từ Oxford/Paris)
# =======================================================
def compute_ap(ranks, nres):
    """Computes average precision for given ranked indexes."""
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1. / nres
    for j in np.arange(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.
    return ap

def compute_map(ranks, gnd, kappas=[1, 5, 10]):
    """Computes the mAP for a given set of returned results."""
    map = 0.
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except KeyError:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # 1-based index
        for j in np.arange(len(kappas)):
            if len(pos) > 0:
                kq = min(max(pos), kappas[j])
                prs[i, j] = (pos <= kq).sum() / kq
            else:
                prs[i, j] = 0.0
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    return map, aps, pr, prs

def evaluate_map_metrics(dataset_name, ranks, gnd, kappas=[1, 5, 10]):
    """Evaluates Easy, Medium, and Hard protocols for Revisited datasets."""
    # Easy Protocol
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

    # Medium Protocol
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

    # Hard Protocol
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

    print(f"\n==============================================")
    print(f"[{dataset_name.upper()}] Evaluation Results (mAP)")
    print(f"==============================================")
    print(f" >> Easy   : {mapE * 100:.2f}")
    print(f" >> Medium : {mapM * 100:.2f}")
    print(f" >> Hard   : {mapH * 100:.2f}")
    print(f"==============================================\n")
    return mapE, mapM, mapH

# =======================================================
# HÀM ĐÁNH GIÁ CHÍNH (PIPELINE TỪ STAGE 1 ĐẾN 3)
# =======================================================
def evaluate_full_pipeline(dataset_name='roxford5k'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang đánh giá mAP trên thiết bị: {device} ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(os.path.dirname(project_root), "data_stage2")
    
    # 1. Load Ground Truth
    gnd_path = os.path.join(os.path.dirname(project_root), "datasets", dataset_name, f"gnd_{dataset_name}.pkl")
    print(f"Loading Ground Truth: {gnd_path}")
    with open(gnd_path, 'rb') as f:
        gnd_data = pickle.load(f)
    gnd = gnd_data['gnd']
    qimlist = gnd_data['qimlist']
    imlist = gnd_data['imlist']
    
    # Tạo từ điển map tên ảnh sang chỉ số index
    img_name_to_idx = {name: idx for idx, name in enumerate(imlist)}
    
    # 2. Load dữ liệu Local Search Stage 1
    json_path = os.path.join(data_dir, f"{dataset_name}_self_pairs.json")
    print(f"Loading JSON distances (Stage 1 output): {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    database = {q["query_name"]: q["results"] for q in data["queries"]}
    
    # 3. Load SuperGlobal Features (Stage 2/3)
    pth_path = os.path.join(data_dir, f"feats_{dataset_name}_RN50.pth")
    print(f"Loading SuperGlobal features: {pth_path}")
    super_global_features = torch.load(pth_path)
    names = super_global_features["names"]
    feats = super_global_features["feats"]
    name2idx = {name: i for i, name in enumerate(names)}
    
    # Tham số
    k_local = 700
    k_sg = 6
    beta_sg = 0.31
    
    ranks = np.zeros((len(imlist), len(qimlist)), dtype=np.int32)
    start_total = time.time()
    
    print("\n[START] Bắt đầu xử lý từng query...")
    # 4. Loop qua từng query
    for i, q_name in enumerate(qimlist):
        if (i+1) % 10 == 0 or i == 0:
            print(f"  -> Đang xử lý query {i+1}/{len(qimlist)}: {q_name}")
            
        # --- STAGE 2: MDS FUSION ---
        matrix, candidates = generate_distance_matrix(q_name, database, dataset_name, k=k_local)
        fused_embeddings = embedding_query_candidate(matrix, q_name, candidates, feats, name2idx, device=device)
        
        # --- STAGE 3: GRAPH DIFFUSION ---
        refined_embeddings = superglobal_reranking_gpu(fused_embeddings, k_sg=k_sg, beta=beta_sg)
        
        # Tính điểm Cosine Similarity (Dùng hàm F.cosine_similarity cho chuẩn học thuật)
        import torch.nn.functional as F
        query_vector = refined_embeddings[0].unsqueeze(0)  # Thêm chiều batch để so sánh
        candidate_vectors = refined_embeddings[1:]
        scores = F.cosine_similarity(candidate_vectors, query_vector, dim=1).cpu().numpy()
        
        # Lấy tên các candidates đã sắp xếp
        sorted_local_indices = scores.argsort()[::-1]
        sorted_cand_names = [candidates[idx]["db_name"] for idx in sorted_local_indices]
        
        # --- XẾP HẠNG CUỐI CÙNG (Cho mAP) ---
        # Chỉ những candidate có mặt trong top 1600 mới có điểm.
        # Những DB images còn lại (không nằm trong candidates) được xếp ngẫu nhiên hoặc ở cuối.
        
        # Lấy index thực sự trong bộ DB (imlist)
        ranked_indices_for_query = []
        added_names = set()
        
        for c_name in sorted_cand_names:
            if c_name in img_name_to_idx:
                ranked_indices_for_query.append(img_name_to_idx[c_name])
                added_names.add(c_name)
                
        # Thêm nốt các ảnh DB không xuất hiện trong top ứng viên
        for db_name in imlist:
            if db_name not in added_names:
                ranked_indices_for_query.append(img_name_to_idx[db_name])
                
        ranks[:, i] = np.array(ranked_indices_for_query)
        
    print(f"\n[DONE] Hoàn tất xử lý {len(qimlist)} queries trong {time.time() - start_total:.2f} giây.")
    
    # LƯU KẾT QUẢ ĐẦU RA (OUTPUT) CHO CÔNG TY
    # Lưu ma trận thứ hạng (ranks) ra file Numpy
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_stage3_final_ranks.npy")
    np.save(output_file, ranks)
    print(f"\n[EXPORT] Đã xuất thành công file kết quả của 70 queries (dùng để nộp cho công ty) tại:")
    print(f" -> {output_file}")
    
    # 5. Tính điểm mAP
    evaluate_map_metrics(dataset_name, ranks, gnd)

if __name__ == "__main__":
    # Đánh giá ROxford5k
    evaluate_full_pipeline("roxford5k")
