import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import MDS

# Configure print encoding to handle non-ASCII console outputs safely on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Helper evaluation functions from revisitop
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

    print(f"\n==================== EVALUATION: {dataset_name} ====================")
    print(f"mAP Easy:   {mapE * 100:.2f}%")
    print(f"mAP Medium: {mapM * 100:.2f}%")
    print(f"mAP Hard:   {mapH * 100:.2f}%")
    print(f"mP@k {kappas} Easy:   {mprE * 100}%")
    print(f"mP@k {kappas} Medium: {mprM * 100}%")
    print(f"mP@k {kappas} Hard:   {mprH * 100}%")
    print("====================================================================\n")
    return {
        'easy': mapE,
        'medium': mapM,
        'hard': mapH
    }

def parse_cann_results(filepath):
    """Parses C++ CANN search results text file."""
    results = {}
    current_query = None
    in_results = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Query:"):
                current_query = line.split("Query:")[1].strip()
                results[current_query] = []
                in_results = False
            elif line.startswith("Rank ") or line.startswith("---"):
                in_results = True
                continue
            elif in_results and current_query:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        db_name = parts[1]
                        results[current_query].append(db_name)
                    except ValueError:
                        pass
    return results

def compute_chamfer_matrix_pytorch(features, device):
    """Computes symmetric Chamfer distance matrix between local features using PyTorch."""
    n = len(features)
    # Move all features to device as a float32 tensor and stack
    feat_tensor = torch.stack(features).to(device).float()  # shape: (n, 600, 128)

    # L2 normalize descriptor vectors (dim=2)
    feat_tensor = feat_tensor / torch.norm(feat_tensor, dim=2, keepdim=True).clamp(min=1e-6)

    # Compute pairwise similarity using max-dot (Chamfer) approach
    S = torch.zeros((n, n), device=device)
    for i in range(n):
        # feat_tensor[i]: (600, 128), feat_tensor.transpose(1, 2): (n, 128, 600)
        # matmul output shape: (n, 600, 600)
        dot_products = torch.matmul(feat_tensor[i], feat_tensor.transpose(1, 2))
        max_sim = dot_products.max(dim=2).values  # (n, 600)
        S[i] = max_sim.mean(dim=1)

    # Symmetrize similarity matrix
    S_sym = (S + S.t()) / 2.0
    # Convert similarity to a distance (higher similarity -> lower distance)
    D = (1.0 - S_sym).clamp(min=0.0)
    return D

def compute_chamfer_query_to_db(query_feat, db_feats_list, device, batch_size=500):
    """Computes Chamfer similarity between 1 query and N database images."""
    q = query_feat.to(device)
    # L2 normalize query
    q = q / torch.norm(q, dim=1, keepdim=True).clamp(min=1e-6)
    
    similarities = []
    for i in range(0, len(db_feats_list), batch_size):
        batch = db_feats_list[i:i+batch_size]
        db_tensor = torch.stack(batch).to(device) # (B, 600, 128)
        # L2 normalize db
        db_tensor = db_tensor / torch.norm(db_tensor, dim=2, keepdim=True).clamp(min=1e-6)
        
        # dot_products: (B, 600, 600)
        dot_products = torch.matmul(q, db_tensor.transpose(1, 2))
        
        # q -> DB
        max_sim_q2db = dot_products.max(dim=2).values # (B, 600)
        sim_q2db = max_sim_q2db.mean(dim=1) # (B,)
        
        # DB -> q
        max_sim_db2q = dot_products.max(dim=1).values # (B, 600)
        sim_db2q = max_sim_db2q.mean(dim=1) # (B,)
        
        # Symmetric similarity
        S_sym = (sim_q2db + sim_db2q) / 2.0
        similarities.append(S_sym.cpu().numpy())
        
    return np.concatenate(similarities)

def superglobal_reranking_full(features, K=6, beta=0.31):
    """
    Applies FULL SuperGlobal Re-ranking (MDescAug Graph Diffusion + RerankwMDA).
    features: (M+1, D) array of L2-normalized features. Index 0 is query.
    K: number of nearest neighbors for the graph (and MDA).
    beta: diffusion parameter.
    """
    Q = features[0:1] # (1, D)
    X = features[1:]  # (M, D)
    M = len(X)
    K = K + 1 # including oneself
    
    # 1. Initial similarities
    sim = np.dot(X, Q.T) # (M, 1)
    ranks_trans = np.argsort(-sim, axis=0).T # (1, M)
    
    # =========================================================
    # BƯỚC 1: Xây dựng đồ thị K-Nearest Neighbors (KNN Graph)
    # =========================================================
    X_tensor = X[ranks_trans[0]] # (M, D) - Sắp xếp M ứng viên theo điểm tương đồng ban đầu
    
    # Tính ma trận similarity giữa M ứng viên với nhau (M x M)
    res_ie = np.dot(X_tensor, X_tensor.T) # (M, M)
    
    # Tìm K hàng xóm gần nhất cho mỗi ứng viên
    res_ie_ranks = np.argsort(-res_ie, axis=1)[:, :K] # (M, K) - Index của K hàng xóm
    res_ie_ranks_value = -np.sort(-res_ie, axis=1)[:, :K] # (M, K) - Điểm similarity của K hàng xóm
    
    # Gán trọng số beta cho các hàng xóm, giữ nguyên 1.0 cho bản thân ứng viên (index 0)
    res_ie_ranks_value[:, 1:] *= beta
    res_ie_ranks_value[:, 0] = 1.0
    
    # =========================================================
    # BƯỚC 2: Graph Diffusion (Lan truyền đặc trưng trên đồ thị)
    # =========================================================
    # Cập nhật đặc trưng mới (x_dba) bằng cách lấy trung bình có trọng số từ K hàng xóm
    x_dba = np.zeros_like(X_tensor) # (M, D)
    for i in range(M):
        neighbors = X_tensor[res_ie_ranks[i]] # (K, D) - Lấy vector của K hàng xóm
        weights = res_ie_ranks_value[i, :, None] # (K, 1) - Lấy trọng số tương ứng
        
        # Công thức Diffusion: F_new = Tổng(F_hàng_xóm * Trọng_số) / Tổng(Trọng_số)
        x_dba[i] = np.sum(neighbors * weights, axis=0) / np.sum(weights)
        
    # Similarity with new features
    res_top1000_dba = np.dot(Q, x_dba.T) # (1, M)
    ranks_trans_1000_pre = np.argsort(-res_top1000_dba, axis=1) # (1, M)
    
    rerank_dba_final = ranks_trans[0][ranks_trans_1000_pre[0]] # (M,)
    
    return rerank_dba_final

FEATURE_CACHE = {}

def load_feature_file(path):
    """Loads feature file, transposes if needed, and checks validity with in-memory caching."""
    path_str = str(path)
    if path_str in FEATURE_CACHE:
        return FEATURE_CACHE[path_str]
        
    if not os.path.exists(path):
        return None
    try:
        feat = np.load(path)
        # We expect shape (600, 128)
        if feat.shape[0] < feat.shape[1]:
            feat = feat.T
        if feat.shape == (600, 128):
            tensor_feat = torch.from_numpy(feat).float()
            FEATURE_CACHE[path_str] = tensor_feat
            return tensor_feat
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None

def main():
    script_dir = Path(__file__).parent.resolve()
    BASE_DIR = script_dir.parent.resolve() # c:\Users\ezycloudx-admin\Desktop\seg\main
    
    CANN_DATA = BASE_DIR / "google-research" / "cann"
    DATASETS_DIR = BASE_DIR / "data" / "datasets"
    OUTPUT_DIR = BASE_DIR / "output" / "stage3"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Detect GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    datasets = ['roxford5k', 'rparis6k']
    
    for dataset in datasets:
        FEATURE_CACHE.clear()  # Clear cache to avoid high memory usage between datasets
        print(f"\n==================================================")
        print(f"PROCESSING DATASET: {dataset}")
        print(f"==================================================")
        
        # Load Query and Database Names lists
        query_names_file = CANN_DATA / f"{dataset}_query_names.txt"
        db_names_file = CANN_DATA / f"{dataset}_database_names.txt"
        
        if not query_names_file.exists() or not db_names_file.exists():
            print(f"Missing names mapping files for {dataset}!")
            continue
            
        with open(query_names_file, 'r') as f:
            query_names = [line.strip() for line in f if line.strip()]
        with open(db_names_file, 'r') as f:
            db_names = [line.strip() for line in f if line.strip()]
            
        # Load Ground Truth pickle file
        pkl_path = DATASETS_DIR / dataset / f"gnd_{dataset}.pkl"
        if not pkl_path.exists():
            print(f"Missing ground truth pickle file at: {pkl_path}")
            continue
        with open(pkl_path, 'rb') as f:
            gnd = pickle.load(f)
            
        imlist = gnd['imlist']
        qimlist = gnd['qimlist']
        
        # Setup output matrix
        db_size = len(imlist)
        nq = len(qimlist)
        ranks_matrix = np.zeros((db_size, nq), dtype=np.int32)
        
        # Configuration for L2G
        p = 0.01
        w = 0.19
        k_candidates = 700
        M_sg = 1600
        k_sg = 6
        beta_sg = 0.31

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # --- LOAD ENTIRE DB LOCAL FEATURES ---
        print("Loading all Database local features into memory for Exact Chamfer Base Search...")
        db_local_feats = []
        for img in imlist:
            feat_path = BASE_DIR / "output" / "stage1" / "features" / dataset / "database" / f"{img}.npy"
            feat = load_feature_file(feat_path)
            if feat is None:
                feat = torch.zeros((600, 128))
            db_local_feats.append(feat)
        print(f"Loaded {len(db_local_feats)} DB features.")
        
        # Process each query in ground truth order
        print("\nRunning Exact Chamfer Base Search, MDS & Re-ranking for each query...")
        for q_idx, q_name in enumerate(qimlist):
            if (q_idx + 1) % 1 == 0:
                print(f"  Query {q_idx+1}/{nq}: {q_name}")
                
            feat_path = BASE_DIR / "output" / "stage1" / "features" / dataset / "query" / f"{q_name}.npy"
            query_feat = load_feature_file(feat_path)
            if query_feat is None:
                query_feat = torch.zeros((600, 128))
                
            # EXACT CHAMFER BASE SEARCH (With Caching)
            cache_dir = OUTPUT_DIR / "base_scores" / dataset
            cache_dir.mkdir(parents=True, exist_ok=True)
            score_cache_path = cache_dir / f"{q_name}.npy"
            
            if score_cache_path.exists():
                base_scores = np.load(score_cache_path)
            else:
                base_scores = compute_chamfer_query_to_db(query_feat, db_local_feats, device)
                np.save(score_cache_path, base_scores)
                
            all_sorted_indices = np.argsort(-base_scores)
            all_candidate_names_sorted = [imlist[i] for i in all_sorted_indices]
            
            candidate_names = all_candidate_names_sorted[:k_candidates]
                
            # Build 701 local set list: query + candidates
            local_set_names = [q_name] + candidate_names
            
            # Load local features
            local_features = []
            valid_mask = []
            
            for idx, img_name in enumerate(local_set_names):
                if idx == 0:
                    # Query
                    feat_path = BASE_DIR / "output" / "stage1" / "features" / dataset / "query" / f"{img_name}.npy"
                else:
                    # Database
                    feat_path = BASE_DIR / "output" / "stage1" / "features" / dataset / "database" / f"{img_name}.npy"
                    
                feat = load_feature_file(feat_path)
                if feat is not None:
                    local_features.append(feat)
                    valid_mask.append(True)
                else:
                    # Append dummy to maintain index alignment
                    local_features.append(torch.zeros((600, 128)))
                    valid_mask.append(False)
                    
            # Compute Chamfer Distance Matrix (701 x 701)
            D = compute_chamfer_matrix_pytorch(local_features, device)
            D = D.cpu().numpy()
            
            # Apply power normalization D^p
            D_mod = np.power(D, p)
            np.fill_diagonal(D_mod, 0.0)
            
            # Mask out invalid features (distance = 1.0)
            for i in range(len(valid_mask)):
                for j in range(len(valid_mask)):
                    if (not valid_mask[i] or not valid_mask[j]) and i != j:
                        D_mod[i, j] = 1.0
                        
            # Run MDS for local embeddings (SMACOF)
            # Paper mentions ~5-10 iterations, and eps=0.1
            mds = MDS(n_components=128, dissimilarity="precomputed", random_state=42, max_iter=15, eps=0.1, n_init=1)
            F_mds = mds.fit_transform(D_mod)
            
            # L2 normalize MDS features
            F_mds_norm = F_mds / np.linalg.norm(F_mds, axis=1, keepdims=True).clip(min=1e-6)
            
            # ==========================================
            # LOAD CVNET-R101 GLOBAL FEATURES (2048-dim)
            # ==========================================
            # 1. Get the names of the M_sg candidates (1600 candidates)
            sg_candidate_names = all_candidate_names_sorted[:M_sg]
            full_sg_names = [q_name] + sg_candidate_names
            
            F_global = []
            for i, img_name in enumerate(full_sg_names):
                sub_dir = "query" if i == 0 else "database"
                feat_path = BASE_DIR / "output" / "stage2" / "features" / dataset / sub_dir / f"{img_name}.npy"
                if feat_path.exists():
                    glob = np.load(feat_path) # (2048,)
                else:
                    glob = np.zeros(2048)
                
                # L2 normalize
                norm_val = np.linalg.norm(glob)
                glob_norm = glob / norm_val if norm_val > 1e-6 else np.zeros(2048)
                F_global.append(glob_norm)
                
            F_global = np.array(F_global) # (1601, 2048)
            
            # ==========================================
            # BLENDING SIMILARITY VIA CONCATENATION
            # ==========================================
            # Top 700 + query have MDS (128-d). Remaining 900 candidates have zero MDS.
            F_mds_full = np.zeros((1601, 128))
            F_mds_full[:len(F_mds_norm)] = F_mds_norm
            
            # Paper meant w_local = 0.19, w_global = 0.81 based on SuperGlobal
            w_local = 0.19
            w_global = 0.81
            
            # F_concat = [sqrt(w_local) * F_mds, sqrt(w_global) * F_global]
            F_concat = np.hstack([
                np.sqrt(w_local) * F_mds_full,
                np.sqrt(w_global) * F_global
            ])
            # ==========================================
            # --- 4. FULL SUPERGLOBAL RE-RANKING ---
            
            # Apply Graph Diffusion (MDescAug) + MDA (RerankwMDA)
            # Apply Graph Diffusion (MDescAug) + MDA (RerankwMDA)
            final_ranks = superglobal_reranking_full(F_concat, K=k_sg, beta=beta_sg)
            
            # Final ranking
            sorted_indices = final_ranks
            sorted_candidate_names = [sg_candidate_names[idx] for idx in sorted_indices]
            
            # Build full ranking list of database size
            remaining_db_names = all_candidate_names_sorted[M_sg:]
            full_ranking_names = sorted_candidate_names + remaining_db_names
            
            # Map names to zero-based indices of imlist
            db_name_to_idx = {name: idx for idx, name in enumerate(imlist)}
            ranked_indices = [db_name_to_idx[name] for name in full_ranking_names]
            
            ranks_matrix[:, q_idx] = np.array(ranked_indices, dtype=np.int32)
            
        # Evaluate mAP scores
        print(f"\nEvaluating performance for {dataset}...")
        scores_dict = evaluate_map_metrics(dataset, ranks_matrix, gnd['gnd'])
        
        # Write results to output file
        out_file = OUTPUT_DIR / f"{dataset}_final_results.txt"
        with open(out_file, 'w') as f:
            f.write("====================================================================\n")
            f.write(f"L2G STAGE 3 FINAL RESULTS: {dataset}\n")
            f.write("====================================================================\n")
            f.write(f"Easy mAP:   {scores_dict['easy'] * 100:.2f}%\n")
            f.write(f"Medium mAP: {scores_dict['medium'] * 100:.2f}%\n")
            f.write(f"Hard mAP:   {scores_dict['hard'] * 100:.2f}%\n")
            f.write("====================================================================\n\n")
            
            for q_idx, q_name in enumerate(qimlist):
                f.write(f"Query: {q_name}\n")
                f.write("-" * 80 + "\n")
                f.write("Rank   Database Image\n")
                f.write("-" * 80 + "\n")
                # Retrieve the top 10 database images from ranks_matrix
                for r in range(10):
                    db_idx = ranks_matrix[r, q_idx]
                    f.write(f"{r+1:<6d} {imlist[db_idx]}\n")
                f.write("\n")
        print(f"Saved results to: {out_file}")

        # Save full ranks matrix as .npy cache (dung cho evaluate.py)
        ranks_npy_path = OUTPUT_DIR / f"{dataset}_ranks.npy"
        np.save(str(ranks_npy_path), ranks_matrix)
        print(f"Saved ranks cache to: {ranks_npy_path}")
        
    print("\nSTAGE 3 EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
