import os
import sys
sys.path.append(str(BASE_DIR / 'src'))
import pickle
import numpy as np
import time
from pathlib import Path
from sklearn.manifold import MDS
import warnings

# Append to path
script_dir = Path(__file__).parent.resolve()
BASE_DIR = script_dir.parent
sys.path.append(str(script_dir))
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "google-research" / "asmk"))
sys.path.append(str(BASE_DIR / "fire" / "lib" / "asmk"))

import torch
from asmk.asmk_method import ASMKMethod
from online.stage4_search.run_cann_search import cann_search

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Helper evaluation functions from revisitop
def compute_ap(ranks, nres):
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

        pos = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.isin(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        pos += 1 
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
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

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
    print("====================================================================\n")
    return {'easy': mapE, 'medium': mapM, 'hard': mapH}

def superglobal_reranking_full(features, K=6, beta=0.31):
    Q = features[0:1] # (1, D)
    X = features[1:]  # (M, D)
    M = len(X)
    K = K + 1 
    
    sim = np.dot(X, Q.T) # (M, 1)
    ranks_trans = np.argsort(-sim, axis=0).T # (1, M)
    
    X_tensor = X[ranks_trans[0]] 
    res_ie = np.dot(X_tensor, X_tensor.T) # (M, M)
    
    res_ie_ranks = np.argsort(-res_ie, axis=1)[:, :K] 
    res_ie_ranks_value = -np.sort(-res_ie, axis=1)[:, :K] 
    
    res_ie_ranks_value[:, 1:] *= beta
    res_ie_ranks_value[:, 0] = 1.0
    
    x_dba = np.zeros_like(X_tensor)
    for i in range(M):
        neighbors = X_tensor[res_ie_ranks[i]] 
        weights = res_ie_ranks_value[i, :, None] 
        x_dba[i] = np.sum(neighbors * weights, axis=0) / np.sum(weights)
        
    res_top1000_dba = np.dot(Q, x_dba.T) # (1, M)
    ranks_trans_1000_pre = np.argsort(-res_top1000_dba, axis=1) # (1, M)
    rerank_dba_final = ranks_trans[0][ranks_trans_1000_pre[0]] # (M,)
    return rerank_dba_final

FEATURE_CACHE = {}

def load_feature_file(path):
    path_str = str(path)
    if path_str in FEATURE_CACHE:
        return FEATURE_CACHE[path_str]
        
    if not os.path.exists(path):
        return None
    try:
        feat = np.load(path)
        if feat.shape[0] < feat.shape[1]:
            feat = feat.T
        if feat.shape == (600, 128):
            FEATURE_CACHE[path_str] = feat
            return feat
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None

def compute_chamfer_matrix_pytorch(feat_tensor, device):
    feat_tensor = feat_tensor / torch.norm(feat_tensor, dim=2, keepdim=True).clamp(min=1e-6)
    n = feat_tensor.shape[0]
    S = torch.zeros((n, n), device=device)
    for i in range(n):
        dot_products = torch.matmul(feat_tensor[i], feat_tensor.transpose(1, 2))
        max_sim = dot_products.max(dim=2).values
        S[i] = max_sim.mean(dim=1)
    return S

def main():
    CANN_DATA = BASE_DIR / "google-research" / "cann"
    DATASETS_DIR = BASE_DIR / "data" / "datasets"
    OUTPUT_DIR = BASE_DIR / "output" / "stage3"
    ASMK_CACHE_DIR = OUTPUT_DIR / "asmk"
    ASMK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    datasets = ['roxford5k', 'rparis6k']
    
    # Configuration for L2G
    p = 0.01
    w_local = 0.19
    w_global = 0.81
    k_candidates = 700
    M_sg = 1600
    k_sg = 6
    beta_sg = 0.31
    
    asmk_params = {
        "index": {"gpu_id": 0}, # Dùng GPU Faiss để chạy KMeans siêu tốc
        "train_codebook": {"codebook": {"size": "64k"}},
        "build_ivf": {
            "kernel": {"binary": True},
            "ivf": {"use_idf": False},
            "quantize": {"multiple_assignment": 1},
            "aggregate": {}
        },
        "query_ivf": {
            "quantize": {"multiple_assignment": 5},
            "aggregate": {},
            "search": {"topk": None},
            "similarity": {"similarity_threshold": 0.0, "alpha": 3.0}
        }
    }
    
    for dataset in datasets:
        FEATURE_CACHE.clear()
        print(f"\n==================================================")
        print(f"PROCESSING DATASET: {dataset}")
        print(f"==================================================")
        
        pkl_path = DATASETS_DIR / dataset / f"gnd_{dataset}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, 'rb') as f:
            gnd = pickle.load(f)
            
        imlist = gnd['imlist']
        qimlist = gnd['qimlist']
        db_size = len(imlist)
        nq = len(qimlist)
        ranks_matrix = np.zeros((db_size, nq), dtype=np.int32)
        
        # --- LOAD ENTIRE DB LOCAL FEATURES ---
        print("Loading all Database local features for ASMK...")
        db_local_feats = []
        db_imids = []
        for idx, img in enumerate(imlist):
            feat_path = BASE_DIR / "output" / "stage1" / "features" / dataset / "database" / f"{img}.npy"
            feat = load_feature_file(feat_path)
            if feat is None:
                feat = np.zeros((600, 128), dtype=np.float32)
            db_local_feats.append(feat)
            db_imids.append(np.full(600, idx, dtype=np.int32))
            
        vecs = np.vstack(db_local_feats).astype(np.float32)
        imids = np.hstack(db_imids).astype(np.int32)
        
        sparse_sim_path = OUTPUT_DIR / f"{dataset}_sparse_sim.pkl"
        print(f"Loading sparse similarity index from {sparse_sim_path}...")
        if not sparse_sim_path.exists():
            print(f"ERROR: Sparse index {sparse_sim_path} not found! Please run build_index.py first.")
            return
        with open(sparse_sim_path, 'rb') as f:
            sparse_sim = pickle.load(f)
            
        print("Initializing ASMK Method for SUPER FAST execution...")
        asmk_method = ASMKMethod.initialize_untrained(asmk_params)
        
        codebook_path = ASMK_CACHE_DIR / f"{dataset}_codebook.pkl"
        if codebook_path.exists():
            print(f"Loading cached codebook from {codebook_path}...")
            asmk_method = asmk_method.train_codebook(None, cache_path=codebook_path)
        else:
            print("Training ASMK Codebook on GPU...", flush=True)
            asmk_method = asmk_method.train_codebook(vecs, cache_path=codebook_path)
            
        ivf_path = ASMK_CACHE_DIR / f"{dataset}_ivf.pkl"
        if ivf_path.exists():
            print(f"Loading cached IVF from {ivf_path}...")
            asmk_dataset = asmk_method.build_ivf(None, None, cache_path=ivf_path)
        else:
            print("Building ASMK IVF...", flush=True)
            asmk_dataset = asmk_method.build_ivf(vecs, imids, cache_path=ivf_path)
            
        print("Loading all Query local features...")
        q_local_feats = []
        for q_idx, q_name in enumerate(qimlist):
            feat_path = BASE_DIR / "output" / "stage1" / "features" / dataset / "query" / f"{q_name}.npy"
            q_feat = load_feature_file(feat_path)
            if q_feat is None:
                q_feat = np.zeros((600, 128), dtype=np.float32)
            q_local_feats.append(q_feat)
        
        print(f"\nRunning MDS & Re-ranking for {nq} queries...", flush=True)
        t_start = time.time()
        for q_idx, q_name in enumerate(qimlist):
            if (q_idx + 1) % 1 == 0:
                print(f"  Processed {q_idx+1}/{nq} queries...", flush=True)
                
            q_feat = q_local_feats[q_idx]
            
            # 1. Base Search via ASMK
            qvecs = q_feat.astype(np.float32)
            qimids_arr = np.zeros(600, dtype=np.int32)
            
            _, _, asmk_ranks, asmk_scores = asmk_dataset.query_ivf(qvecs, qimids_arr)
            all_sorted_indices = asmk_ranks[0]
            base_scores = asmk_scores[0]
            
            # Map indices to names
            all_candidate_names_sorted = [imlist[i] for i in all_sorted_indices]
            
            # If the database returns less than M_sg, pad with remaining
            if len(all_candidate_names_sorted) < db_size:
                missing = [name for name in imlist if name not in all_candidate_names_sorted]
                all_candidate_names_sorted.extend(missing)
                all_sorted_indices = np.append(all_sorted_indices, [imlist.index(name) for name in missing])
            
            candidate_names = all_candidate_names_sorted[:k_candidates]
            sg_candidate_names = all_candidate_names_sorted[:M_sg]
            
            # 2. Local Similarity Matrix for 701 images
            k = len(candidate_names)
            W_local = np.zeros((k+1, k+1), dtype=np.float32)
            
            # (a) Compute Exact Chamfer for Query vs top-K
            cand_indices = all_sorted_indices[:k]
            cand_feats = [db_local_feats[idx] for idx in cand_indices]
            
            q = torch.tensor(q_feat).unsqueeze(0).to(device)
            db = torch.tensor(np.stack(cand_feats)).to(device)
            q = q / torch.norm(q, dim=2, keepdim=True).clamp(min=1e-6)
            db = db / torch.norm(db, dim=2, keepdim=True).clamp(min=1e-6)
            
            dot_products = torch.matmul(q, db.transpose(1, 2))
            S_q_db = dot_products.max(dim=2).values.mean(dim=1)
            S_db_q = dot_products.max(dim=1).values.mean(dim=1)
            q_sims = ((S_q_db + S_db_q) / 2.0).cpu().numpy()
            
            W_local[0, 1:] = q_sims
            W_local[1:, 0] = q_sims
            
            # (b) Lookup sparse index for candidate vs candidate
            for r in range(k):
                idx_r = cand_indices[r]
                for c in range(r, k):
                    idx_c = cand_indices[c]
                    if idx_r == idx_c:
                        sim_val = 1.0
                    else:
                        sim_val = sparse_sim.get(idx_r, {}).get(idx_c, 0.0)
                    
                    W_local[r+1, c+1] = sim_val
                    W_local[c+1, r+1] = sim_val
            
            # Convert ASMK similarity W_local to distance D_mod
            max_sim = np.max(W_local)
            if max_sim > 0:
                W_norm = W_local / max_sim
            else:
                W_norm = W_local
            D = np.clip(1.0 - W_norm, 0.0, 1.0)
            D_mod = np.power(D, p)
            np.fill_diagonal(D_mod, 0.0)
            
            # 3. MDS
            mds = MDS(n_components=128, dissimilarity="precomputed", random_state=42, max_iter=15, eps=0.1, n_init=1)
            F_mds = mds.fit_transform(D_mod)
            F_mds_norm = F_mds / np.linalg.norm(F_mds, axis=1, keepdims=True).clip(min=1e-6)
            
            # 4. LOAD CVNET GLOBAL FEATURES
            full_sg_names = [q_name] + sg_candidate_names
            F_global = []
            for i, img_name in enumerate(full_sg_names):
                sub_dir = "query" if i == 0 else "database"
                feat_path = BASE_DIR / "output" / "stage2" / "features" / dataset / sub_dir / f"{img_name}.npy"
                if os.path.exists(feat_path):
                    glob = np.load(feat_path)
                else:
                    glob = np.zeros(2048)
                norm_val = np.linalg.norm(glob)
                glob_norm = glob / norm_val if norm_val > 1e-6 else np.zeros(2048)
                F_global.append(glob_norm)
            F_global = np.array(F_global) # (M_sg+1, 2048)
            
            # 5. BLENDING SIMILARITY
            F_mds_full = np.zeros((len(F_global), 128))
            F_mds_full[:len(F_mds_norm)] = F_mds_norm
            F_concat = np.hstack([
                np.sqrt(w_local) * F_mds_full,
                np.sqrt(w_global) * F_global
            ])
            
            # 6. SUPERGLOBAL RE-RANKING
            final_ranks = superglobal_reranking_full(F_concat, K=k_sg, beta=beta_sg)
            sorted_candidate_names = [sg_candidate_names[idx] for idx in final_ranks]
            
            remaining_db_names = all_candidate_names_sorted[M_sg:]
            full_ranking_names = sorted_candidate_names + remaining_db_names
            
            db_name_to_idx = {name: idx for idx, name in enumerate(imlist)}
            ranked_indices = [db_name_to_idx[name] for name in full_ranking_names]
            
            ranks_matrix[:, q_idx] = np.array(ranked_indices, dtype=np.int32)
            
        print(f"Finished {nq} queries in {time.time() - t_start:.2f}s!")
        print(f"\nEvaluating performance for {dataset}...")
        scores_dict = evaluate_map_metrics(dataset, ranks_matrix, gnd['gnd'])
        
        out_file = OUTPUT_DIR / f"{dataset}_fast_asmk_final_results.txt"
        with open(out_file, 'w') as f:
            f.write("====================================================================\n")
            f.write(f"L2G STAGE 3 FINAL RESULTS: {dataset}\n")
            f.write("====================================================================\n")
            f.write(f"Easy mAP:   {scores_dict['easy'] * 100:.2f}%\n")
            f.write(f"Medium mAP: {scores_dict['medium'] * 100:.2f}%\n")
            f.write(f"Hard mAP:   {scores_dict['hard'] * 100:.2f}%\n")
            f.write("====================================================================\n\n")
            
        ranks_npy_path = OUTPUT_DIR / f"{dataset}_fast_asmk_ranks.npy"
        np.save(str(ranks_npy_path), ranks_matrix)
        
    print("\nSTAGE 3 EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
