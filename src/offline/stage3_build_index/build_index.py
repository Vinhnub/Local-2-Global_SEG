import os
import sys
import pickle
import numpy as np
import time
import torch
from pathlib import Path
from tqdm import tqdm

script_dir = Path(__file__).parent.resolve()
BASE_DIR = ''
#sys.path.append(str(BASE_DIR / 'src'))

from src.online.stage4_search.run_cann_search import cann_search

def load_feature_file(path):
    if not os.path.exists(path):
        return None
    try:
        feat = np.load(path)
        if feat.shape[0] < feat.shape[1]:
            feat = feat.T
        if feat.shape == (600, 128):
            return feat
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None

def compute_symmetric_chamfer(q_feat, cand_feats, device):
    """
    Computes symmetric Chamfer similarity between a single query and N candidates.
    q_feat: (600, 128)
    cand_feats: (N, 600, 128)
    Returns: (N,) symmetric similarity array
    """
    q = torch.tensor(q_feat).unsqueeze(0).to(device)
    db = torch.tensor(cand_feats).to(device)
    
    q = q / torch.norm(q, dim=2, keepdim=True).clamp(min=1e-6)
    db = db / torch.norm(db, dim=2, keepdim=True).clamp(min=1e-6)
    
    dot_products = torch.matmul(q, db.transpose(1, 2)) # (N, 600, 600)
    
    S_q_db = dot_products.max(dim=2).values.mean(dim=1) # (N,)
    S_db_q = dot_products.max(dim=1).values.mean(dim=1) # (N,)
    
    return ((S_q_db + S_db_q) / 2.0).cpu().numpy()

def build_index(dataset_name, project_root):
    base_dir = Path(project_root) / 'output' / 'stage1' / 'features' / dataset_name
    db_dir = base_dir / 'database'
    print(db_dir)
    if not db_dir.is_dir():
        print(f'No database features found for dataset {dataset_name}. Skipping indexing.')
        return

    # Load gnd to get exact imlist order
    pkl_path = Path(project_root) / "data" / "datasets" / dataset_name / f"gnd_{dataset_name}.pkl"
    if not pkl_path.exists():
        print(f"Ground truth not found for {dataset_name}. Skipping.")
        return
        
    with open(pkl_path, 'rb') as f:
        gnd = pickle.load(f)
    imlist = gnd['imlist']

    print(f"Loading {len(imlist)} database features for {dataset_name}...")
    db_local_feats = []
    for img in imlist:
        feat_path = db_dir / f"{img}.npy"
        feat = load_feature_file(feat_path)
        if feat is None:
            feat = np.zeros((600, 128), dtype=np.float32)
        db_local_feats.append(feat)

    k_candidates = 701 # top-k nearest index images + itself
    print(f"Running CANN search (Offline DB vs DB) for top-{k_candidates} candidates...")
    t0 = time.time()
    cann_ranks = cann_search(db_local_feats, db_local_feats, k_candidates=k_candidates)
    print(f"CANN search finished in {time.time() - t0:.1f}s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for computing exact Chamfer sparse similarities...")

    sparse_sim = {}
    
    t1 = time.time()
    db_feats_np = np.stack(db_local_feats) # (M, 600, 128)
    
    # Process batch by batch
    batch_size = 100
    for i in tqdm(range(0, len(db_local_feats), batch_size), desc="Computing Chamfer similarities"):
        batch_indices = range(i, min(i + batch_size, len(db_local_feats)))
        for idx in batch_indices:   
            q_feat = db_local_feats[idx]
            cand_indices = cann_ranks[idx]
            
            # Filter out itself to not store it redundantly, but actually let's just store all top-k
            cand_feats = db_feats_np[cand_indices]
            
            # Compute exact symmetric Chamfer similarity
            sims = compute_symmetric_chamfer(q_feat, cand_feats, device)
            
            if idx not in sparse_sim:
                sparse_sim[idx] = {}
                
            for c_idx, sim in zip(cand_indices, sims):
                sparse_sim[idx][c_idx] = float(sim)
                # Ensure symmetry in dictionary
                if c_idx not in sparse_sim:
                    sparse_sim[c_idx] = {}
                sparse_sim[c_idx][idx] = float(sim)
                
    print(f"Exact Chamfer computation finished in {time.time() - t1:.1f}s")

    # Save sparse index
    out_dir = Path(project_root) / 'output' / 'stage3'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}_sparse_sim.pkl"
    
    print(f"Saving sparse similarity index to {out_file}...")
    with open(out_file, 'wb') as f:
        pickle.dump(sparse_sim, f)
        
    print(f"Done building offline index for {dataset_name}!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build Sparse Similarity Index of FIRe features')
    parser.add_argument('--dataset', default='roxford5k', help='Dataset name (e.g., roxford5k or rparis6k)')
    parser.add_argument('--project_root', default=str(BASE_DIR), help='Root directory of the project')
    args = parser.parse_args()
    build_index(args.dataset, args.project_root)
