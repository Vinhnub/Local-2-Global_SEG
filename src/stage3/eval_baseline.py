"""
Đánh giá Baseline (Đường cơ sở)

Script này dùng để đánh giá chất lượng thô của bộ đặc trưng SuperGlobal (CVNet) 
mà KHÔNG sử dụng qua các thuật toán Stage 2 hay Stage 3.
Việc này cung cấp điểm số mAP cơ sở để so sánh xem pipeline Local-to-Global 
kéo được bao nhiêu điểm so với việc chỉ dùng đặc trưng gốc.
"""
import os
import sys
import pickle
import json
import torch
import numpy as np

# Đảm bảo import được module từ thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.stage3.evaluate_map import evaluate_map_metrics

def eval_baseline(dataset_name='roxford5k'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(os.path.dirname(project_root), "data_stage2")
    
    gnd_path = os.path.join(os.path.dirname(project_root), "datasets", dataset_name, f"gnd_{dataset_name}.pkl")
    with open(gnd_path, 'rb') as f:
        gnd_data = pickle.load(f)
    gnd = gnd_data['gnd']
    qimlist = gnd_data['qimlist']
    imlist = gnd_data['imlist']
    
    pth_path = os.path.join(data_dir, f"feats_{dataset_name}_RN50.pth")
    super_global_features = torch.load(pth_path)
    names = super_global_features["names"]
    feats = super_global_features["feats"]
    
    name2idx = {name: i for i, name in enumerate(names)}
    
    # Load CANN results
    cann_path = os.path.join(data_dir, f"{dataset_name}_query_results.json")
    with open(cann_path, 'r', encoding='utf-8') as f:
        cann_data = json.load(f)
    queries = {q['query_name']: q['results'] for q in cann_data['queries']}
    
    ranks = np.zeros((len(imlist), len(qimlist)), dtype=np.int32)
    
    for i, q_name in enumerate(qimlist):
        q_lookup = q_name if q_name in name2idx else f"{q_name}.jpg"
        if q_lookup in name2idx:
            q_idx = name2idx[q_lookup]
            q_feat = feats[q_idx]
            q_feat = q_feat / torch.norm(q_feat).clamp(min=1e-6)
        else:
            q_feat = torch.zeros(2048)
            
        cann_results = queries.get(q_name, [])[:1600]
        cand_names = [r['db_name'] for r in cann_results]
        
        db_feats = []
        for db_name in cand_names:
            db_lookup = db_name if db_name in name2idx else f"{db_name}.jpg"
            if db_lookup in name2idx:
                db_feat = feats[name2idx[db_lookup]]
                db_feats.append(db_feat / torch.norm(db_feat).clamp(min=1e-6))
            else:
                db_feats.append(torch.zeros_like(q_feat))
                
        if len(db_feats) > 0:
            db_feats = torch.stack(db_feats)
            scores = torch.matmul(db_feats, q_feat).cpu().numpy()
            
            # Sort candidates by score
            sorted_indices = np.argsort(-scores)
            sorted_candidate_names = [cand_names[idx] for idx in sorted_indices]
            
            # Build full ranking list
            remaining_db_names = [name for name in imlist if name not in cand_names]
            full_ranking_names = sorted_candidate_names + remaining_db_names
            
            # Map names to indices
            for j, db_name in enumerate(full_ranking_names):
                ranks[j, i] = imlist.index(db_name)
        else:
            # Fallback
            ranks[:, i] = np.arange(len(imlist))
        
    evaluate_map_metrics(dataset_name, ranks, gnd)

if __name__ == "__main__":
    eval_baseline("roxford5k")
