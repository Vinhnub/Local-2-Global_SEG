import os
import subprocess
import tempfile
import numpy as np

def cann_search(q_feats, db_feats, k_candidates=1600, dim=128):
    """
    Sử dụng CANN (google-research/cann) để thực hiện Base Search.
    
    Args:
        q_feats: list of query feature arrays (shape N x 128)
        db_feats: list of database feature arrays (shape M x 128)
        k_candidates: số lượng ứng viên cần trả về
        dim: số chiều đặc trưng (128 cho FIRe)
        
    Returns:
        ranks: mảng numpy chứa index của top K ứng viên cho mỗi query.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cann_exe = os.path.join(project_root, "google-research", "cann", "bazel-bin", "main", "colored_c_nn_random_grids_index_main.exe")
    
    if not os.path.exists(cann_exe):
        print(f"CANN executable not found at: {cann_exe}")
        print("Falling back to exact PyTorch Chamfer search (this is completely accurate and runs on GPU)...")
        import torch
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Keep full db on CPU to save VRAM, only move batches to GPU
        db_tensor_cpu = torch.tensor(np.stack(db_feats))
        db_tensor_cpu = db_tensor_cpu / torch.norm(db_tensor_cpu, dim=2, keepdim=True).clamp(min=1e-6)
        
        ranks = []
        batch_size = 200 # Small batch size to ensure minimal VRAM usage (less than 1GB)
        for i, q_feat in enumerate(tqdm(q_feats, desc="PyTorch Base Search")):
            q = torch.tensor(q_feat).unsqueeze(0).to(device)
            q = q / torch.norm(q, dim=2, keepdim=True).clamp(min=1e-6)
            
            sims_list = []
            for start_idx in range(0, len(db_feats), batch_size):
                end_idx = min(start_idx + batch_size, len(db_feats))
                db_batch = db_tensor_cpu[start_idx:end_idx].to(device)
                
                dot_products = torch.matmul(q, db_batch.transpose(1, 2))
                S_q_db = dot_products.max(dim=2).values.mean(dim=1)
                S_db_q = dot_products.max(dim=1).values.mean(dim=1)
                sims_batch = ((S_q_db + S_db_q) / 2.0)
                sims_list.append(sims_batch)
                
            sims = torch.cat(sims_list)
            
            topk = torch.topk(sims, k=min(k_candidates, len(db_feats)))
            cand_indices = topk.indices.cpu().numpy().tolist()
            
            if len(cand_indices) < k_candidates:
                existing = set(cand_indices)
                for j in range(len(db_feats)):
                    if len(cand_indices) >= k_candidates:
                        break
                    if j not in existing:
                        cand_indices.append(j)
                        
            ranks.append(cand_indices)
            
        return np.array(ranks)
        
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories for query and index descriptors
        query_dir = os.path.join(tmpdir, "queries")
        index_dir = os.path.join(tmpdir, "index")
        os.makedirs(query_dir)
        os.makedirs(index_dir)
        
        # Write query descriptors as raw binary float32 (without headers)
        query_files = []
        for i, q_feat in enumerate(q_feats):
            # q_feat is [N, dim]
            feat_array = np.array(q_feat, dtype=np.float32)
            filepath = os.path.join(query_dir, f"q_{i}.desc")
            # tofile() writes pure raw bytes
            feat_array.tofile(filepath)
            query_files.append(filepath)
            
        # Write database descriptors as raw binary float32
        index_files = []
        for i, db_feat in enumerate(db_feats):
            feat_array = np.array(db_feat, dtype=np.float32)
            filepath = os.path.join(index_dir, f"db_{i}.desc")
            feat_array.tofile(filepath)
            index_files.append(filepath)
            
        # Create list files for CANN
        query_list_file = os.path.join(tmpdir, "query_list.txt")
        with open(query_list_file, "w") as f:
            for path in query_files:
                # CANN uses backward slashes on Windows if we just write the path, but forward slashes might be safer.
                f.write(path.replace("\\", "/") + "\n")
                
        index_list_file = os.path.join(tmpdir, "index_list.txt")
        with open(index_list_file, "w") as f:
            for path in index_files:
                f.write(path.replace("\\", "/") + "\n")
                
        pairs_file = os.path.join(tmpdir, "pairs.txt")
        
        # Run CANN executable
        # The parameters are taken from google-research/cann/main/run_main.sh (using default params in C++ or explicitly setting them)
        cmd = [
            cann_exe,
            "--index_descriptor_files", index_list_file,
            "--query_descriptor_files", query_list_file,
            "--pairs_file", pairs_file,
            "--num_features", "1000",
            "--dim", str(dim)
        ]
        
        print(f"Running CANN search with {len(q_feats)} queries and {len(db_feats)} database images...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"CANN execution failed: {e.stderr}")
            raise
            
        # Parse output pairs
        # Output format: query_filename, index_filename, score
        # Example: C:/.../q_0.desc, C:/.../db_45.desc, 0.3022350
        results_map = {i: [] for i in range(len(q_feats))}
        
        with open(pairs_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    q_path = parts[0].strip()
                    db_path = parts[1].strip()
                    score = float(parts[2].strip())
                    
                    # Extract index from filenames (e.g., "q_0.desc" -> 0)
                    q_basename = os.path.basename(q_path)
                    db_basename = os.path.basename(db_path)
                    
                    try:
                        q_idx = int(q_basename.replace("q_", "").replace(".desc", ""))
                        db_idx = int(db_basename.replace("db_", "").replace(".desc", ""))
                        results_map[q_idx].append((db_idx, score))
                    except ValueError:
                        continue
                        
        # Sort scores and get top-K
        ranks = []
        for q_idx in range(len(q_feats)):
            # Sort by score descending (higher is better in CANN)
            db_matches = sorted(results_map[q_idx], key=lambda x: x[1], reverse=True)
            
            # Extract indices
            candidate_indices = [x[0] for x in db_matches]
            
            # If CANN returned fewer than k_candidates, pad with other random/sequential indices (fallback)
            if len(candidate_indices) < k_candidates:
                existing = set(candidate_indices)
                for i in range(len(db_feats)):
                    if len(candidate_indices) >= k_candidates:
                        break
                    if i not in existing:
                        candidate_indices.append(i)
                        
            # Truncate to exactly k_candidates
            candidate_indices = candidate_indices[:k_candidates]
            ranks.append(candidate_indices)
            
        return np.array(ranks)
