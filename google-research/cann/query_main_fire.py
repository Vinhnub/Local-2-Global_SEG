#!/usr/bin/env python3
import os
import sys
import time
import pickle
import importlib.util
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile
from scipy.spatial.distance import cdist

# Setup paths
fire_path = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/fire"

spec = importlib.util.spec_from_file_location("dataset", os.path.join(fire_path, "dataset.py"))
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)

sys.path.insert(0, fire_path)
import fire_network


class FIReExtractor:
    """FIRe feature extractor"""
    
    def __init__(self, fire_path, num_features=600):
        self.fire_path = fire_path
        self.num_features = num_features
        self.net = None
        self.norm_rgb = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {self.device}")
        
        model_path = os.path.join(self.fire_path, "model", "fire_SfM_120k.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print("  Loading FIRe model...")
        state = torch.load(model_path, map_location=self.device)
        state["net_params"]["pretrained"] = None
        self.net = fire_network.init_network(**state["net_params"])
        self.net.load_state_dict(state["state_dict"])
        self.net.to(self.device)
        self.net.eval()
        
        self.norm_rgb = tvf.Normalize(
            **dict(zip(["mean", "std"], self.net.runtime["mean_std"]))
        )
        print("  Model loaded successfully!")
    
    def extract(self, image):
        try:
            img_tensor = tvf.ToTensor()(image).unsqueeze(0)
            img_tensor = self.norm_rgb(img_tensor).to(self.device)
            
            scales = [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25]
            
            with torch.no_grad():
                local_feats = self.net.forward_local(
                    img_tensor,
                    features_num=self.num_features,
                    scales=scales
                )
            
            features = local_feats[0].squeeze(-1).squeeze(0).t()
            return features.cpu().numpy()
        except Exception as e:
            print(f"  Error extracting features: {e}")
            return None


class L2GQueryCANN:
    """
    L2G Query with CANN - ĐÚNG THEO PAPER
    Stage 1: Local Feature Search using CANN
    """
    
    def __init__(self, n_candidates=100, top_k=1600, p=0.01, fire_path=fire_path):
        """
        Args:
            n_candidates: Số DB images để tìm ban đầu (CANN parameter)
            top_k: Số kết quả cuối cùng (paper: 1600)
            p: Power modulation (paper: 0.01)
        """
        self.n_candidates = n_candidates  # CANN parameter: tìm bao nhiêu images ban đầu
        self.top_k = top_k
        self.p = p
        self.fire_path = fire_path
        
        self.db_names = None
        self.db_local_features = None
        self.precomputed_neighbors = None
        self.precomputed_distances = None
        
        self.dataset_name = None
        self.index_dir = None
        self.extractor = None
    
    def init_fire(self):
        if self.extractor is None:
            print("\nInitializing FIRe extractor...")
            self.extractor = FIReExtractor(self.fire_path, num_features=600)
        return self.extractor
    
    def chamfer_similarity(self, query_local, db_local):
        """Asymmetric Chamfer similarity with power modulation"""
        distances = cdist(query_local, db_local, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        avg_distance = np.mean(min_distances)
        similarity = 1.0 / (avg_distance + 1e-8)
        if self.p != 1.0:
            similarity = np.sign(similarity) * np.abs(similarity) ** self.p
        return similarity
    
    def load_index(self, dataset):
        """Load pre-built index"""
        if dataset == 'roxford5k':
            self.index_dir = 'l2g_index_roxford5k'
            self.dataset_name = 'ROxford5K'
        elif dataset == 'rparis6k':
            self.index_dir = 'l2g_index_rparis6k'
            self.dataset_name = 'RParis6K'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        if not os.path.exists(self.index_dir):
            print(f"  ✗ Index not found: {self.index_dir}")
            return False
        
        print(f"\nLoading index: {self.dataset_name}")
        print("-"*60)
        
        # Load database names
        with open(os.path.join(self.index_dir, 'db_names.txt'), 'r') as f:
            self.db_names = [line.strip() for line in f.readlines()]
        print(f"  ✓ Loaded {len(self.db_names)} database names")
        
        # Load local features
        local_file = os.path.join(self.index_dir, 'db_local_features.npz')
        if os.path.exists(local_file):
            data = np.load(local_file, allow_pickle=True)
            self.db_local_features = data['features']
            print(f"  ✓ Loaded local features: {len(self.db_local_features)} images")
        else:
            raise FileNotFoundError(f"Local features not found: {local_file}")
        
        # Load precomputed neighbors (CANN)
        neighbors_file = os.path.join(self.index_dir, 'neighbors.pkl')
        if os.path.exists(neighbors_file):
            with open(neighbors_file, 'rb') as f:
                data = pickle.load(f)
            self.precomputed_neighbors = data['neighbors']
            self.precomputed_distances = data['distances']
            print(f"  ✓ Loaded precomputed neighbors: {len(self.precomputed_neighbors)} images")
            print(f"    Each has {len(self.precomputed_neighbors[0])} neighbors")
        else:
            print(f"  ⚠ Warning: neighbors.pkl not found")
            self.precomputed_neighbors = None
            self.precomputed_distances = None
        
        return True
    
    def load_queries_from_folder(self, folder_path):
        """Load query features from folder"""
        folder_path = Path(folder_path)
        all_features = []
        all_names = []
        
        files = []
        for ext in ['*.npy', '*.jpg', '*.jpeg', '*.png']:
            files.extend(folder_path.glob(ext))
        files = sorted(files)
        
        if not files:
            print(f"  No query files found in {folder_path}")
            return None, None
        
        print(f"\n  Found {len(files)} query files")
        
        for filepath in files:
            print(f"  Processing: {filepath.name}")
            try:
                if filepath.suffix.lower() == '.npy':
                    features = np.load(filepath)
                    if features is not None:
                        all_features.append(features)
                        all_names.append(filepath.stem)
                else:
                    img = Image.open(filepath).convert('RGB')
                    features = self.extractor.extract(img)
                    if features is not None:
                        all_features.append(features)
                        all_names.append(filepath.stem)
            except Exception as e:
                print(f"    Error: {e}")
        
        if not all_features:
            return None, None
        
        print(f"\n  Total: {len(all_features)} queries loaded")
        return all_features, all_names
    
    def search_cann(self, query_features, query_names):
        """
        CANN Search - ĐÚNG THEO PAPER:
        
        Step 1: Query tìm top-n_candidates DB images gần nhất
                (dùng Chamfer similarity với toàn bộ DB)
        
        Step 2: Lấy pre-computed neighbors của các DB images đó
        
        Step 3: Accumulate scores từ các neighbors
        
        Step 4: Sort và lấy top-k kết quả cuối cùng
        """
        print(f"\n{'='*80}")
        print(f"CANN SEARCH - STAGE 1 (LOCAL FEATURE SEARCH)")
        print(f"{'='*80}")
        print(f"  Queries: {len(query_features)}")
        print(f"  Database: {len(self.db_names)} images")
        print(f"  Step 1: Find top-{self.n_candidates} DB images")
        print(f"  Step 2: Accumulate from precomputed neighbors")
        print(f"  Step 3: Return top-{self.top_k} results")
        print(f"  Similarity: Chamfer (asymmetric) with p={self.p}")
        print("-"*80)
        
        results = {}
        timing = {
            'total': 0,
            'step1_chamfer': 0,
            'step2_accumulate': 0,
            'step3_sort': 0,
            'per_query': []
        }
        
        total_start = time.time()
        
        for i, query_feat in enumerate(query_features):
            q_start = time.time()
            
            # ===== STEP 1: Find top-n_candidates DB images =====
            step1_start = time.time()
            similarities = []
            for db_feat in self.db_local_features:
                sim = self.chamfer_similarity(query_feat, db_feat)
                similarities.append(sim)
            
            # Get top-n_candidates indices
            top_candidates_idx = np.argsort(similarities)[-self.n_candidates:][::-1]
            step1_time = time.time() - step1_start
            
            # ===== STEP 2: Accumulate from precomputed neighbors =====
            step2_start = time.time()
            candidate_scores = {}
            
            # Add direct scores from top candidates
            for idx in top_candidates_idx:
                candidate_scores[idx] = similarities[idx]
            
            # Accumulate from precomputed neighbors
            if self.precomputed_neighbors is not None:
                for db_idx in top_candidates_idx:
                    neighbors = self.precomputed_neighbors[db_idx]
                    distances = self.precomputed_distances[db_idx]
                    
                    for n_idx, dist in zip(neighbors, distances):
                        if n_idx in candidate_scores:
                            candidate_scores[n_idx] += dist * 0.3  # Weight
                        else:
                            candidate_scores[n_idx] = dist * 0.2
            
            step2_time = time.time() - step2_start
            
            # ===== STEP 3: Sort and get top-k =====
            step3_start = time.time()
            sorted_candidates = sorted(
                candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.top_k]
            step3_time = time.time() - step3_start
            
            q_time = time.time() - q_start
            
            # Save results
            query_name = query_names[i] if i < len(query_names) else f"query_{i:04d}"
            results[query_name] = []
            
            for rank, (idx, score) in enumerate(sorted_candidates, 1):
                db_name = self.db_names[idx] if idx < len(self.db_names) else f"db_{idx:06d}"
                results[query_name].append((rank, db_name, float(score)))
            
            # Update timing
            timing['step1_chamfer'] += step1_time
            timing['step2_accumulate'] += step2_time
            timing['step3_sort'] += step3_time
            timing['total'] += q_time
            timing['per_query'].append({
                'query': query_name,
                'total': q_time,
                'step1': step1_time,
                'step2': step2_time,
                'step3': step3_time,
                'candidates': len(sorted_candidates)
            })
            
            # Print progress
            print(f"  [{i+1:3d}/{len(query_features)}] {query_name:30s} "
                  f"{q_time:.4f}s | candidates={len(sorted_candidates)}")
        
        # Summary
        avg_time = timing['total'] / len(query_features)
        print(f"\n{'='*80}")
        print("TIMING SUMMARY:")
        print(f"  Total queries: {len(query_features)}")
        print(f"  Total time: {timing['total']:.3f}s")
        print(f"  Average per query: {avg_time:.4f}s")
        print(f"    - Step 1 (Chamfer): {timing['step1_chamfer']/len(query_features):.4f}s "
              f"({timing['step1_chamfer']/timing['total']*100:.1f}%)")
        print(f"    - Step 2 (Accumulate): {timing['step2_accumulate']/len(query_features):.4f}s "
              f"({timing['step2_accumulate']/timing['total']*100:.1f}%)")
        print(f"    - Step 3 (Sort): {timing['step3_sort']/len(query_features):.4f}s "
              f"({timing['step3_sort']/timing['total']*100:.1f}%)")
        print(f"{'='*80}")
        
        return results, timing
    
    def save_results(self, results, output_file, dataset_name, timing=None):
        """Save search results"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"L2G CANN RESULTS - STAGE 1\n")
            f.write(f"{'='*100}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Index: {self.index_dir}\n")
            f.write(f"Total queries: {len(results)}\n")
            f.write(f"Top-{self.top_k} results per query\n")
            f.write(f"CANN n_candidates: {self.n_candidates}\n")
            f.write(f"Similarity: Chamfer (asymmetric) with p={self.p}\n")
            
            if timing:
                f.write(f"\nTIMING:\n")
                f.write(f"  Total: {timing['total']:.3f}s\n")
                f.write(f"  Average: {timing['total']/len(results):.4f}s/query\n")
            
            f.write("="*100 + "\n\n")
            
            for query_name, neighbors in results.items():
                f.write(f"Query: {query_name}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Rank':<8} {'Database Image':<50} {'Score':<20}\n")
                f.write("-"*80 + "\n")
                
                # Show top 20
                for rank, db_name, score in neighbors[:20]:
                    f.write(f"{rank:<8} {db_name:<50} {score:.8f}\n")
                
                if len(neighbors) > 20:
                    f.write(f"... ({len(neighbors) - 20} more results)\n")
                
                f.write(f"\nTotal: {len(neighbors)} results\n")
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"\n✓ Results saved to: {output_file}")


def main():
    print("="*80)
    print("L2G QUERY - CANN SEARCH (STAGE 1)")
    print("Paper: Global-to-Local or Local-to-Global?")
    print("="*80)
    
    query_root = Path("query")
    if not query_root.exists():
        query_root.mkdir(parents=True)
        print(f"\nCreated: {query_root}/")
        print("Please add query files to:")
        print("  - query/roxford5k/")
        print("  - query/rparis6k/")
        return
    
    # Parameters from paper
    N_CANDIDATES = 100  # CANN parameter: number of initial candidates
    TOP_K = 1600        # Final results for re-ranking
    P = 0.01            # Power modulation
    
    datasets = [
        {
            'name': 'roxford5k',
            'display': 'ROxford5K',
            'folder': query_root / 'roxford5k',
            'output': query_root / 'roxford5k_cann_results.txt'
        },
        {
            'name': 'rparis6k',
            'display': 'RParis6K',
            'folder': query_root / 'rparis6k',
            'output': query_root / 'rparis6k_cann_results.txt'
        }
    ]
    
    q = L2GQueryCANN(
        n_candidates=N_CANDIDATES,
        top_k=TOP_K,
        p=P,
        fire_path=fire_path
    )
    q.init_fire()
    
    for ds in datasets:
        print(f"\n{'='*80}")
        print(f"{ds['display']}")
        print(f"{'='*80}")
        
        if not ds['folder'].exists():
            ds['folder'].mkdir(parents=True, exist_ok=True)
            print(f"Created folder: {ds['folder']}")
            continue
        
        if not q.load_index(ds['name']):
            continue
        
        query_features, query_names = q.load_queries_from_folder(ds['folder'])
        
        if query_features is None:
            print(f"No queries found in {ds['folder']}")
            continue
        
        results, timing = q.search_cann(query_features, query_names)
        q.save_results(results, ds['output'], ds['display'], timing)
    
    print("\n" + "="*80)
    print("✓ QUERY COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()