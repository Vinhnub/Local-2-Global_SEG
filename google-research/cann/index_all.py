#!/usr/bin/env python3
import numpy as np
import os
import time
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class L2GIndexer:
    def __init__(self, k=700, M=1600, p=0.01, num_local=600):
        """
        Args:
            k: số neighbors precompute cho mỗi image (paper: 700)
            M: số candidates cho re-ranking (paper: 1600)
            p: power modulation cho Chamfer similarity (paper: 0.01)
            num_local: số local features mỗi image (FIRe: 600)
        """
        self.k = k
        self.M = M
        self.p = p
        self.num_local = num_local
        
        self.db_features = None
        self.db_names = None
        self.db_local_features = None
        self.pca = None
        self.db_dim = None
        self.aligned_features = None
        self.neighbors = None
        self.neighbor_distances = None
        self.query_dim = None
        
    def load_database(self, db_bin, db_names):
        """Load database features từ .bin file"""
        print(f"\nLoading database features...")
        self.db_features, self.db_names = self._load_bin(db_bin, db_names)
        self.db_dim = self.db_features.shape[1]
        print(f"  Database: {self.db_features.shape[0]} images x {self.db_dim} dims")
        return self.db_features, self.db_names
    
    def _load_bin(self, bin_file, names_file):
        """Load features từ .bin và names từ .txt"""
        # Đọc file .bin
        with open(bin_file, 'rb') as f:
            data = f.read()
        
        # Đọc names
        with open(names_file, 'r') as f:
            names = [line.strip() for line in f.readlines()]
        
        # Tính dimension
        num_images = len(names)
        dim = len(data) // (num_images * 4)  # 4 bytes cho float32
        features = np.frombuffer(data, dtype=np.float32).reshape(num_images, dim)
        
        return features, names
    
    def load_query(self, query_bin, query_names):
        """Load query features để biết dimension"""
        print(f"\nLoading query features...")
        query_features, query_names = self._load_bin(query_bin, query_names)
        self.query_dim = query_features.shape[1]
        print(f"  Query: {query_features.shape[0]} images x {self.query_dim} dims")
        return query_features, query_names
    
    def align_with_query_dim(self, query_dim):
        """Align database to query dimension using PCA"""
        if self.db_dim != query_dim:
            print(f"\n  Aligning database from {self.db_dim} to {query_dim} dims using PCA...")
            self.pca = PCA(n_components=query_dim)
            self.aligned_features = self.pca.fit_transform(self.db_features)
            print(f"  Database aligned to {self.aligned_features.shape}")
            self.db_dim = query_dim
        else:
            self.aligned_features = self.db_features.copy()
        return self.aligned_features
    
    def chamfer_similarity(self, query_feat, db_feat):
        """
        Asymmetric Chamfer similarity với power modulation p=0.01
        """
        distances = cdist(query_feat, db_feat, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        avg_distance = np.mean(min_distances)
        similarity = 1.0 / (avg_distance + 1e-8)
        if self.p != 1.0:
            similarity = np.sign(similarity) * np.abs(similarity) ** self.p
        return similarity
    
    def precompute_neighbors(self):
        """
        Pre-compute top-k neighbors cho mỗi database image
        Sử dụng Chamfer similarity asymmetric
        """
        n_images = len(self.db_names)
        total_features = self.aligned_features.shape[0]
        
        print(f"\n{'='*60}")
        print(f"PRE-COMPUTING NEIGHBORS")
        print(f"{'='*60}")
        print(f"  Database images: {n_images}")
        print(f"  Total features: {total_features}")
        print(f"  Expected features: {n_images * self.num_local}")
        print(f"  K (neighbors): {self.k}")
        print(f"  Power p: {self.p}")
        print("-"*60)
        
        start_time = time.time()
        
        # Kiểm tra và điều chỉnh num_local nếu cần
        if total_features != n_images * self.num_local:
            print(f"  WARNING: Feature count mismatch!")
            print(f"  Adjusting num_local to {total_features // n_images}")
            self.num_local = total_features // n_images
        
        # Chuyển aligned_features thành list local features
        print(f"\n  Reshaping to local features...")
        self.db_local_features = []
        
        for i in range(n_images):
            start = i * self.num_local
            end = start + self.num_local
            if end <= total_features:
                local_feat = self.aligned_features[start:end]
            else:
                local_feat = self.aligned_features[start:]
                if len(local_feat) < self.num_local:
                    pad = np.zeros((self.num_local - len(local_feat), self.aligned_features.shape[1]))
                    local_feat = np.vstack([local_feat, pad])
            self.db_local_features.append(local_feat)
        
        print(f"  Created {len(self.db_local_features)} images")
        print(f"  Each image: {self.db_local_features[0].shape}")
        
        # Compute similarity matrix (batched để tránh memory overflow)
        print(f"\n  Computing Chamfer similarity matrix...")
        print(f"  This will take ~{n_images * n_images / 1e6:.1f} million comparisons")
        
        similarity_matrix = np.zeros((n_images, n_images))
        
        # Tính toán theo batch để tiết kiệm memory
        batch_size = 100
        for i in range(0, n_images, batch_size):
            end_i = min(i + batch_size, n_images)
            for j in range(0, n_images, batch_size):
                end_j = min(j + batch_size, n_images)
                
                for ii in range(i, end_i):
                    for jj in range(j, end_j):
                        if ii == jj:
                            similarity_matrix[ii, jj] = 0
                        else:
                            similarity_matrix[ii, jj] = self.chamfer_similarity(
                                self.db_local_features[ii], 
                                self.db_local_features[jj]
                            )
            
            print(f"    Progress: {end_i}/{n_images} rows")
        
        # Tìm top-k neighbors cho mỗi image
        print(f"\n  Finding top-{self.k} neighbors...")
        self.neighbors = []
        self.neighbor_distances = []
        
        for i in range(n_images):
            sim = similarity_matrix[i].copy()
            sim[i] = -np.inf
            
            # Lấy top-k indices
            top_indices = np.argsort(sim)[-self.k:][::-1]
            top_scores = sim[top_indices]
            
            self.neighbors.append(top_indices)
            self.neighbor_distances.append(top_scores)
            
            if (i+1) % 500 == 0:
                print(f"    Processed {i+1}/{n_images} images...")
        
        elapsed = time.time() - start_time
        print(f"\n  ✓ Pre-computed neighbors in {elapsed:.2f}s")
        print(f"  ✓ Each image has {self.k} neighbors")
        
        return self.neighbors, self.neighbor_distances
    
    def save_index(self, output_dir='.'):
        """Save index với đầy đủ local features"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n{'='*60}")
        print(f"SAVING INDEX")
        print(f"{'='*60}")
        print(f"  Output: {output_dir}")
        print("-"*60)
        
        # 1. Save database names
        names_file = os.path.join(output_dir, 'db_names.txt')
        with open(names_file, 'w') as f:
            for name in self.db_names:
                f.write(f"{name}\n")
        print(f"  ✓ Saved: {names_file} ({len(self.db_names)} images)")
        
        # 2. Save local features (cho CANN search)
        local_file = os.path.join(output_dir, 'db_local_features.npz')
        np.savez_compressed(local_file, features=self.db_local_features)
        print(f"  ✓ Saved: {local_file}")
        print(f"    - {len(self.db_local_features)} images")
        print(f"    - Each: {self.db_local_features[0].shape}")
        
        # 3. Save aligned features (cho re-ranking)
        features_file = os.path.join(output_dir, 'db_features_aligned.bin')
        with open(features_file, 'wb') as f:
            f.write(self.aligned_features.astype(np.float32).tobytes())
        print(f"  ✓ Saved: {features_file}")
        print(f"    - Shape: {self.aligned_features.shape}")
        
        # 4. Save precomputed neighbors (CANN)
        neighbors_file = os.path.join(output_dir, 'neighbors.pkl')
        with open(neighbors_file, 'wb') as f:
            pickle.dump({
                'neighbors': self.neighbors,
                'distances': self.neighbor_distances,
                'k': self.k,
                'M': self.M,
                'p': self.p,
                'similarity': 'chamfer_asymmetric',
                'num_local': self.num_local,
                'n_images': len(self.db_names)
            }, f)
        print(f"  ✓ Saved: {neighbors_file}")
        print(f"    - {len(self.neighbors)} images")
        print(f"    - Each: {self.k} neighbors")
        
        # 5. Save PCA if exists
        if self.pca is not None:
            pca_file = os.path.join(output_dir, 'pca.pkl')
            with open(pca_file, 'wb') as f:
                pickle.dump(self.pca, f)
            print(f"  ✓ Saved: {pca_file}")
        
        # 6. Save query dimension
        dim_file = os.path.join(output_dir, 'query_dim.txt')
        with open(dim_file, 'w') as f:
            f.write(f"{self.query_dim}\n")
        print(f"  ✓ Saved: {dim_file}")
        
        # 7. Save index info
        info_file = os.path.join(output_dir, 'index_info.txt')
        with open(info_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("L2G INDEX INFORMATION\n")
            f.write("="*60 + "\n")
            f.write(f"Number of images: {len(self.db_names)}\n")
            f.write(f"Feature dimension (original): {self.db_dim}\n")
            f.write(f"Query dimension: {self.query_dim}\n")
            f.write(f"Number of local features per image: {self.num_local}\n")
            f.write(f"Local feature shape: {self.db_local_features[0].shape}\n")
            f.write(f"k (neighbors per image): {self.k}\n")
            f.write(f"M (candidates for re-ranking): {self.M}\n")
            f.write(f"p (power modulation): {self.p}\n")
            f.write(f"Similarity: Chamfer (asymmetric)\n")
            f.write(f"PCA aligned: {self.pca is not None}\n")
            f.write(f"\nTotal index size: ~{self._get_index_size(output_dir):.2f} MB\n")
        print(f"  ✓ Saved: {info_file}")
        
        # 8. Calculate total size
        total_size = self._get_index_size(output_dir)
        print(f"\n  Total index size: {total_size:.2f} MB")
        print(f"\n  ✓ Index saved successfully!")
    
    def _get_index_size(self, output_dir):
        """Tính tổng kích thước index"""
        total_size = 0
        for file in os.listdir(output_dir):
            filepath = os.path.join(output_dir, file)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # MB
    
    def process(self, db_bin, db_names, query_bin, query_names, output_dir):
        """Full pipeline: load -> align -> precompute -> save"""
        print("\n" + "="*80)
        print("L2G INDEXING PIPELINE")
        print("="*80)
        
        # 1. Load database
        self.load_database(db_bin, db_names)
        
        # 2. Load query để biết dimension
        _, _ = self.load_query(query_bin, query_names)
        
        # 3. Align database to query dimension
        self.align_with_query_dim(self.query_dim)
        
        # 4. Precompute neighbors (700 neighbors per image)
        self.precompute_neighbors()
        
        # 5. Save index
        self.save_index(output_dir)
        
        print("\n" + "="*80)
        print("✓ INDEXING COMPLETE!")
        print("="*80)


def main():
    print("="*80)
    print("L2G INDEXER - BUILD INDEX FOR CANN SEARCH")
    print("Paper: Global-to-Local or Local-to-Global?")
    print("="*80)
    
    BASE_DIR = "." 
    
    datasets = [
        {
            'name': 'ROxford5K',
            'db_bin': os.path.join(BASE_DIR, 'roxford5k_database_features.bin'),
            'db_names': os.path.join(BASE_DIR, 'roxford5k_database_names.txt'),
            'query_bin': os.path.join(BASE_DIR, 'roxford5k_query_features.bin'),
            'query_names': os.path.join(BASE_DIR, 'roxford5k_query_names.txt'),
            'output_dir': 'l2g_index_roxford5k'
        },
        {
            'name': 'RParis6K',
            'db_bin': os.path.join(BASE_DIR, 'rparis6k_database_features.bin'),
            'db_names': os.path.join(BASE_DIR, 'rparis6k_database_names.txt'),
            'query_bin': os.path.join(BASE_DIR, 'rparis6k_query_features.bin'),
            'query_names': os.path.join(BASE_DIR, 'rparis6k_query_names.txt'),
            'output_dir': 'l2g_index_rparis6k'
        }
    ]
    
    for ds in datasets:
        print(f"\n{'='*80}")
        print(f"INDEXING {ds['name']}")
        print(f"{'='*80}")
        
        # Check all files exist
        all_exist = True
        for key in ['db_bin', 'db_names', 'query_bin', 'query_names']:
            if not os.path.exists(ds[key]):
                print(f"  ✗ File not found: {ds[key]}")
                all_exist = False
        
        if not all_exist:
            print(f"\n  Skipping {ds['name']}...")
            print("  Please run feature extraction and conversion first!")
            continue
        
        # Create indexer with paper parameters
        indexer = L2GIndexer(
            k=700,      # top-k neighbors (paper)
            M=1600,     # candidates for re-ranking (paper)
            p=0.01,     # power modulation (paper)
            num_local=600  # FIRe features per image
        )
        
        # Process
        indexer.process(
            db_bin=ds['db_bin'],
            db_names=ds['db_names'],
            query_bin=ds['query_bin'],
            query_names=ds['query_names'],
            output_dir=ds['output_dir']
        )
    
    print("\n" + "="*80)
    print(" INDEXES CREATED !")
    print("  - db_names.txt            : Database image names")
    print("  - db_local_features.npz   : Local features for CANN search")
    print("  - db_features_aligned.bin : Aligned features for re-ranking")
    print("  - neighbors.pkl           : Pre-computed neighbors (700 per image)")
    print("  - pca.pkl                 : PCA model (if used)")
    print("  - index_info.txt          : Index information")
    print("  - query_dim.txt           : Query feature dimension")

if __name__ == "__main__":
    main()