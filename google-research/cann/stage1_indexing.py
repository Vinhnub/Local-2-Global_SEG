#!/usr/bin/env python3
"""
Stage 1: Database Indexing for L2G Retrieval
- Load database features
- Align dimensions using PCA
- Pre-compute top-k neighbors for each database image
- Save index files for fast querying
"""

import numpy as np
import os
import time
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from collections import defaultdict

class L2GIndexer:
    def __init__(self, k=700, M=1600):
        """
        Initialize L2G Indexer
        
        Args:
            k: Number of top neighbors to store per image (default: 700)
            M: Number of candidates for re-ranking (default: 1600)
        """
        self.k = k
        self.M = M
        self.db_features = None
        self.db_names = None
        self.pca = None
        self.db_dim = None
        self.aligned_features = None
        
        # Store pre-computed neighbors
        self.neighbors = None  # For each DB image, store top-k neighbors
        self.neighbor_distances = None
        
    def load_database(self, db_bin, db_names):
        """Load database features"""
        print("="*80)
        print("STAGE 1: DATABASE INDEXING")
        print("="*80)
        
        print("\nLoading database features...")
        self.db_features, self.db_names = self._load_bin(db_bin, db_names)
        self.db_dim = self.db_features.shape[1]
        print(f"  Database: {self.db_features.shape[0]} images x {self.db_dim} dims")
        
        return self.db_features, self.db_names
    
    def _load_bin(self, bin_file, names_file):
        """Load features from .bin file"""
        with open(bin_file, 'rb') as f:
            features = np.frombuffer(f.read(), dtype=np.float32)
        
        with open(names_file, 'r') as f:
            names = [line.strip() for line in f.readlines()]
        
        num_images = len(names)
        dim = len(features) // num_images
        features = features.reshape(num_images, dim)
        
        return features, names
    
    def align_with_query_dim(self, query_dim, fit_pca=True):
        """
        Align database to query dimension using PCA
        
        Args:
            query_dim: Target dimension (query features dimension)
            fit_pca: Whether to fit PCA (True) or use existing PCA
        """
        if self.db_dim != query_dim:
            print(f"\n  Aligning database from {self.db_dim} to {query_dim} dims using PCA...")
            
            if fit_pca:
                self.pca = PCA(n_components=query_dim)
                self.aligned_features = self.pca.fit_transform(self.db_features)
            else:
                self.aligned_features = self.pca.transform(self.db_features)
            
            print(f"  ✅ Database aligned to {self.aligned_features.shape}")
            self.db_dim = query_dim
        else:
            self.aligned_features = self.db_features.copy()
        
        return self.aligned_features
    
    def compute_similarity(self, features, db_features):
        """Compute cosine similarity"""
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        db_norm = db_features / (np.linalg.norm(db_features, axis=1, keepdims=True) + 1e-8)
        return np.dot(features_norm, db_norm.T)
    
    def precompute_neighbors(self, query_features=None, verbose=True):
        """
        Pre-compute top-k neighbors for each database image
        If query_features is provided, align to query dimension
        """
        print(f"\nPre-computing top-{self.k} neighbors for {len(self.db_names)} database images...")
        print("-"*60)
        
        start_time = time.time()
        
        # Use aligned features
        features = self.aligned_features
        
        # Compute pairwise similarities (only for DB images)
        # For efficiency, we can compute in batches
        similarities = self.compute_similarity(features, features)
        
        # For each image, get top-k neighbors (excluding itself)
        self.neighbors = []
        self.neighbor_distances = []
        
        for i in range(len(features)):
            # Get similarities for image i
            sim = similarities[i]
            
            # Set self-similarity to -inf to exclude itself
            sim[i] = -np.inf
            
            # Get top-k indices
            top_indices = np.argsort(sim)[-self.k:][::-1]
            top_scores = sim[top_indices]
            
            self.neighbors.append(top_indices)
            self.neighbor_distances.append(top_scores)
            
            if (i+1) % 500 == 0:
                print(f"  Processed {i+1}/{len(features)} images...")
        
        elapsed = time.time() - start_time
        print(f"  ✅ Pre-computed neighbors in {elapsed:.2f}s")
        print(f"  ✅ Stored {self.k} neighbors per image")
        
        return self.neighbors, self.neighbor_distances
    
    def save_index(self, output_dir='.'):
        """
        Save index files for fast querying
        Files saved:
        - db_names.txt: Database image names
        - db_features_aligned.bin: Aligned database features
        - neighbors.pkl: Pre-computed neighbors
        - pca.pkl: PCA model for transforming queries
        - index_info.txt: Index metadata
        """
        print(f"\nSaving index to {output_dir}...")
        print("-"*60)
        
        # 1. Save database names
        names_file = os.path.join(output_dir, 'db_names.txt')
        with open(names_file, 'w') as f:
            for name in self.db_names:
                f.write(f"{name}\n")
        print(f"  ✅ Saved {names_file}")
        
        # 2. Save aligned features
        features_file = os.path.join(output_dir, 'db_features_aligned.bin')
        with open(features_file, 'wb') as f:
            f.write(self.aligned_features.astype(np.float32).tobytes())
        print(f"  ✅ Saved {features_file}")
        
        # 3. Save neighbors
        neighbors_file = os.path.join(output_dir, 'neighbors.pkl')
        with open(neighbors_file, 'wb') as f:
            pickle.dump({
                'neighbors': self.neighbors,
                'distances': self.neighbor_distances,
                'k': self.k,
                'M': self.M
            }, f)
        print(f"  ✅ Saved {neighbors_file}")
        
        # 4. Save PCA model
        if self.pca is not None:
            pca_file = os.path.join(output_dir, 'pca.pkl')
            with open(pca_file, 'wb') as f:
                pickle.dump(self.pca, f)
            print(f"  ✅ Saved {pca_file}")
        
        # 5. Save index info
        info_file = os.path.join(output_dir, 'index_info.txt')
        with open(info_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("L2G INDEX INFORMATION\n")
            f.write("="*60 + "\n")
            f.write(f"Number of images: {len(self.db_names)}\n")
            f.write(f"Feature dimension: {self.db_dim}\n")
            f.write(f"k (neighbors per image): {self.k}\n")
            f.write(f"M (candidates for re-ranking): {self.M}\n")
            f.write(f"PCA aligned: {self.pca is not None}\n")
        print(f"  ✅ Saved {info_file}")
        
        print(f"\n✅ Index saved successfully to {output_dir}")
        
    def process(self, db_bin, db_names, query_dim=None, output_dir='.'):
        """
        Full indexing pipeline
        
        Args:
            db_bin: Database features file
            db_names: Database names file
            query_dim: Query feature dimension (for PCA alignment)
            output_dir: Output directory for index files
        """
        # Load database
        self.load_database(db_bin, db_names)
        
        # Align if query_dim provided
        if query_dim is not None and query_dim != self.db_dim:
            self.align_with_query_dim(query_dim, fit_pca=True)
        else:
            self.aligned_features = self.db_features.copy()
            self.db_dim = self.db_features.shape[1]
        
        # Pre-compute neighbors
        self.precompute_neighbors()
        
        # Save index
        self.save_index(output_dir)
        
        print("\n" + "="*80)
        print("✅ INDEXING COMPLETE!")
        print("="*80)
        print(f"\nIndex files saved in: {output_dir}")
        print("\nNext step: Run stage2_query.py to search")

def main():
    # Configuration
    db_bin = 'roxford5k_database_features.bin'
    db_names = 'roxford5k_database_names.txt'
    query_dim = 128  # Query feature dimension
    
    # Create indexer
    indexer = L2GIndexer(k=700, M=1600)
    
    # Process
    indexer.process(
        db_bin=db_bin,
        db_names=db_names,
        query_dim=query_dim,
        output_dir='l2g_index'
    )

if __name__ == "__main__":
    main()
