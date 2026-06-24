#!/usr/bin/env python3
"""
Local-to-Global (L2G) Image Retrieval System
Based on paper: "Global-to-Local or Local-to-Global? Enhancing Image Retrieval with Efficient Local Search and Effective Global Re-ranking"
"""

import numpy as np
import os
import time
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
import heapq

class L2GRetrieval:
    def __init__(self, k=700, M=1600, w=0.19, p=0.01, eps=0.1):
        """
        Initialize L2G Retrieval System
        
        Args:
            k: Number of top-ranked images for MDS embedding (default: 700)
            M: Number of candidates for re-ranking (default: 1600)
            w: Weight for SuperGlobal features (default: 0.19)
            p: Power for Chamfer similarity (default: 0.01)
            eps: Convergence threshold for MDS (default: 0.1)
        """
        self.k = k
        self.M = M
        self.w = w
        self.p = p
        self.eps = eps
        
        self.db_features = None
        self.db_names = None
        self.query_features = None
        self.query_names = None
        
    def load_features(self, db_bin, db_names, query_bin, query_names):
        """Load database and query features"""
        print("Loading database features...")
        self.db_features, self.db_names = self._load_bin(db_bin, db_names)
        print(f"  Database: {self.db_features.shape[0]} images x {self.db_features.shape[1]} dims")
        
        print("Loading query features...")
        self.query_features, self.query_names = self._load_bin(query_bin, query_names)
        print(f"  Queries: {self.query_features.shape[0]} images x {self.query_features.shape[1]} dims")
    
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
    
    def compute_chamfer_similarity(self, query_features, db_features):
        """
        Compute Chamfer similarity (asymmetric)
        As used in CANN paper for local feature retrieval
        """
        # Normalize features
        query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        db_norm = db_features / (np.linalg.norm(db_features, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity = np.dot(query_norm, db_norm.T)
        
        # Apply power modulation (p parameter from paper)
        if self.p != 1.0:
            similarity = np.sign(similarity) * np.abs(similarity) ** self.p
        
        return similarity
    
    def get_top_k_initial(self, query_idx):
        """
        Get initial top-k results using local features (CANN-like)
        """
        query_feat = self.query_features[query_idx:query_idx+1]
        
        # Compute Chamfer similarity
        similarities = self.compute_chamfer_similarity(query_feat, self.db_features)[0]
        
        # Get top M candidates
        top_indices = np.argsort(similarities)[-self.M:][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def mds_embedding(self, query_idx, top_indices):
        """
        Compute MDS embedding for query + top-k images
        As described in Section 3.3 of the paper
        """
        # Get features for query and top-k images
        k = min(self.k, len(top_indices))
        selected_indices = top_indices[:k]
        
        # Get features
        all_features = np.vstack([
            self.query_features[query_idx:query_idx+1],
            self.db_features[selected_indices]
        ])
        
        # Compute pairwise Chamfer similarity
        n = len(all_features)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_chamfer_similarity(
                        all_features[i:i+1], 
                        all_features[j:j+1]
                    )[0][0]
                    similarity_matrix[i, j] = sim
        
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.maximum(distance_matrix, 0)  # Ensure non-negative
        
        # Apply MDS using SMACOF algorithm (as mentioned in paper)
        print(f"    Computing MDS for query {query_idx} with {k} neighbors...")
        mds = MDS(
            n_components=min(128, n-1),
            metric=True,
            n_init=1,
            max_iter=300,
            eps=self.eps,
            random_state=42,
            dissimilarity='precomputed'
        )
        
        embedding = mds.fit_transform(distance_matrix)
        
        return embedding
    
    def re_rank_with_global(self, query_idx, top_indices, mds_embedding):
        """
        Re-rank using global features with MDS embedding
        As described in Section 3.1 of the paper
        """
        k = min(self.k, len(top_indices))
        selected_indices = top_indices[:k]
        
        # Get SuperGlobal-like features (using normalized features)
        sg_features = self.db_features[selected_indices]
        sg_query = self.query_features[query_idx:query_idx+1]
        
        # Normalize for cosine similarity
        sg_query_norm = normalize(sg_query, norm='l2')[0]
        sg_db_norm = normalize(sg_features, norm='l2')
        
        # Compute SuperGlobal similarity
        sg_similarities = np.dot(sg_db_norm, sg_query_norm)
        
        # Get MDS embedding for database images (exclude query)
        mds_db_embedding = mds_embedding[1:]  # Skip query
        mds_query_embedding = mds_embedding[0:1]
        
        # Normalize MDS embeddings
        mds_db_norm = normalize(mds_db_embedding, norm='l2')
        mds_query_norm = normalize(mds_query_embedding, norm='l2')[0]
        
        # Compute MDS similarities
        mds_similarities = np.dot(mds_db_norm, mds_query_norm)
        
        # Combine scores with weight w (from paper)
        # w = 0.19 from optimal parameters
        combined_scores = (1 - self.w) * mds_similarities + self.w * sg_similarities[:k]
        
        # Re-rank indices based on combined scores
        re_ranked_indices = np.argsort(combined_scores)[::-1]
        re_ranked_scores = combined_scores[re_ranked_indices]
        
        return selected_indices[re_ranked_indices], re_ranked_scores
    
    def search(self, query_idx, verbose=True):
        """
        Full L2G search pipeline for a single query
        """
        start_time = time.time()
        
        # Step 1: Initial local feature retrieval (CANN)
        if verbose:
            print(f"\nProcessing query {query_idx}: {self.query_names[query_idx]}")
        
        top_indices, top_scores = self.get_top_k_initial(query_idx)
        
        # Step 2: MDS embedding (Section 3.3)
        mds_embedding = self.mds_embedding(query_idx, top_indices)
        
        # Step 3: Re-ranking with global features (Section 3.1)
        re_ranked_indices, re_ranked_scores = self.re_rank_with_global(
            query_idx, top_indices, mds_embedding
        )
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"    Query time: {elapsed:.3f}s")
        
        return re_ranked_indices, re_ranked_scores
    
    def search_all(self, top_k=700):
        """
        Search for all queries and get top_k results
        """
        results = {}
        total_time = 0
        
        print(f"\n{'='*80}")
        print(f"L2G SEARCH - Getting top {top_k} results for each query")
        print(f"Parameters: k={self.k}, M={self.M}, w={self.w}, p={self.p}")
        print(f"{'='*80}")
        
        for i in range(len(self.query_features)):
            start_time = time.time()
            indices, scores = self.search(i, verbose=False)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            query_name = self.query_names[i]
            results[query_name] = []
            
            for j, (idx, score) in enumerate(zip(indices[:top_k], scores[:top_k])):
                db_name = self.db_names[idx]
                results[query_name].append((j+1, db_name, float(score)))
            
            print(f"  Query {i+1}/{len(self.query_features)}: {query_name} - Done in {elapsed:.3f}s")
        
        print(f"\nTotal time: {total_time:.3f}s, Average: {total_time/len(self.query_features):.3f}s/query")
        
        return results
    
    def save_results(self, results, output_file, dataset_name, top_k=700):
        """Save results to file"""
        with open(output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write(f"{dataset_name} - L2G TOP {top_k} NEAREST NEIGHBORS\n")
            f.write(f"Parameters: k={self.k}, M={self.M}, w={self.w}, p={self.p}\n")
            f.write("="*100 + "\n\n")
            
            for query_name, neighbors in results.items():
                f.write(f"Query: {query_name}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Rank':<6} {'Database Image':<35} {'Score':<15}\n")
                f.write("-"*80 + "\n")
                
                for rank, db_name, score in neighbors:
                    f.write(f"{rank:<6} {db_name:<35} {score:.8f}\n")
                
                f.write("\n")
        
        print(f"\n✅ Results saved to {output_file}")


def main():
    # Initialize L2G system with optimal parameters from paper
    # Paper: k=700, M=1600, w=0.19, p=0.01, eps=0.1
    l2g = L2GRetrieval(k=700, M=1600, w=0.19, p=0.01, eps=0.1)
    
    # Load features
    l2g.load_features(
        db_bin='roxford5k_database_features.bin',
        db_names='roxford5k_database_names.txt',
        query_bin='roxford5k_query_features.bin',
        query_names='roxford5k_query_names.txt'
    )
    
    # Search for all queries
    results = l2g.search_all(top_k=700)
    
    # Save results
    l2g.save_results(results, 'roxford5k_l2g_results.txt', 'ROxford5K', top_k=700)
    
    # Print sample results
    print(f"\n{'='*80}")
    print("SAMPLE RESULTS (First query, top 10)")
    print(f"{'='*80}")
    first_query = list(results.keys())[0]
    print(f"\nQuery: {first_query}")
    print("-"*60)
    for rank, db_name, score in results[first_query][:10]:
        print(f"  {rank:4d}. {db_name:<35} {score:.8f}")


if __name__ == "__main__":
    main()
