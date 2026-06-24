#!/usr/bin/env python3
import os
import pickle
import numpy as np
from collections import Counter

def check_index(dataset='roxford5k', expected_k=700):
    index_dir = f'l2g_index_{dataset}'
    
    print(f"\n{'='*70}")
    print(f"CHECKING INDEX: {dataset.upper()}")
    print(f"{'='*70}")
    
    if not os.path.exists(index_dir):
        print(f"❌ Index not found: {index_dir}")
        return False
    
    all_ok = True
    features = None
    names = []
    
    # 1. Check names
    try:
        names_file = os.path.join(index_dir, 'db_names.txt')
        with open(names_file, 'r') as f:
            names = [line.strip() for line in f.readlines()]
        print(f"✅ Names: {len(names)} images")
        if len(names) > 0:
            print(f"   First 5: {names[:5]}")
    except Exception as e:
        print(f"❌ Error loading names: {e}")
        return False
    
    # 2. Check features (try local first, then global)
    try:
        local_file = os.path.join(index_dir, 'db_features_local.npz')
        global_file = os.path.join(index_dir, 'db_features_aligned.bin')
        
        if os.path.exists(local_file):
            data = np.load(local_file, allow_pickle=True)
            features = data['features']
            print(f"✅ Local features: {len(features)} images")
            
            if len(features) > 0:
                print(f"   Feature shape: {features[0].shape}")
                print(f"   Feature type: Local (600 features per image)")
                
                # Check if all features have same shape
                shapes = [f.shape for f in features]
                if len(set(shapes)) == 1:
                    print(f"   ✅ All features have same shape: {shapes[0]}")
                else:
                    print(f"   ❌ Mixed shapes: {set(shapes)}")
                    all_ok = False
                    
        elif os.path.exists(global_file):
            with open(global_file, 'rb') as f:
                features_data = np.frombuffer(f.read(), dtype=np.float32)
            
            # Calculate dimension
            if len(names) > 0:
                dim = len(features_data) // len(names)
                features = features_data.reshape(len(names), dim)
                print(f"✅ Global features: {features.shape[0]} images x {features.shape[1]} dims")
                print(f"   Feature type: Global (single vector per image)")
            else:
                print(f"❌ No names found to determine feature shape")
                all_ok = False
        else:
            print(f"❌ No feature file found (neither local nor global)")
            all_ok = False
            
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        all_ok = False
    
    # 3. Check mapping between names and features
    if features is not None and len(names) == len(features):
        print(f"✅ Names and features match: {len(names)} = {len(features)}")
    elif features is not None:
        print(f"❌ Mismatch: {len(names)} names vs {len(features)} features")
        all_ok = False
    
    # 4. Check neighbors
    try:
        neighbors_file = os.path.join(index_dir, 'neighbors.pkl')
        with open(neighbors_file, 'rb') as f:
            data = pickle.load(f)
        
        neighbors = data['neighbors']
        distances = data['distances']
        k_value = data.get('k', expected_k)
        m_value = data.get('M', None)
        p_value = data.get('p', None)
        
        print(f"✅ Neighbors loaded")
        print(f"   k (neighbors per image): {k_value}")
        print(f"   M (candidates): {m_value}")
        print(f"   p (power modulation): {p_value}")
        print(f"   Similarity: {data.get('similarity', 'N/A')}")
        
        # Check neighbor count
        print(f"\n📊 Neighbor Count Check:")
        counts = [len(n) for n in neighbors]
        unique_counts = set(counts)
        
        if len(unique_counts) == 1 and counts[0] == expected_k:
            print(f"   ✅ All {len(neighbors)} images have exactly {expected_k} neighbors")
        else:
            print(f"   ❌ Images have different neighbor counts:")
            print(f"      Total images: {len(neighbors)}")
            print(f"      Expected: {expected_k}")
            print(f"      Range: min={min(counts)}, max={max(counts)}")
            print(f"      Unique counts: {unique_counts}")
            all_ok = False
            
            # Show distribution
            print(f"\n   Distribution:")
            counter = Counter(counts)
            for count, num_images in sorted(counter.items())[:10]:
                print(f"      {count} neighbors: {num_images} images")
            if len(counter) > 10:
                print(f"      ... and {len(counter) - 10} more")
        
        # Check self-neighbor
        print(f"\n📊 Self-Neighbor Check:")
        self_neighbors = sum(1 for i, n in enumerate(neighbors) if i in n)
        if self_neighbors == 0:
            print(f"   ✅ No image is neighbor to itself (good)")
        else:
            print(f"   ⚠️  {self_neighbors} images have self as neighbor")
            all_ok = False
        
        # Check duplicate neighbors
        print(f"\n📊 Duplicate Check:")
        duplicates = []
        for i, n in enumerate(neighbors):
            if len(set(n)) != len(n):
                duplicates.append(i)
        
        if duplicates:
            print(f"   ❌ {len(duplicates)} images have duplicate neighbors")
            for idx in duplicates[:5]:
                print(f"      Image {idx}: {len(set(neighbors[idx]))}/{len(neighbors[idx])} unique")
            if len(duplicates) > 5:
                print(f"      ... and {len(duplicates) - 5} more")
            all_ok = False
        else:
            print(f"   ✅ No duplicate neighbors found")
            
    except Exception as e:
        print(f"❌ Error loading neighbors: {e}")
        all_ok = False
    
    # 5. Check PCA
    try:
        pca_file = os.path.join(index_dir, 'pca.pkl')
        if os.path.exists(pca_file):
            with open(pca_file, 'rb') as f:
                pca = pickle.load(f)
            print(f"✅ PCA loaded: components={pca.components_.shape}")
        else:
            print(f"ℹ️  No PCA file found (not required)")
    except Exception as e:
        print(f"❌ Error loading PCA: {e}")
        all_ok = False
    
    # 6. Check query dimension
    try:
        dim_file = os.path.join(index_dir, 'query_dim.txt')
        if os.path.exists(dim_file):
            with open(dim_file, 'r') as f:
                query_dim = int(f.read().strip())
            print(f"✅ Query dimension: {query_dim}")
    except Exception as e:
        print(f"ℹ️  No query_dim file found (not required)")
    
    # 7. Check info file
    try:
        info_file = os.path.join(index_dir, 'index_info.txt')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = f.read()
            print(f"✅ Index info file exists")
    except Exception as e:
        print(f"ℹ️  No index_info file found")
    
    # Summary
    print(f"\n{'='*70}")
    if all_ok:
        print(f"✅ INDEX {dataset.upper()} IS VALID!")
    else:
        print(f"❌ INDEX {dataset.upper()} HAS ISSUES!")
    print(f"{'='*70}")
    
    # Quick summary
    print(f"\n📋 Summary:")
    print(f"   Images: {len(names)}")
    if features is not None:
        if isinstance(features, np.ndarray):
            print(f"   Feature shape: {features.shape}")
        elif len(features) > 0:
            print(f"   Feature shape: {features[0].shape}")
    print(f"   Neighbors per image: {k_value}")
    print(f"   Status: {'✅ OK' if all_ok else '❌ NEEDS FIX'}")
    
    return all_ok

def main():
    print("="*70)
    print("L2G INDEX VALIDATION TOOL")
    print("="*70)
    
    datasets = ['roxford5k', 'rparis6k']
    
    for dataset in datasets:
        check_index(dataset, expected_k=700)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()