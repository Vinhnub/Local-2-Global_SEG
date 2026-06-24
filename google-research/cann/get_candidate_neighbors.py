#!/usr/bin/env python3

# # With command line arguments
# python3 get_candidate_neighbors.py roxford5k all_souls_000000 700
# python3 get_candidate_neighbors.py rparis6k paris_defense_000605 700

# # With partial arguments (will ask for missing ones)
# python3 get_candidate_neighbors.py roxford5k all_souls_000000
# python3 get_candidate_neighbors.py roxford5k

import os
import pickle
import sys

def get_neighbors(dataset='roxford5k', image_name='all_souls_000000', top_k=700):
    index_dir = f'l2g_index_{dataset}'
    
    if not os.path.exists(index_dir):
        print(f"Index not found: {index_dir}")
        return
    
    with open(os.path.join(index_dir, 'db_names.txt'), 'r') as f:
        db_names = [line.strip() for line in f.readlines()]
    
    try:
        query_idx = db_names.index(image_name)
        print(f"Found '{image_name}' at index {query_idx}")
    except ValueError:
        print(f"Image '{image_name}' not found in database!")
        print(f"Available images: {db_names[:5]}...")
        return
    
    with open(os.path.join(index_dir, 'neighbors.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    neighbors = data['neighbors'][query_idx]
    distances = data['distances'][query_idx]
    k = data.get('k', 700)
    
    print(f"\nTop {min(top_k, len(neighbors))} neighbors for: {image_name}")
    print("="*80)
    print(f"{'Rank':<6} {'Database Image':<50} {'Score':<15}")
    print("-"*80)
    
    for rank, (idx, score) in enumerate(zip(neighbors[:top_k], distances[:top_k]), 1):
        db_name = db_names[idx]
        print(f"{rank:<6} {db_name:<50} {score:.8f}")
    
    output_file = f"{image_name}_neighbors.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Top {min(top_k, len(neighbors))} neighbors for: {image_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Rank':<6} {'Database Image':<50} {'Score':<15}\n")
        f.write("-"*80 + "\n")
        for rank, (idx, score) in enumerate(zip(neighbors[:top_k], distances[:top_k]), 1):
            db_name = db_names[idx]
            f.write(f"{rank:<6} {db_name:<50} {score:.8f}\n")
    
    print(f"\nSaved to: {output_file}")
    return neighbors[:top_k], distances[:top_k]

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        dataset = sys.argv[1]
    else:
        dataset = input("Dataset (roxford5k/rparis6k) [roxford5k]: ").strip() or 'roxford5k'
    
    if len(sys.argv) >= 3:
        image_name = sys.argv[2]
    else:
        image_name = input("Image name: ").strip()
    
    if len(sys.argv) >= 4:
        top_k = int(sys.argv[3])
    else:
        top_k = int(input("Number of neighbors [700]: ").strip() or '700')
    
    get_neighbors(dataset, image_name, top_k)