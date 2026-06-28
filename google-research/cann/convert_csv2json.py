#!/usr/bin/env python3
"""
Convert CANN CSV files to JSON
Mỗi CSV được chuyển thành JSON cùng thư mục với file CSV
"""

import pandas as pd
import json
import time
from pathlib import Path


def csv_to_json(csv_file, top_k=1600):
    """
    Convert CANN CSV to JSON
    
    Args:
        csv_file: Đường dẫn file CSV
        top_k: Số neighbors (700 cho self-query, 1600 cho query)
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"  ✗ File not found: {csv_path}")
        return None
    
    print(f"\n  Converting: {csv_path.name}")
    
    # Đọc CSV
    df = pd.read_csv(csv_path, header=None, names=['query', 'database', 'score'])
    
    # Parse results
    results = {}
    
    for query in df['query'].unique():
        query_results = df[df['query'] == query].sort_values('score')
        query_name = Path(query).stem
        
        results[query_name] = []
        for rank, (_, row) in enumerate(query_results.iterrows(), 1):
            db_name = Path(row['database']).stem
            results[query_name].append({
                'rank': rank,
                'db_name': db_name,
                'score': float(row['score'])  # CANN score (matching ratio)
            })
    
    # Build JSON
    output_data = {
        'metadata': {
            'dataset': csv_path.stem,
            'method': 'CANN (Binary)',
            'top_k': top_k,
            'similarity': 'Matching Ratio (HIGHER = BETTER)',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_queries': len(results),
            'total_results': sum(len(r) for r in results.values())
        },
        'queries': []
    }
    
    for query_name, query_results in results.items():
        output_data['queries'].append({
            'query_name': query_name,
            'results': query_results
        })
    
    # Save JSON cùng thư mục với CSV
    json_file = csv_path.with_suffix('.json')
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved: {json_file.name}")
    print(f"    Queries: {len(results)}")
    print(f"    Total results: {sum(len(r) for r in results.values())}")
    print(f"    Size: {json_file.stat().st_size / 1024:.1f} KB")
    
    return output_data


def main():
    print("="*60)
    print("CONVERT CANN CSV TO JSON")
    print("Mỗi JSON được tạo cùng thư mục với CSV")
    print("="*60)
    
    # Đường dẫn các file CSV
    BASE_DIR = Path("/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG")
    
    csv_files = [
        # Query results (70 queries × 1600 neighbors)
        BASE_DIR / "query_results" / "roxford5k_query_results.csv",
        BASE_DIR / "query_results" / "rparis6k_query_results.csv",
        
        # Self-query results (database × 700 neighbors)
        BASE_DIR / "index_output" / "roxford5k_self_pairs.csv",
        BASE_DIR / "index_output" / "rparis6k_self_pairs.csv",
    ]
    
    for csv_file in csv_files:
        # Xác định top_k dựa trên tên file
        if 'query' in csv_file.name:
            top_k = 1600
        else:
            top_k = 700
        
        csv_to_json(csv_file, top_k=top_k)
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print("\nJSON files created:")
    for csv_file in csv_files:
        if csv_file.exists():
            print(f"  - {csv_file.with_suffix('.json')}")


if __name__ == "__main__":
    main()