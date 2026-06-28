#!/usr/bin/env python3
"""
Convert FIRe features (.npy) sang định dạng .desc cho CANN tool
"""
import numpy as np
import os
from pathlib import Path
import struct


def convert_npy_to_cann_desc(input_folder, output_folder, max_images=None):
    """
    Chuyển đổi tất cả .npy trong folder sang .desc cho CANN
    
    Args:
        input_folder: Thư mục chứa các file .npy (mỗi file là local features (600, D))
        output_folder: Thư mục output cho các file .desc
        max_images: Giới hạn số ảnh (None = tất cả)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả file .npy
    npy_files = sorted([f for f in input_path.glob('*.npy')])
    
    if max_images:
        npy_files = npy_files[:max_images]
    
    print(f"Found {len(npy_files)} .npy files in {input_folder}")
    print(f"Output: {output_folder}")
    
    desc_files = []
    
    for i, npy_file in enumerate(npy_files):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(npy_files)}: {npy_file.name}")
        
        try:
            # Load features
            features = np.load(npy_file)
            # Fix shape mismatch if features are (128, 600) instead of (600, 128)
            if features.shape[0] < features.shape[1]:
                features = features.T
            # features shape: (num_local, D) với num_local=600
            
            num_local = features.shape[0]
            dim = features.shape[1]
            
            # Tạo file .desc
            desc_file = output_path / npy_file.with_suffix('.desc').name
            
            # Ghi theo định dạng CANN:
            # Header: num_local (int32), dim (int32)
            # Body: features (float32)
            with open(desc_file, 'wb') as f:
                # Write header
                f.write(struct.pack('ii', num_local, dim))
                # Write features
                f.write(features.astype(np.float32).tobytes())
            
            desc_files.append(desc_file)
            
        except Exception as e:
            print(f"  Error processing {npy_file.name}: {e}")
    
    print(f"\nConverted {len(desc_files)} files to .desc")
    
    # Tạo file list cho CANN
    list_file = output_path / 'descriptor_list.txt'
    with open(list_file, 'w') as f:
        for desc_file in desc_files:
            f.write(f"{desc_file}\n")
    
    print(f"Created descriptor list file: {list_file}")
    
    return desc_files, list_file


def main():
    # Resolve project root dynamically
    script_dir = Path(__file__).parent.resolve()
    BASE_DIR = script_dir.parent.parent.resolve()
    
    FEATURE_DIR = BASE_DIR / "output" / "stage1" / "features"
    OUTPUT_DIR = BASE_DIR / "output" / "stage1.2" / "cann_data"
    
    datasets = [
        {
            'name': 'roxford5k',
            'input': FEATURE_DIR / 'roxford5k' / 'database',
            'output': OUTPUT_DIR / 'roxford5k' / 'database'
        },
        {
            'name': 'roxford5k',
            'input': FEATURE_DIR / 'roxford5k' / 'query',
            'output': OUTPUT_DIR / 'roxford5k' / 'query'
        },
        {
            'name': 'rparis6k',
            'input': FEATURE_DIR / 'rparis6k' / 'database',
            'output': OUTPUT_DIR / 'rparis6k' / 'database'
        },
        {
            'name': 'rparis6k',
            'input': FEATURE_DIR / 'rparis6k' / 'query',
            'output': OUTPUT_DIR / 'rparis6k' / 'query'
        }
    ]
    
    print("="*60)
    print("CONVERT FIRe FEATURES TO CANN FORMAT (.desc)")
    print("="*60)
    
    for ds in datasets:
        if not ds['input'].exists():
            print(f"\n⚠ Skipping {ds['name']} - input not found: {ds['input']}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Dataset: {ds['name']}")
        print(f"Type: {ds['input'].parent.name}")
        print(f"{'='*60}")
        
        convert_npy_to_cann_desc(ds['input'], ds['output'])
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print("\nThe .desc files have been created in:")
    print("  cann_data/roxford5k/database/")
    print("  cann_data/roxford5k/query/")
    print("  cann_data/rparis6k/database/")
    print("  cann_data/rparis6k/query/")


if __name__ == "__main__":
    main()