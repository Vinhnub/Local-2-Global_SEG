import numpy as np
import os
from pathlib import Path

def load_npy_features(folder_path, output_prefix, max_images=None, pool_method='mean'):
    """    
    Args:
        folder_path: Đường dẫn thư mục chứa .npy
        output_prefix: Prefix cho file output
        max_images: Giới hạn số ảnh (None = tất cả)
        pool_method: 'mean' hoặc 'max' hoặc 'none' (giữ nguyên local features)
    """
    all_features = []
    all_names = []
    
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    
    if max_images:
        npy_files = npy_files[:max_images]
    
    print(f"Tìm thấy {len(npy_files)} file .npy trong {folder_path}")
    
    for i, filename in enumerate(npy_files):
        if (i+1) % 100 == 0:
            print(f"  Đang xử lý file {i+1}/{len(npy_files)}: {filename}")
        
        filepath = os.path.join(folder_path, filename)
        try:
            # Load .npy
            features = np.load(filepath)
            
            # Pooling theo phương pháp
            if pool_method == 'mean':
                # Mean pooling: (600, D) -> (D,)
                if features.ndim == 2:
                    pooled = features.mean(axis=0)
                elif features.ndim == 1:
                    pooled = features
                else:
                    print(f"   Shape không mong đợi {features.shape} cho {filename}")
                    continue
            elif pool_method == 'max':
                # Max pooling: (600, D) -> (D,)
                if features.ndim == 2:
                    pooled = features.max(axis=0)
                elif features.ndim == 1:
                    pooled = features
                else:
                    print(f"   Shape không mong đợi {features.shape} cho {filename}")
                    continue
            else:
                # Giữ nguyên local features
                pooled = features
            
            all_features.append(pooled)
            # Lấy tên file không có extension
            name = filename.replace('.npy', '')
            all_names.append(name)
            
        except Exception as e:
            print(f"  Lỗi đọc file {filename}: {e}")
    
    if not all_features:
        print(f"Không có feature nào được đọc từ {folder_path}")
        return None, None
    
    if pool_method == 'none':
        print(f"Giữ nguyên local features: {len(all_features)} images, mỗi image có {all_features[0].shape}")
        # Lưu từng file riêng hoặc lưu thành 1 file lớn
        output_bin = f"{output_prefix}_features_list.bin"
        # Cách đơn giản: lưu thành dict
        np.savez_compressed(output_bin, features=all_features, names=all_names)
        output_names = f"{output_prefix}_names.txt"
        with open(output_names, 'w') as f:
            for name in all_names:
                f.write(f"{name}\n")
        print(f"Đã tạo {output_bin} (nén)")
        return output_bin, output_names
    else:
        # Nếu đã pooling, lưu thành array
        features_array = np.array(all_features, dtype=np.float32)
        
        # Ghi file .bin
        output_bin = f"{output_prefix}_features.bin"
        with open(output_bin, 'wb') as f:
            f.write(features_array.tobytes())
        
        # Ghi file danh sách tên
        output_names = f"{output_prefix}_names.txt"
        with open(output_names, 'w') as f:
            for name in all_names:
                f.write(f"{name}\n")
        
        print(f"\nĐã tạo {output_bin}: {features_array.shape[0]} ảnh x {features_array.shape[1]} chiều")
        print(f"Đã tạo {output_names}")
        
        return output_bin, output_names

BASE_DIR = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/data/feature/fire"

# ROxford5K
print("="*60)
print("ROXFORD5K")
print("="*60)

database_path = os.path.join(BASE_DIR, "roxford5k/database/")
query_path = os.path.join(BASE_DIR, "roxford5k/query/")

if not os.path.exists(database_path):
    print(f" Database path không tồn tại: {database_path}")
    print("Kiểm tra lại đường dẫn!")
else:
    print("\n=== Convert DATABASE ===")
    db_bin, db_names = load_npy_features(database_path, "roxford5k_database", pool_method='mean')

if not os.path.exists(query_path):
    print(f" Query path không tồn tại: {query_path}")
    print("Kiểm tra lại đường dẫn!")
else:
    print("\n=== Convert QUERY ===")
    q_bin, q_names = load_npy_features(query_path, "roxford5k_query", pool_method='mean')

# RParis6K
print("\n" + "="*60)
print("RPARIS6K")
print("="*60)

database_path = os.path.join(BASE_DIR, "rparis6k/database/")
query_path = os.path.join(BASE_DIR, "rparis6k/query/")

if not os.path.exists(database_path):
    print(f" Database path không tồn tại: {database_path}")
    print("Kiểm tra lại đường dẫn!")
else:
    print("\n=== Convert DATABASE ===")
    db_bin, db_names = load_npy_features(database_path, "rparis6k_database", pool_method='mean')

if not os.path.exists(query_path):
    print(f" Query path không tồn tại: {query_path}")
    print("Kiểm tra lại đường dẫn!")
else:
    print("\n=== Convert QUERY ===")
    q_bin, q_names = load_npy_features(query_path, "rparis6k_query", pool_method='mean')

print("\n" + "="*60)
print("Done!")
print("="*60)