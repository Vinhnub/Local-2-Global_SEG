import os
import shutil
import sys
import pickle

PKL_FILE = "/mnt/d/SEGMain/SEGCode/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl"
IMAGE_SOURCE_DIR = "/mnt/d/SEGMain/SEGCode/revisitop/data/datasets/rparis6k/jpg"
OUTPUT_DIR = "/mnt/d/SEGMain/SEGCode/data_after_split/rparis6k"


def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def find_image_in_jpg(jpg_dir, image_name):
    """Tìm ảnh trong thư mục jpg"""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in extensions:
        # Tên gốc
        img_path = os.path.join(jpg_dir, image_name + ext)
        if os.path.exists(img_path):
            return img_path
        
        # Với tiền tố paris_
        img_path = os.path.join(jpg_dir, f"paris_{image_name}{ext}")
        if os.path.exists(img_path):
            return img_path
        
        # Với tiền tố oxford_
        img_path = os.path.join(jpg_dir, f"oxford_{image_name}{ext}")
        if os.path.exists(img_path):
            return img_path
    
    return None

def split_query_db_images(pkl_file, source_dir, output_dir):
    
    if not os.path.exists(pkl_file):
        print(f"PKL file not found: {pkl_file}")
        return False
    
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return False
    
    # Load pkl
    print("Loading PKL file...")
    gnd = load_pkl(pkl_file)
    
    dataset_name = os.path.basename(pkl_file).replace('gnd_', '').replace('.pkl', '')
    print(f"Dataset: {dataset_name}")
    
    # Tạo thư mục output
    query_dir = os.path.join(output_dir, 'query')
    db_dir = os.path.join(output_dir, 'database')
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    
    print(f"Source JPG: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"{len(gnd['qimlist'])} queries, {len(gnd['imlist'])} database images")
    print("=" * 60)
    
    # ===== XỬ LÝ QUERY IMAGES =====
    print(f"\nProcessing {len(gnd['qimlist'])} QUERY images...")
    query_found = 0
    
    for i, qim_name in enumerate(gnd['qimlist']):
        img_path = find_image_in_jpg(source_dir, qim_name)
        
        if img_path:
            ext = os.path.splitext(img_path)[1]
            dest_path = os.path.join(query_dir, f"query_{i:04d}{ext}")
            
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(query_dir, f"query_{i:04d}_{counter}{ext}")
                counter += 1
            
            shutil.copy2(img_path, dest_path)
            query_found += 1
            if i < 5:
                print(f" Query {i:04d}: {os.path.basename(img_path)}")
        else:
            print(f" Query {i:04d}: {qim_name} NOT FOUND")
    
    print(f"\n Found {query_found}/{len(gnd['qimlist'])} query images")
    
    # ===== XỬ LÝ DATABASE IMAGES =====
    print(f"\n Processing {len(gnd['imlist'])} DATABASE images...")
    db_found = 0
    
    for i, im_name in enumerate(gnd['imlist']):
        img_path = find_image_in_jpg(source_dir, im_name)
        
        if img_path:
            ext = os.path.splitext(img_path)[1]
            dest_path = os.path.join(db_dir, f"db_{i:06d}{ext}")
            
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(db_dir, f"db_{i:06d}_{counter}{ext}")
                counter += 1
            
            shutil.copy2(img_path, dest_path)
            db_found += 1
            
            if (i + 1) % 500 == 0:
                print(f"   Progress: {i+1}/{len(gnd['imlist'])}")
        else:
            if i < 5:
                print(f" DB {i:06d}: {im_name} NOT FOUND")
    
    print(f"\n Found {db_found}/{len(gnd['imlist'])} database images")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Query images: {query_found}/{len(gnd['qimlist'])}")
    print(f"Database images: {db_found}/{len(gnd['imlist'])}")
    print(f"Query folder: {query_dir}")
    print(f"Database folder: {db_dir}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    if not os.path.exists(PKL_FILE):
        print(f"PKL file not found: {PKL_FILE}")
        sys.exit(1)
    
    print("=" * 60)
    print("SPLITTING QUERY AND DATABASE IMAGES")
    print("=" * 60)
    
    # Kiểm tra thư mục jpg
    if not os.path.exists(IMAGE_SOURCE_DIR):
        print(f"JPG folder not found: {IMAGE_SOURCE_DIR}")
        sys.exit(1)
    
    # Đếm ảnh trong jpg
    jpg_files = [f for f in os.listdir(IMAGE_SOURCE_DIR) 
                 if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    print(f"Found {len(jpg_files)} images in jpg folder")
    
    if len(jpg_files) == 0:
        print("No images found in jpg folder!")
        sys.exit(1)
    
    split_query_db_images(
        pkl_file=PKL_FILE,
        source_dir=IMAGE_SOURCE_DIR,
        output_dir=OUTPUT_DIR
    )
    
    print("\nDone!")