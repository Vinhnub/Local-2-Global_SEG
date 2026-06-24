import os
import sys
import importlib.util

fire_path = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/fire"

spec = importlib.util.spec_from_file_location("dataset", os.path.join(fire_path, "dataset.py"))
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)

configdataset = dataset.configdataset

import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile
import numpy as np

sys.path.insert(0, fire_path)
import fire_network

DATASET_DIR = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/data/datasets"
PKL_FILE = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/data/datasets/roxford5k/gnd_roxford5k.pkl" 
OUTPUT_DIR = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/data/feature/fire"

def load_fire_model():
    """Load FIRe model and return network, transform, device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model_dir = os.path.join(fire_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fire_SfM_120k.pth")
    
    if not os.path.exists(model_path):
        print("Model not found!")
        print("Please download model first. Run: python evaluate.py eval_fire.yml")
        sys.exit(1)
    
    print("Loading FIRe model...")
    state = torch.load(model_path, map_location=device)
    state["net_params"]["pretrained"] = None
    net = fire_network.init_network(**state["net_params"])
    net.load_state_dict(state["state_dict"])
    net.to(device)
    net.eval()
    print("Model loaded successfully!")
    
    norm_rgb = tvf.Normalize(
        **dict(zip(["mean", "std"], net.runtime["mean_std"]))
    )
    
    return net, norm_rgb, device

# Extract FIRe Features
def extract_fire_features(image, net, norm_rgb, device, num_features=600):
    try:
        img_tensor = tvf.ToTensor()(image).unsqueeze(0)
        img_tensor = norm_rgb(img_tensor)
        img_tensor = img_tensor.to(device)
        
        scales = [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25]
        
        with torch.no_grad():
            local_feats = net.forward_local(
                img_tensor, 
                features_num=num_features, 
                scales=scales
            )
        
        features = local_feats[0].squeeze(-1).squeeze(0).t()
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def pil_loader(path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_image_name(filepath):
    """Lấy tên file không có extension"""
    return os.path.splitext(os.path.basename(filepath))[0]

# Process Dataset
def process_dataset(dataset_dir, pkl_file, output_dir, num_features=600):
    
    # Load model
    net, norm_rgb, device = load_fire_model()
    
    dataset_name = os.path.basename(pkl_file).replace('gnd_', '').replace('.pkl', '')
    print(f'\n>> Processing: {dataset_name}')
    
    cfg = configdataset(dataset_name, dataset_dir)
    
    # ===== Lấy ground truth đúng cách =====
    # cfg['gnd'] là dict có key 'gnd' chứa danh sách ground truth
    if isinstance(cfg['gnd'], dict) and 'gnd' in cfg['gnd']:
        gnd_data = cfg['gnd']['gnd']  # Danh sách ground truth
    else:
        gnd_data = cfg['gnd']  # Nếu đã là list
    
    # ===== DEBUG: Kiểm tra cấu hình =====
    print("\n" + "=" * 60)
    print("DEBUG: Checking configuration")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Number of queries: {cfg['nq']}")
    print(f"Number of database: {cfg['n']}")
    print(f"Type of cfg['gnd']: {type(cfg['gnd'])}")
    print(f"Type of gnd_data: {type(gnd_data)}")
    print(f"Number of ground truth entries: {len(gnd_data)}")
    
    if len(gnd_data) > 0:
        print(f"First gnd entry: {gnd_data[0]}")
        print(f"Has 'bbx': {'bbx' in gnd_data[0]}")
    
    # Kiểm tra file ảnh
    print("\nChecking image files...")
    for i in range(min(3, cfg['nq'])):
        try:
            qim_path = cfg['qim_fname'](cfg, i)
            exists = os.path.exists(qim_path)
            print(f"Query {i}: {qim_path} -> {'EXISTS' if exists else 'MISSING'}")
        except Exception as e:
            print(f"Query {i}: Error - {e}")
    
    for i in range(min(3, cfg['n'])):
        try:
            im_path = cfg['im_fname'](cfg, i)
            exists = os.path.exists(im_path)
            print(f"DB {i}: {im_path} -> {'EXISTS' if exists else 'MISSING'}")
        except Exception as e:
            print(f"DB {i}: Error - {e}")
    print("=" * 60)
    
    # Tạo thư mục output
    output_root = os.path.join(output_dir, dataset_name)
    query_dir = os.path.join(output_root, 'query')
    db_dir = os.path.join(output_root, 'database')
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    
    print(f"Output root: {output_root}")
    print(f"Query directory: {query_dir}")
    print(f"Database directory: {db_dir}")
    print("-" * 60)
    
    # ===== XỬ LÝ QUERY =====
    print(f"\nProcessing {cfg['nq']} QUERY images...")
    query_success = 0
    query_failed = 0
    
    for i in range(cfg['nq']):
        try:
            qim_path = cfg['qim_fname'](cfg, i)
            
            # Kiểm tra file tồn tại
            if not os.path.exists(qim_path):
                print(f"Query {i+1:3d}/{cfg['nq']}: File not found - {qim_path}")
                query_failed += 1
                continue
            
            qim = pil_loader(qim_path)
            
            # Lấy bounding box từ ground truth
            if i < len(gnd_data):
                bbx = gnd_data[i]['bbx']
                qim_cropped = qim.crop(bbx)
            else:
                print(f"Query {i+1:3d}/{cfg['nq']}: No ground truth for index {i}")
                query_failed += 1
                continue
            
            features = extract_fire_features(qim_cropped, net, norm_rgb, device, num_features)
            
            if features is not None:
                original_name = get_image_name(qim_path)
                output_path = os.path.join(query_dir, f"{original_name}.npy")
                np.save(output_path, features)
                print(f"Query {i+1:3d}/{cfg['nq']}: {original_name} -> {features.shape}")
                query_success += 1
            else:
                print(f"Query {i+1:3d}/{cfg['nq']}: Failed - features is None")
                query_failed += 1
                
        except Exception as e:
            print(f"Query {i+1:3d}/{cfg['nq']}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            query_failed += 1
    
    print(f"\nQuery processing complete: {query_success} success, {query_failed} failed")
    
    # ===== XỬ LÝ DATABASE =====
    print(f"\nProcessing {cfg['n']} DATABASE images...")
    db_success = 0
    db_failed = 0
    
    for i in range(cfg['n']):
        try:
            im_path = cfg['im_fname'](cfg, i)
            
            # Kiểm tra file tồn tại
            if not os.path.exists(im_path):
                print(f"DB {i+1:6d}/{cfg['n']}: File not found - {im_path}")
                db_failed += 1
                continue
            
            im = pil_loader(im_path)
            features = extract_fire_features(im, net, norm_rgb, device, num_features)
            
            if features is not None:
                original_name = get_image_name(im_path)
                output_path = os.path.join(db_dir, f"{original_name}.npy")
                np.save(output_path, features)
                print(f"DB {i+1:6d}/{cfg['n']}: {original_name} -> {features.shape}")
                db_success += 1
            else:
                print(f"DB {i+1:6d}/{cfg['n']}: Failed - features is None")
                db_failed += 1
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{cfg['n']}")
                
        except Exception as e:
            print(f"DB {i+1:6d}/{cfg['n']}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            db_failed += 1
    
    print(f"\nDatabase processing complete: {db_success} success, {db_failed} failed")
    
    # ===== KẾT QUẢ =====
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Queries: {query_success}/{cfg['nq']} successful")
    print(f"  Database: {db_success}/{cfg['n']} successful")
    print(f"  Features saved to: {output_root}")
    print("=" * 60)
    
    return output_root

# MAIN
if __name__ == "__main__":
    try:
        print("=" * 60)
        print("FIRE FEATURE EXTRACTOR - Local-to-Global SEG")
        print("=" * 60)
        
        # Kiểm tra file PKL
        if not os.path.exists(PKL_FILE):
            print(f"PKL file not found: {PKL_FILE}")
            print("Please check the path!")
            sys.exit(1)
        else:
            print(f"PKL file found: {PKL_FILE}")
        
        # Kiểm tra thư mục dataset
        dataset_path = os.path.join(DATASET_DIR, "rparis6k", "jpg")
        if not os.path.exists(dataset_path):
            print(f"Dataset directory not found: {dataset_path}")
            print("Please check the path!")
            sys.exit(1)
        else:
            print(f"Dataset directory found: {dataset_path}")
            # Liệt kê 5 file đầu tiên
            files = os.listdir(dataset_path)[:5]
            print(f"First 5 files in dataset: {files}")
        
        # Kiểm tra thư mục output
        print(f"Output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\nStarting feature extraction...")
        print("-" * 60)
        
        process_dataset(
            dataset_dir=DATASET_DIR,
            pkl_file=PKL_FILE,
            output_dir=OUTPUT_DIR,
            num_features=600
        )
        
        print("\nDone! Feature extraction completed successfully!")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)