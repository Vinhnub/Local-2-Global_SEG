# PROCESS_IMAGES_WITH_FIRE.py
import os
import sys
import importlib.util

fire_path = "/mnt/d/SEGMain/SEGCode/fire"

spec = importlib.util.spec_from_file_location("dataset", os.path.join(fire_path, "dataset.py"))
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)

configdataset = dataset.configdataset

import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile
import numpy as np

# Import fire_network từ fire
sys.path.insert(0, fire_path)
import fire_network

DATASET_DIR = "/mnt/d/SEGMain/SEGCode/revisitop/data/datasets/rparis6k/jpg"
PKL_FILE = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/data/datasets/rparis6k/gnd_rparis6k.pkl" 
OUTPUT_DIR = "/mnt/d/SEGMain/SEGCode/Local-2-Global_SEG/data/features/fire"

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

# Process Dataset
def process_dataset(dataset_dir, pkl_file, output_dir, num_features=600):
    
    # Load model
    net, norm_rgb, device = load_fire_model()
    
    dataset_name = os.path.basename(pkl_file).replace('gnd_', '').replace('.pkl', '')
    print(f'\n>> Processing: {dataset_name}')
    
    cfg = configdataset(dataset_name, dataset_dir)
    
    # Tạo thư mục output
    output_root = os.path.join(output_dir, dataset_name)
    query_dir = os.path.join(output_root, 'query')
    db_dir = os.path.join(output_root, 'database')
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    
    print(f"Output: {output_root}")
    print(f"{cfg['nq']} queries, {cfg['n']} database images")
    print("-" * 60)
    
    # QUERY 
    print(f"\nProcessing {cfg['nq']} QUERY images...")
    for i in range(cfg['nq']):
        try:
            qim = pil_loader(cfg['qim_fname'](cfg, i))
            qim_cropped = qim.crop(cfg['gnd'][i]['bbx'])
            features = extract_fire_features(qim_cropped, net, norm_rgb, device, num_features)
            
            if features is not None:
                output_path = os.path.join(query_dir, f"query_{i:04d}.npy")
                np.save(output_path, features)
                print(f"Query {i+1:3d}/{cfg['nq']}: {features.shape}")
            else:
                print(f"Query {i+1:3d}/{cfg['nq']}: Failed")
        except Exception as e:
            print(f"Query {i+1:3d}/{cfg['nq']}: {e}")
    
    # ===== XỬ LÝ DATABASE =====
    print(f"\nProcessing {cfg['n']} DATABASE images...")
    for i in range(cfg['n']):
        try:
            im = pil_loader(cfg['im_fname'](cfg, i))
            features = extract_fire_features(im, net, norm_rgb, device, num_features)
            
            if features is not None:
                output_path = os.path.join(db_dir, f"db_{i:06d}.npy")
                np.save(output_path, features)
                print(f"DB {i+1:6d}/{cfg['n']}: {features.shape}")
            else:
                print(f"DB {i+1:6d}/{cfg['n']}: Failed")
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{cfg['n']}")
        except Exception as e:
            print(f"DB {i+1:6d}/{cfg['n']}: {e}")
    
    print("\n" + "=" * 60)
    print(f"DONE Features saved to: {output_root}")
    print("=" * 60)
    return output_root

# MAIN
if __name__ == "__main__":
    if not os.path.exists(PKL_FILE):
        print(f"PKL file not found: {PKL_FILE}")
        print("Please check the path!")
        sys.exit(1)
    
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)
    
    process_dataset(
        dataset_dir=DATASET_DIR,
        pkl_file=PKL_FILE,
        output_dir=OUTPUT_DIR,
        num_features=600
    )
    
    print("\nDone!")