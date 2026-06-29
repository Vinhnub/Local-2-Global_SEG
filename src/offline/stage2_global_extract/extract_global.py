import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import math
from pathlib import Path
import sys
BASE_DIR = ''
# Add src to pythonpath
sys.path.append(os.path.join(str(BASE_DIR), 'src', 'core', 'SuperGlobal-main'))

from model.CVNet_Rerank_model import CVNet_Rerank



def extract_dataset(dataset_name, model, device):
    print(f"Extracting CVNet-R101 features for {dataset_name}...")
    
    # Paths
    img_dir = os.path.join(str(BASE_DIR), "data", "datasets", dataset_name, "jpg")

    out_dir_q = os.path.join(
        str(BASE_DIR), "output", "stage2", "features", dataset_name, "query"
    )

    out_dir_db = os.path.join(
        str(BASE_DIR), "output", "stage2", "features", dataset_name, "database"
    )

    os.makedirs(out_dir_q, exist_ok=True)
    os.makedirs(out_dir_db, exist_ok=True)
    # Lists
    q_list_path = os.path.join(
    str(BASE_DIR), "google-research", "cann", f"{dataset_name}_query_names.txt"
)

    db_list_path = os.path.join(
        str(BASE_DIR), "google-research", "cann", f"{dataset_name}_database_names.txt"
    )
    
    with open(q_list_path, "r") as f:
        q_names = [line.strip() for line in f if line.strip()]
    with open(db_list_path, "r") as f:
        db_names = [line.strip() for line in f if line.strip()]
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Find all images recursively and map stem to path
    img_path_map = {}

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                stem = os.path.splitext(file)[0]
                img_path_map[stem] = os.path.join(root, file)
        
    scale = 3
    
    def extract_image(img_name, out_dir, is_query=False):
        if img_name not in img_path_map:
            print(f"Warning: Image {img_name} not found in {img_dir}")
            return
            
        img_path = img_path_map[img_name]
            
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = model.extract_global_descriptor(img_tensor, True, True, True, scale)
                feat = feat.cpu().numpy().squeeze() # (2048,)
                
            np.save(os.path.join(out_dir, f"{img_name}.npy"), feat)
            print (f"Extracted {img_name} to {out_dir}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("Extracting queries...")
    for q in q_names:
        extract_image(q, out_dir_q, is_query=True)
        
    print("Extracting database...")
    for idx, db in enumerate(db_names):
        if (idx+1) % 500 == 0:
            print(f"Processed {idx+1}/{len(db_names)} database images...")
        extract_image(db, out_dir_db, is_query=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = CVNet_Rerank(RESNET_DEPTH=101, REDUCTION_DIM=2048, relup=False)
    
    # Load weights
    weight_path = BASE_DIR + "model_weights" + '/' + "CVPR2022_CVNet_R101.pyth"
    print(f"Loading weights from {weight_path}...")
    state_dict = torch.load(weight_path, map_location='cpu')
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print("Fallback: loading into encoder_q")
        model.encoder_q.load_state_dict(new_state_dict, strict=False)
        
    model = model.to(device)
    model.eval()
    
    #extract_dataset("roxford5k", model, device)
    extract_dataset("rparis6k", model, device)

if __name__ == "__main__":
    main()
