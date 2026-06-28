# dataset.py - Configuration for ROxford and RParis datasets
import os
import pickle
import numpy as np

_image_cache = {}

def _find_image_path(base_dir, name):
    # Thử đường dẫn phẳng trước
    flat_path = os.path.join(base_dir, name + '.jpg')
    if os.path.exists(flat_path):
        return flat_path
        
    # Nếu không thấy, quét đệ quy và dùng cache
    global _image_cache
    if base_dir not in _image_cache:
        print(f"Scanning directory recursively for images: {base_dir}...")
        _image_cache[base_dir] = {}
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        for root, _, files in os.walk(base_dir):
            for f in files:
                fname, ext = os.path.splitext(f)
                if ext in extensions:
                    _image_cache[base_dir][fname.lower()] = os.path.join(root, f)
        print(f"Directory scan complete. Found {len(_image_cache[base_dir])} images.")
        
    cache = _image_cache[base_dir]
    if name.lower() in cache:
        return cache[name.lower()]
        
    # Thử thêm tiền tố "paris_"
    if f"paris_{name}".lower() in cache:
        return cache[f"paris_{name}".lower()]
        
    return flat_path # fall back

def configdataset(dataset_name, dir_main):
    """
    Load configuration for ROxford or RParis dataset
    
    Args:
        dataset_name: 'roxford5k' or 'rparis6k'
        dir_main: Directory containing dataset folders
    
    Returns:
        cfg: Configuration dictionary
    """
    # Đường dẫn đến file annotation
    if dataset_name == 'roxford5k':
        gnd_file = os.path.join(dir_main, 'roxford5k', 'gnd_roxford5k.pkl')
        im_dir = os.path.join(dir_main, 'roxford5k', 'jpg')
        qim_dir = os.path.join(dir_main, 'roxford5k', 'jpg')
    elif dataset_name == 'rparis6k':
        gnd_file = os.path.join(dir_main, 'rparis6k', 'gnd_rparis6k.pkl')
        im_dir = os.path.join(dir_main, 'rparis6k', 'jpg')
        qim_dir = os.path.join(dir_main, 'rparis6k', 'jpg')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load ground truth
    with open(gnd_file, 'rb') as f:
        gnd = pickle.load(f)
    
    # Tạo config
    cfg = {
        'gnd': gnd,
        'qimlist': gnd['qimlist'],
        'imlist': gnd['imlist'],
        'nq': len(gnd['qimlist']),
        'n': len(gnd['imlist']),
        'im_fname': lambda cfg, i: _find_image_path(im_dir, cfg['imlist'][i]),
        'qim_fname': lambda cfg, i: _find_image_path(qim_dir, cfg['qimlist'][i]),
    }
    
    return cfg

