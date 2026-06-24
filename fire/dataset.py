# dataset.py - Configuration for ROxford and RParis datasets
import os
import pickle
import numpy as np

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
        'im_fname': lambda cfg, i: os.path.join(im_dir, cfg['imlist'][i] + '.jpg'),
        'qim_fname': lambda cfg, i: os.path.join(qim_dir, cfg['qimlist'][i] + '.jpg'),
    }
    
    return cfg
