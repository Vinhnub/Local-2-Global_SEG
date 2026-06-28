import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path("fire/lib/asmk")))
import asmk
from asmk.asmk_method import ASMKMethod

# Mock features: 1000 images, 600 keypoints each, 128 dim
np.random.seed(0)
dummy_features = [np.random.rand(600, 128).astype(np.float32) for _ in range(10)]
dummy_imids = [np.full(600, i, dtype=np.int32) for i in range(10)]

des_train = dummy_features[0]
method = ASMKMethod.initialize_untrained({})
method = method.train_codebook(des_train)
print("Codebook trained!")
