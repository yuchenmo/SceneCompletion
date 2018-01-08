import numpy as np
import os
import os.path as op
import cv2
from tqdm import tqdm
from FeatureExtractor import get_gist
from utils import ensure_dir, info

input_dir = "./dataset/raw_image"
catalog = {}
paths = []
feats = []

for (root, dirs, files) in os.walk(input_dir):
    for f in files:
        if f.split('.')[-1].lower() in ['jpg', 'bmp', 'png']:
            path = op.join(root, f)
            catalog[len(catalog)] = {
                'path': path  # For possible further metadata
            }

info("Extracting GIST descriptor", domain=__file__)

for i in tqdm(catalog):
    vec = get_gist(cv2.imread(catalog[i]['path']))

    paths.append(catalog[i]['path'])
    feats.append(vec)

np.savez("./dataset/feature.npz", Path=paths, Feat=feats)
info("Preprocess completed! {} images loaded into dataset".format(len(paths)), domain=__file__)

    
    

