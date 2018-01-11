import numpy as np
import os
import os.path as op
import cv2
from tqdm import tqdm
import multiprocessing
from FeatureExtractor import get_gist_C_implementation
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

processnum = 16
unit = len(catalog) // processnum + 1
def getfeat(start, end, use_tqdm=False):
    subpaths, subfeats = [], []
    for i in (range(start, end) if not use_tqdm else tqdm(range(start, end))):
        img = cv2.imread(catalog[i]['path'])
        vec = get_gist_C_implementation(img)
        subpaths.append(catalog[i]['path'])
        subfeats.append(vec)
    return subpaths, subfeats

pool = multiprocessing.Pool()
processes = []

info("Starting worker processes", domain=__file__)
for pid in tqdm(range(1, processnum)):
    processes.append(pool.apply_async(getfeat, args=(pid * unit, min((pid + 1) * unit, len(catalog)))))

subpath, subfeat = getfeat(0, unit, use_tqdm=True)
paths, feats = subpath, subfeat

info("Joining worker processes", domain=__file__)
for pid in tqdm(range(processnum - 1)):
    subpath, subfeat = processes[pid].get()
    paths += subpath
    feats += subfeat

"""
for i in tqdm(catalog):
    img = cv2.imread(catalog[i]['path'])
    vec = get_gist_C_implementation(img)
    paths.append(catalog[i]['path'])
    feats.append(vec)
"""
np.savez("./dataset/feature.npz", Path=paths, Feat=feats)
info("Preprocess completed! {} images loaded into dataset".format(len(paths)), domain=__file__)

    
    

