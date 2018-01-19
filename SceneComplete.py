import numpy as np
import cv2
import scipy
import scipy.spatial
from tqdm import tqdm
from FeatureExtractor import get_gist_C_implementation
from GraphCut import graphcut
from ContextMatch import matchall
from IPython import embed
from utils import *

TESTNUM = 1158

img = cv2.imread("./dataset/inputs/IMG_{}.bmp".format(TESTNUM))
mask = cv2.imread("./dataset/inputs/IMG_{}_mask.bmp".format(TESTNUM))[:, :, 0]

# Modify mask format
rawmask = mask.astype('uint8')
rawmask[rawmask > 0] = 1
rawmask3d = np.expand_dims(rawmask, axis=2)
rawmask3d = np.concatenate((rawmask3d, rawmask3d, rawmask3d), axis=2)

mask_boundary2 = cv2.dilate(rawmask, np.ones((3, 3), dtype=np.uint8))
dilated_mask = cv2.dilate(mask_boundary2, np.ones((3, 3), dtype=np.uint8), iterations=78)
mask = cv2.dilate(dilated_mask, np.ones((3, 3), dtype=np.uint8))

mask *= 3
mask[np.bitwise_and(mask > 0, dilated_mask == 0)] = 1
mask[np.bitwise_and(mask_boundary2 > 0, rawmask == 0)] = 2
mask[rawmask > 0] = 0

fullmask = mask.copy()
fullmask[rawmask > 0] = 3

nzy, nzx = np.nonzero(mask > 0)
y1, y2, x1, x2 = min(nzy), max(nzy), min(nzx), max(nzx)
roiy, roix = y2 - y1, x2 - x1


gist = get_gist_C_implementation(img, mask)


info("Loading image dataset", domain=__file__)
dataset = np.load("./dataset/feature.npz")
path, feat = dataset['Path'], dataset['Feat']

info("Matching features", domain=__file__)
tree = scipy.spatial.cKDTree(feat)
distances, indexes = tree.query(gist, k=200, eps=1e-8, p=2)  # Score part 1
indexes = np.array(indexes).astype('int32')
candidates = []

if not op.exists("Matchinfo_{}.npz".format(TESTNUM)):
    info("Loading image dataset", domain=__file__)
    dataset = np.load("./dataset/feature.npz")
    path, feat = dataset['Path'], dataset['Feat']

    info("Matching features", domain=__file__)
    tree = scipy.spatial.cKDTree(feat)
    distances, indexes = tree.query(gist, k=200, eps=1e-8, p=2)  # Score part 1
    indexes = np.array(indexes).astype('int32')
    candidates = []

    for i, idx in enumerate(indexes):
        candidate = cv2.imread(path[idx])
        scale = max(roiy / candidate.shape[0], roix / candidate.shape[1])
        if scale > 1.0:
            candidate = cv2.resize(candidate, (0, 0), fx=scale * 1.22, fy=scale * 1.22)
        candidates.append(candidate)

    info("Selecting matching position", domain=__file__)
    match_info = matchall(img, candidates, (mask > 0))
    match_cost = list(map(lambda x: x[0], match_info))
    np.savez("Matchinfo_{}.npz".format(TESTNUM), Matchinfo=match_info, Candidates=candidates, Path=path, Feat=feat, Distances=distances)
else:
    info("Npz found. Loading...", domain=__file__)
    file = np.load("Matchinfo_{}.npz".format(TESTNUM))
    match_info, candidates, path, feat, distances = file['Matchinfo'], file['Candidates'], file['Path'], file['Feat'], file['Distances']
    match_cost = list(map(lambda x: x[0], match_info))
    

info("Calculating boundary", domain=__file__)
maxflow = []   # Score part 3
segmaps = []

for i in range(len(candidates)):
    scale, boa_y, boa_x, roiy1, roix1, roiy2, roix2 = tuple(match_info[i][1:])
    boa_y, boa_x, roiy1, roix1, roiy2, roix2 = int(boa_y), int(boa_x), int(roiy1), int(roix1), int(roiy2), int(roix2)
    candidates[i] = cv2.resize(candidates[i], (0, 0), fx=scale, fy=scale)
    if boa_y < 0:
        candidates[i] = candidates[i][-boa_y:, :]
        boa_y = 0
    if boa_x < 0:
        candidates[i] = candidates[i][:, -boa_x:]
        boa_x = 0
    candidates[i] = np.pad(candidates[i], [(boa_y, 0), (boa_x, 0), (0, 0)], 'reflect')
    if img.shape[0] > candidates[i].shape[0]:
        candidates[i] = np.pad(candidates[i], [(0, img.shape[0] - candidates[i].shape[0]), (0, 0), (0, 0)], 'reflect')
    if img.shape[1] > candidates[i].shape[1]:
        candidates[i] = np.pad(candidates[i], [(0, 0), (0, img.shape[1] - candidates[i].shape[1]), (0, 0)], 'reflect')
    candidates[i] = candidates[i][:img.shape[0], :img.shape[1]]
    # cv2.imwrite("./candidates/{}_processed.png".format(i), (candidates[i] * 0.7 + img * 0.3).astype('uint8')) 

if not op.exists("Segmaps_{}.npz".format(TESTNUM)):
    info("Running GraphCut algorithm", domain=__file__)
    for i in tqdm(range(len(candidates))):
        candidate = candidates[i]
        img1grad, img2grad = cv2.Sobel(
            img, -1, 1, 1), cv2.Sobel(candidate, -1, 1, 1)
        
        scale, boa_y, boa_x, roiy1, roix1, roiy2, roix2 = tuple(match_info[i][1:])
        # scale, offset_y, offset_x, roiy1, roix1, roiy2, roix2 = match_info[i][1], match_info[i][2], match_info[i][3], match_info[i][-4], match_info[i][-3], match_info[i][-2], match_info[i][-1]
        # roimask, roi = mask[roiy1: roiy2, roix1: roix2], img1grad[roiy1: roiy2, roix1: roix2]
        segmap, cost = graphcut(img1grad, img2grad, mask)
        segmap[rawmask > 0] = 2
        maxflow.append(cost)
        segmaps.append(segmap)
    np.savez("Segmaps_{}.npz".format(TESTNUM), Segmaps=segmaps, Maxflow=np.array(maxflow))
else:
    info("Npz found. Loading...", domain=__file__)
    file = np.load("Segmaps_{}.npz".format(TESTNUM)) 
    segmaps, maxflow = file['Segmaps'], file['Maxflow']

final_score = np.array(distances) + np.array(match_cost) + np.array(maxflow)
winners = final_score.argsort()[:100]

info("Applying Poisson blending", domain=__file__)
ensure_dir("./results", renew=True)
# show_img = img.copy().astype('float32')
# show_img[mask > 0] *= 0.5
# show_img = show_img.astype('uint8')
# cv2.imwrite("original.png", show_img)
for order, i in enumerate(winners):
    candidate = candidates[i]
    segmap = segmaps[i]
    scale, boa_y, boa_x, roiy1, roix1, roiy2, roix2 = tuple(match_info[i][1:])
    # scale, offset_y, offset_x, y1, x1, roilen = match_info[i][1], match_info[i][2], match_info[i][3], match_info[i][4], match_info[i][5], match_info[i][6]
    # candidate = cv2.resize(candidate, (0, 0), fx=scale, fy=scale)
    # candidate = candidate[offset_y: offset_y + roilen, offset_x: offset_x + roilen]
    # candidate_mask = fullmask[y1: y1 + roilen, x1: x1 + roilen]
    # candidate_mask = segmap[offset_y: offset_y + roilen, offset_x: offset_x + roilen]
    candidate_mask = (segmap >= 2)
    # mixture = img.copy()
    # mixture[candidate_mask > 0] = candidate[candidate_mask]
    points = np.where(candidate_mask > 0)
    ymin, ymax, xmin, xmax = min(points[0]), max(points[0]), min(points[1]), max(points[1])
    ycent, xcent = int((ymin + ymax) // 2), int((xmin + xmax) // 2)

    mixture = cv2.seamlessClone(candidate[ymin: ymax, xmin: xmax], img, candidate_mask.astype('uint8')[ymin: ymax, xmin: xmax] * 255, (xcent, ycent), cv2.NORMAL_CLONE)
    cv2.imwrite("./results/{}_{}.png".format(order, i), mixture)

info("Scene completion finished", domain=__file__)