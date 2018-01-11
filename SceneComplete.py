import numpy as np
import cv2
import scipy
import scipy.spatial
from FeatureExtractor import get_gist_C_implementation
from GraphCut import graphcut
from ContextMatch import matchall
from IPython import embed
from utils import *

img = cv2.imread("./dataset/inputs/test.bmp")
mask = cv2.imread("./dataset/inputs/testmask.bmp")[:, :, 0]

gist = get_gist_C_implementation(img, mask)

info("Loading image dataset", domain=__file__)
dataset = np.load("./dataset/feature.npz")
path, feat = dataset['Path'], dataset['Feat']

info("Matching features", domain=__file__)
tree = scipy.spatial.cKDTree(feat)
distances, indexes = tree.query(gist, k=200, eps=1e-8, p=2)  # Score part 1
indexes = np.array(indexes).astype('int32')

candidates = []
ensure_dir("./candidates", renew=True)
for i, idx in enumerate(indexes):
    candidate = cv2.imread(path[idx])
    candidates.append(candidate)
    cv2.imwrite("./candidates/{}.png".format(i), candidate)

# Modify mask format
rawmask = mask.astype('uint8')
rawmask[rawmask > 0] = 1

mask_boundary2 = cv2.dilate(rawmask, np.ones((3, 3), dtype=np.uint8))
dilated_mask = cv2.dilate(mask_boundary2, np.ones((3, 3), dtype=np.uint8), iterations=78)
mask = cv2.dilate(dilated_mask, np.ones((3, 3), dtype=np.uint8))

mask[rawmask > 0] = 0
mask[np.bitwise_and(mask_boundary2 > 0, rawmask == 0)] = 2
mask[np.bitwise_and(mask > 0, dilated_mask == 0)] = 1

info("Selecting matching position", domain=__file__)
# Score part 2
match_info = matchall(img, candidates, (mask > 0))
match_cost = list(map(lambda x: x[0], match_info))

info("Calculating boundary", domain=__file__)

maxflow = []   # Score part 3
for i, candidate in enumerate(candidates):
    img1grad, img2grad = cv2.Sobel(
        img, -1, 1, 1), cv2.Sobel(candidates, -1, 1, 1)
    segmap, cost = graphcut(img1grad, img2grad, mask)
    maxflow.append(cost)

final_score = np.array(distances) + np.array(match_cost) + np.array(maxflow)
winners = final_score.argsort()[:20]

ensure_dir("./results", renew=True)
for i in winners:
    candidate = candidates[i]
    candidate_info = match_info[i]
    candidate = cv2.resize(candidate, 0, fx=candidate_info[1], fy=candidate_info[1])
    mixture = cv2.seamlessClone(candidate, img, (mask > 0), (candidate_info[2], candidate_info[3]), cv2.MIXED_CLONE)
    cv2.imwrite("./results/{}.png".format(i), mixture)

info("Completed!", domain=__file__)