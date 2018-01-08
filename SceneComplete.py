import numpy as np
import cv2
import scipy
import scipy.spatial
from FeatureExtractor import *
from GraphCut import graphcut
from utils import *

img = cv2.imread("./dataset/img.png")
mask = cv2.imread("./dataset/mask.png")

# TODO: GIST under mask
gist = get_gist(img)

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
    candidates.append(candidates)
    cv2.imwrite("./candidates/{}.png".format(i), candidate)

info("Selecting matching position", domain=__file__)
# TODO: Search for matching position
# Score part 2
match_cost = []

# Modify mask format for graph cut
rawmask = mask[0].astype('uint8')  # 3 channels wwwwww
rawmask[rawmask > 0] = 1

mask_boundary2 = cv2.dilate(rawmask, np.ones(3, 3))
dilated_mask = cv2.dilate(mask_boundary2, np.ones(3, 3), iterations=78)
mask = cv2.dilate(dilated_mask, np.ones(3, 3))

mask[rawmask > 0] = 0
mask[np.bitwise_and(mask_boundary2 > 0, rawmask == 0)] = 2
mask[np.bitwise_and(mask > 0, dilated_mask == 0)] = 1

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
    # TODO: Poisson blending
    mixture = None
    cv2.imwrite("./results/{}.png".format(i), mixture)

info("Completed!", domain=__file__)