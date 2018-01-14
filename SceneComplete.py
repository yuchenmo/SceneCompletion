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
distances, indexes = tree.query(gist, k=2, eps=1e-8, p=2)  # Score part 1
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

fullmask = rawmask.copy()
fullmask[mask > 0] = 1

info("Selecting matching position", domain=__file__)
# Score part 2
match_info = matchall(img, candidates, (mask > 0))
match_cost = list(map(lambda x: x[0], match_info))
np.savez("Matchinfo.npz", Data=match_info)
info("Calculating boundary", domain=__file__)

maxflow = []   # Score part 3
for i, candidate in enumerate(candidates):
    img1grad, img2grad = cv2.Sobel(
        img, -1, 1, 1), cv2.Sobel(candidate, -1, 1, 1)
    scale, offset_y, offset_x = match_info[i][1], match_info[i][2], match_info[i][3]
    img2grad = cv2.resize(img2grad, (0, 0), fx=scale, fy=scale)
    img2grad = img2grad[offset_y: offset_y + mask.shape[0], offset_x: offset_x + mask.shape[1]]
    # embed()
    segmap, cost = graphcut(img1grad, img2grad, mask)
    maxflow.append(cost)

final_score = np.array(distances) + np.array(match_cost) + np.array(maxflow)
winners = final_score.argsort()[:20]

ensure_dir("./results", renew=True)
show_img = img.copy().astype('float32')
show_img[mask > 0] *= 0.5
show_img = show_img.astype('uint8')
cv2.imwrite("original.png", show_img)
for i in winners:
    candidate = candidates[i]
    scale, offset_y, offset_x, y1, x1, roilen = match_info[i][1], match_info[i][2], match_info[i][3], match_info[i][4], match_info[i][5], match_info[i][6]
    candidate = cv2.resize(candidate, (0, 0), fx=scale, fy=scale)
    candidate = candidate[offset_y: offset_y + roilen, offset_x: offset_x + roilen]
    candidate_mask = fullmask[y1: y1 + roilen, x1: x1 + roilen]
    # candidate_mask = cv2.erode(candidate_mask, np.ones((3, 3)), iterations=)

    cv2.circle(show_img, (int(y1 + roilen / 2), int(x1 + roilen / 2)), 3, (0, 255, 0), 1)
    cv2.rectangle(show_img, (x1, y1), (x1 + roilen, y1 + roilen), (0, 0, 255), 2)
    cv2.imwrite("original1.png", show_img)
    mixture = cv2.seamlessClone(candidate, img, (candidate_mask > 0).astype('uint8') * 255, (int(x1 + roilen / 2), int(y1 + roilen / 2)), cv2.NORMAL_CLONE)
    cv2.imwrite("./results/{}.png".format(i), mixture)

info("Completed!", domain=__file__)