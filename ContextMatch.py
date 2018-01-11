import numpy as np
from scipy.signal import convolve2d
import cv2
import multiprocessing
from tqdm import tqdm
from functools import reduce
from utils import info
from FeatureExtractor import get_lab, get_texture
from IPython import embed


def _tosquare(y1, y2, x1, x2, ymax, xmax):
    dy, dx = y2 - y1, x2 - x1
    if dy > dx:
        dd = dy - dx
        if x2 + dd < xmax:
            return y1, y2, x1, x2 + dd
        elif dy < xmax:
            return y1, y2, x1 - (dd - (xmax - x2)), xmax
        else:
            return y1, y2, x1, x2
    elif dx > dy:
        dd = dx - dy
        if y2 + dd < ymax:
            return y1, y2 + dd, x1, x2
        elif dx < ymax:
            return y1 - (dd - (ymax - y2)), ymax, x1, x2
        else:
            return y1, y2, x1, x2
    return y1, y2, x1, x2


def match(img1, img2, mask, show):
    """
    Here mask is the 80px boundary area.
    Img1 is input. Img2 is candidate.
    Scale(0.81, 0.9, 1) + Translation
    SSD error, L*a*b* color space, median filter.
    """

    img2 = np.array(img2)
    scales = [1., 0.9, 0.81]

    imgs = [cv2.resize(img2, (0, 0), fx=x, fy=x) for x in scales]
    nzy, nzx = np.nonzero(mask > 0)
    y1, y2, x1, x2 = min(nzy), max(nzy), min(nzx), max(nzx)
    y1, y2, x1, x2 = _tosquare(y1, y2, x1, x2, mask.shape[0], mask.shape[1])
    assert (y2 - y1 == x2 - x1), "The hole cannot be fit into a square! (y2, y1, x2, x1) = ({}, {}, {}, {})".format(y2, y1, x2, x1)

    roi, roimask = img1[y1: y2, x1: x2], mask[y1: y2, x1: x2]
    # roimask = np.expand_dims(roimask, axis=2)
    # roimask = np.concatenate([roimask, roimask, roimask], axis=2)

    def cost(error1_map, error2_map, offset_y, offset_x, scale):
        # error1 = A2C_sum[offset_y, offset_x] + B2C_sum[0] - 2 * ABC_sum[offset_y, offset_x]
        # error1 = (((_roi - _img2[offset_y: offset_y + _roi.shape[0], offset_x: offset_x + _roi.shape[1]]) * roimask) ** 2).sum()
        error1 = error1_map[offset_y, offset_x]
        # TODO: Beffer function f(x) from (0, inf) to (1, inf) with low f'(x)
        error1 *= ((offset_y / scale - y1) ** 2 +
                   (offset_x / scale - x1) ** 2) + 1
        error2 = error2_map[offset_y, offset_x]
        return error1 + error2

    best_cost = np.inf
    best_params = None

    # TODO: Time complexity
    _roi = get_lab(roi)
    grad1 = cv2.Sobel(roi, -1, 1, 1)
    if show:
        info("Applying median filter", domain=__file__)
    text1 = get_texture(grad1)
    # print("Cost num = {}".format(3 * (img2.shape[0] - roi.shape[0]) * (img2.shape[1] - roi.shape[1])))

    B, C = roi, roimask
    B2 = (B ** 2).sum(axis=2)
    C_3d = np.expand_dims(C, axis=2)
    C_3d = np.concatenate((C_3d, C_3d, C_3d), axis=2)
    BC = B * C_3d
    B2C = B2 * C
    kern = np.ones_like(C)
    # embed()

    if show:
        info("Convolving B2C", domain=__file__)
    B2C_sum = convolve2d(B2C, kern, mode='valid')
    
    D = text1
    D2 = (D ** 2).sum(axis=2)

    for i in range(3):
        info("Scale {}".format(i))
        _img2 = get_lab(imgs[i])
        A = _img2
        A2 = (A ** 2).sum(axis=2)
        if show:
            info("Convolving A2C", domain=__file__)
        A2C_sum = convolve2d(A2, C[::-1, ::-1], mode='valid')
        if show:
            info("Convolving ABC", domain=__file__)
        ABC_sum = sum(
            [convolve2d(A[:, :, i], BC[::-1, ::-1, i], mode='valid') for i in range(3)])
        error1_map = A2C_sum + B2C_sum[0] - 2 * ABC_sum

        grad2 = cv2.Sobel(imgs[i], -1, 1, 1)
        text2 = get_texture(grad2)

        E = text2
        E2 = (E ** 2).sum(axis=2)

        # kern is the same size as roi, B and D
        D2_sum = convolve2d(D2, kern, mode='valid')
        E2_sum = convolve2d(E2, kern, mode='valid')
        if show:
            info("Convolving DE. i = {}".format(i), domain=__file__)
        DE_sum = sum(
            [convolve2d(E[:, :, i], D[::-1, ::-1, i], mode='valid') for i in range(3)])
        error2_map = D2_sum + E2_sum - 2 * DE_sum

        for y in range(_img2.shape[0] - roi.shape[0]):
            for x in range(_img2.shape[1] - roi.shape[1]):
                # print("Scale {}, y = {}, x = {}".format(i, y, x))
                c = cost(error1_map, error2_map, y, x, scales[i])
                if c < best_cost:
                    best_cost = c
                    best_params = (best_cost, scales[i], y, x)

    return best_params


def matchall_worker(img1, imgs, mask, show_tqdm=False):
    part_result = []
    for cand in (imgs if not show_tqdm else tqdm(imgs)):
        part_result.append(match(img1, cand, mask, show_tqdm))
    return part_result


def matchall(img1, imgs, mask):
    process_num = 1
    unit = len(imgs) // process_num + 1
    pool = multiprocessing.Pool(processes=process_num)
    result_process = []
    results = []

    info("Start matching processes", domain=__file__)
    for i in range(1, process_num):
        result_process.append(pool.apply_async(matchall_worker, args=(
            img1, imgs[i * unit: min((i + 1) * unit, len(imgs))], mask)))

    results.append(matchall_worker(img1, imgs[0: unit], mask, show_tqdm=True))

    info("Joining matching processes", domain=__file__)
    for i in range(process_num - 1):
        results.append(result_process[i].get())

    pool.close()
    pool.join()
    return reduce(lambda x, y: x + y, results)
