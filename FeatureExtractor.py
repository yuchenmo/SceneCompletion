import numpy as np
from scipy.signal import medfilt
import cv2
import os.path as op
from PIL import Image
from skimage import transform

"""
The image had better be squared... And width and height can both be divided by 4 
"""

import gist
def get_gist_C_implementation(img, mask=None):
    """
    Extract GIST descriptor of an image. Implemented by C.
    This implementation cannot change orientation and filter num, but it's really faster than MATLAB.
    """
    _img = transform.resize(img, (1024, 1024), preserve_range=True).astype('uint8')
    _mask = transform.resize(img, (1024, 1024), preserve_range=True).astype('uint8')
    if mask is None:
        return gist.extract(_img)

    _img[_mask > 0] = 0

    # 1440 = 3 * (6 * 5) * 4 * 4 for colored imgs
    descriptor = gist.extract(_img).reshape((3, 30, 4, 4))
    weight = np.zeros((4, 4)).astype('float32')
    unity, unitx = mask.shape[0] // 4, mask.shape[1] // 4
    for _y in range(4):
        for _x in range(4):
            weight[_y, _x] = np.sum(mask[_y * unity: (_y + 1) * unity, _x * unitx: (_x + 1) * unitx] > 0) / (unity * unitx)

    for c in range(3):
        for i in range(30):
            descriptor[c, i] *= weight

    return descriptor.reshape((-1, ))

import matlab.engine
def get_gist_MATLAB_implementation(img, mask=None):
    """
    Antonio's original code. To switch different orientation and filter num.
    GIST feature order: (c, w, h)
    Attention: This mask is the original mask, not the mask of the border area 
    """
    _img = img.copy()

    if mask is not None:
        _mask = np.expand_dims(mask, axis=2)
        _mask = np.concatenate((_mask, _mask, _mask), axis=2)
        _img[_mask > 0] = 0
    
    cv2.imwrite("./gist_Antonio/input.png", _img)
    eng = matlab.engine.start_matlab()
    eng.cd("{}/gist_Antonio".format(op.abspath('.')))
    eng.run("getGIST", nargout=0)

    if mask is None:
        return np.array(eng.workspace['gist'])
    # 480 = (6 * 5) * 4 * 4
    descriptor = np.array(eng.workspace['gist']).reshape((30, 4, 4))
    weight = np.zeros((4, 4)).astype('float32')
    unity, unitx = mask.shape[0] // 4, mask.shape[1] // 4
    for _y in range(4):
        for _x in range(4):
            weight[_y, _x] = np.sum(mask[_y * unity: (_y + 1) * unity, _x * unitx: (_x + 1) * unitx] > 0) / (unity * unitx)

    for i in range(30):
        descriptor[i] *= weight

    return descriptor.reshape((-1, ))

def get_texture(img):
    """
    Median filter
    """
    return medfilt(img, np.array([5, 5, 1]))

def get_lab(img):
    """
    BGR2L*a*b*
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

if __name__ == "__main__":
    img = cv2.imread("./dataset/raw_image/test.bmp")
    mask = cv2.imread("./dataset/raw_image/testmask.bmp")[:, :, 0]
    mask[mask > 0] = 1
    print(get_gist_MATLAB_implementation(img, mask).shape)