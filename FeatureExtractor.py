import numpy as np
from scipy.signal import medfilt
import cv2
from PIL import Image
import gist
from skimage import transform

def get_gist(img):
    """
    Extract GIST descriptor of an image.
    """
    _img = Image.fromarray(img)
    # TODO: Upsample to 4x4
    _img = transform.resize(img, (4, 4), preserve_range=True).astype('uint8')
    descriptor = gist.extract(_img)

    return descriptor

def get_texture(img):
    """
    Median filter
    """
    return medfilt(img, np.array([5, 5, 1]))

def get_lab(img):
    """
    BGR2L*a*b*
    """
    return cv2.cvtColor(img, cv2.CV_BGR2Lab)

if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    print(get_gist(img).shape)