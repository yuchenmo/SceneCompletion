import numpy as np
from scipy.signal import convolve2d
from utils import info
from IPython import embed

"""
We want \sum(((A-B)C)^2) = \sum(A^2*C) + \sum(B^2*C) - 2\sum(A*B*C).
Here '*' means elemwise product.
"""

"""
A = np.ones((10, 10))
B = np.ones((3, 3)) * 5

kern = np.ones_like(B)

A2 = A ** 2
B2 = B ** 2
term1 = convolve2d(A2, kern, mode='valid')
term2 = convolve2d(B2, kern, mode='valid')
term3 = convolve2d(A, B[::-1, ::-1], mode='valid')

embed()
"""

import time
time_base = None
end = None

def tik():
    global time_base
    time_base = time.time()

def tok(name="(undefined)"):
    info("Time used in {}: {}".format(name, time.time() - time_base), domain=__file__)

kernsize = 35
A = np.ones((1024, 1024))
B = np.ones((kernsize, kernsize)) * 5

tik()
for i in range(1024 - kernsize):
    for j in range(1024 - kernsize):
        C = ((A[i: i + kernsize, j: j + kernsize] - B) ** 2).sum()
tok("Bruteforce method")
tik()
A2 = A ** 2
B2 = B ** 2
kern = np.ones_like(B)
term1 = convolve2d(A2, kern, mode='valid')
term2 = convolve2d(B2, kern, mode='valid')
term3 = convolve2d(A, B[::-1, ::-1], mode='valid')
result = term1 - 2 * term3 + term2[0]
tok("FFT method")
