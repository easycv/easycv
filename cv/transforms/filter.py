from scipy import signal,ndimage
import numpy as np


def convolve(image, kernel=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), method='same'):
    image = signal.fftconvolve(image, kernel[:, :, np.newaxis], mode=method)
    return image.astype(np.uint8)


def correlate(image, kernel, method='constant'):
    image = ndimage.correlate(image, kernel, mode=method)
    return image.astype(np.uint8)


