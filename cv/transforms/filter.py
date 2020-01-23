from scipy import signal,ndimage
import numpy as np

from cv.transforms.base import Transform



def correlate(image, kernel, method='constant'):
    image = ndimage.correlate(image, kernel, mode=method)
    return image.astype(np.uint8)


class Convolve(Transform):
    arguments = {'image': None, 'kernel': None, 'method': 'same'}

    def apply(self, image):
        image = signal.fftconvolve(image, self.arguments['kernel'][:, :, np.newaxis], mode=self.arguments['method'])
        return image
