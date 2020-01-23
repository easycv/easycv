from scipy import signal
import numpy as np

from cv.transforms.base import Transform


class Convolve(Transform):
    arguments = {'image': None, 'kernel': None, 'method': 'same'}

    def apply(self, image):
        image = signal.fftconvolve(image, self.arguments['kernel'][:, :, np.newaxis], mode=self.arguments['method'])
        return image
