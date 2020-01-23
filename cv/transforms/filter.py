from scipy import signal
import numpy as np

from cv.transforms.base import Transform


class Correlate(Transform):
    arguments = {'kernel': None, 'mode': 'full'}

    def apply(self, image):
        image = signal.correlate(image, self.arguments['kernel'], mode=self.arguments['mode'], method='fft')
        return image


class Convolve(Transform):
    arguments = {'kernel': None, 'mode': 'full'}

    def apply(self, image):
        image = signal.convolve(image, self.arguments['kernel'], mode=self.arguments['mode'], method='fft')
        return image
