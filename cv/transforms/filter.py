from scipy import signal
import numpy as np

from cv.transforms.base import Transform


class Correlate(Transform):
    default_args = {'kernel': None, 'mode': 'full'}

    def apply(self, image, **kwargs):
        image = signal.correlate(image, kwargs['kernel'], mode=kwargs['mode'], method='fft')
        return image


class Convolve(Transform):
    default_args = {'kernel': None, 'mode': 'full'}

    def apply(self, image, **kwargs):
        print(kwargs)
        image = signal.convolve(image, kwargs['kernel'], mode=kwargs['mode'], method='fft')
        return image
