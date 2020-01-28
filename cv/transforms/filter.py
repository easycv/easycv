from scipy.ndimage import correlate1d
import numpy as np

from cv.transforms.base import Transform


class Correlate1d(Transform):
    default_args = {'kernel': None, 'mode': 'reflect', 'axis': 0}

    def apply(self, image, **kwargs):
        image = correlate1d(image, kwargs['kernel'], output=np.uint16, axis=kwargs['axis'], mode=kwargs['mode'])
        return image


class Convolve1d(Transform):
    default_args = {'kernel': None, 'mode': 'reflect', 'axis': 0}

    def apply(self, image, **kwargs):
        kwargs['kernel'] = kwargs['kernel'][::-1]
        return Correlate1d(**kwargs).process(image)
