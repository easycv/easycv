from scipy.ndimage import correlate1d
import numpy as np

from cv.transforms.base import Transform


class Correlate1d(Transform):
    default_args = {'kernel': None, 'mode': 'reflect', 'axis': 0}

    def apply(self, image, **kwargs):
        image = correlate1d(image, kwargs['kernel'], axis=kwargs['axis'], mode=kwargs['mode'])
        return image


class Correlate2d(Transform):
    default_args = {'kernel': None, 'mode': 'reflect'}

    def apply(self, image, **kwargs):
        image = Correlate1d(kernel=kwargs['kernel'][0], axis=0, mode=kwargs['mode']).process(image)
        image = Correlate1d(kernel=kwargs['kernel'][1], axis=1, mode=kwargs['mode']).process(image)
        return image


class Convolve1d(Transform):
    default_args = {'kernel': None, 'mode': 'reflect', 'axis': 0}

    def apply(self, image, **kwargs):
        kwargs['kernel'] = kwargs['kernel'][::-1]
        return Correlate1d(**kwargs).process(image)


class Convolve2d(Transform):
    default_args = {'kernel': None, 'mode': 'reflect'}

    def apply(self, image, **kwargs):
        print(kwargs['kernel'])
        kwargs['kernel'] = kwargs['kernel'][:][::-1]
        return Correlate2d(**kwargs).process(image)