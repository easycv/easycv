import numpy as np

from cv.validators import Option, List, Number
from cv.transforms.base import Transform


class GrayScale(Transform):
    default_args = {
        'method': Option(['luma', 'averaging', 'desaturation', 'decomposition_max', 'decomposition_min'], default=0)
    }

    def apply(self, image, **kwargs):
        if kwargs['method'] == 'averaging':
            return np.average(image, axis=2)
        elif kwargs['method'] == 'luma':
            return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)
        elif kwargs['method'] == 'desaturation':
            return (image.max(axis=2) + image.min(axis=2)) / 2
        elif kwargs['method'] == 'decomposition_max':
            return image.max(axis=2)
        elif kwargs['method'] == 'decomposition_min':
            return image.min(axis=2)


class FilterChannels(Transform):
    default_args = {
        'channels': List(Number(min_value=0, max_value=2, only_integer=True))
    }

    def apply(self, image, **kwargs):
        channels = np.array(kwargs['channels'])
        image[:, :, channels] = 0
        return image
