import numpy as np

from cv.transforms.base import Transform
from cv.errors.transforms import InvalidArgumentError


class GrayScale(Transform):
    arguments = {'method': 'luma'}


    def apply(self, image):
        if self.arguments['method'] == 'averaging':
            return np.average(image, axis=2).astype(np.uint8)
        elif self.arguments['method'] == 'luma':
            return np.average(image, weights=[0.299, 0.587, 0.114], axis=2).astype(np.uint8)
        elif self.arguments['method'] == 'desaturation':
            return ((image.max(axis=2) + image.min(axis=2)) / 2).astype(np.uint8)
        elif self.arguments['method'] == 'decomposition_max':
            return image.max(axis=2).astype(np.uint8)
        elif self.arguments['method'] == 'decomposition_min':
            return image.min(axis=2).astype(np.uint8)


class FilterChannels(Transform):
    arguments = {'channels': []}

    def apply(self, image):
        channels = np.array(self.arguments['channels'])
        if any(channels > 2) or any(channels < 0):
            raise InvalidArgumentError(('channels',))
        image[:, :, channels] = 0
        return image
