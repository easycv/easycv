import cv2
import numpy as np

from cv.validators import Option, List, Number
from cv.transforms.base import Transform


class GrayScale(Transform):
    def apply(self, image, **kwargs):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class FilterChannels(Transform):
    default_args = {
        'channels': List(Number(min_value=0, max_value=2, only_integer=True))
    }

    def apply(self, image, **kwargs):
        channels = np.array(kwargs['channels'])
        image[:, :, channels] = 0
        return image
