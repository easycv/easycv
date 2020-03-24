import cv2
import numpy as np

from cv.validators import Option, List, Number
from cv.transforms.base import Transform


class GrayScale(Transform):
    """
    This class represents an **GrayScale**.

    GrayScale is a transform that turns image into grayscale
    """

    def apply(self, image, **kwargs):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image


class FilterChannels(Transform):
    """
    This class represents an **FilterChannels**.

    FilterChannels is a transform that removes channel(s)

    :param channels: List of channels to remove
    :type channels: :class:`list`
    :param scheme: Type of image abstraction
    :type scheme: :class:`str`
    """

    default_args = {
        "channels": List(Number(min_value=0, max_value=2, only_integer=True)),
        "scheme": Option(["rgb", "bgr"], default=0),
    }

    def apply(self, image, **kwargs):
        channels = np.array(kwargs["channels"])
        if kwargs["scheme"] == "rgb":
            channels = 2 - channels
        if len(channels) > 0:
            image[:, :, channels] = 0
        return image
