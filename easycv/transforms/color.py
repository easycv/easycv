import cv2
import numpy as np

from easycv.validators import Option, List, Number
from easycv.transforms.base import Transform


class GrayScale(Transform):
    """
    GrayScale is a transform that turns an image into grayscale.
    """

    def apply(self, image, **kwargs):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image


class FilterChannels(Transform):
    """
    FilterChannels is a transform that removes color channel(s).

    :param channels: List of channels to remove
    :type channels: :class:`list`
    :param scheme: Image color scheme (rgb or bgr), defaults to "rgb"
    :type scheme: :class:`str`, optional
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
