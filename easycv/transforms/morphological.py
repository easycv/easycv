import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Number
from easycv.utils import morp_methods


class Erode(Transform):
    """
    Erode is a transform that erodes away the boundaries of objects on an image.

    :param size: Kernel size, defaults to 5
    :type size: :class:`int`, optional
    :param iterations: Number of iterations, defaults to 1
    :type iterations: :class:`int`, optional
    """

    arguments = {
        "size": Number(min_value=1, only_integer=True, only_odd=True, default=5),
        "iterations": Number(min_value=1, only_integer=True, default=1),
    }

    def process(self, image, **kwargs):
        kernel = np.ones((kwargs["size"], kwargs["size"]), np.uint8)
        return cv2.erode(image, kernel, iterations=kwargs["iterations"])


class Dilate(Transform):
    """
    Dilate is a transform that dilates the boundaries of objects on an image.

    :param size: Kernel size, defaults to 5
    :type size: :class:`int`, optional
    :param iterations: Number of iterations, defaults to 1
    :type iterations: :class:`int`, optional
    """

    arguments = {
        "size": Number(min_value=1, only_integer=True, only_odd=True, default=5),
        "iterations": Number(min_value=1, only_integer=True, default=1),
    }

    def process(self, image, **kwargs):
        kernel = np.ones((kwargs["size"], kwargs["size"]), np.uint8)
        return cv2.dilate(image, kernel, iterations=kwargs["iterations"])


class Morphology(Transform):
    """
    Morphology is a transform that applies different morphological operation to images. Available \
    operations are:

    \t**∙ opening** - Opening is an Erosion followed by a dilation\n
    \t**∙ closing** - Closing is the reverse of Opening\n
    \t**∙ tophat** - Difference between the image and it's opening\n
    \t**∙ blackhat** - Difference between the image and it's closing\n

    :param size: Kernel size, defaults to 5
    :type size: :class:`int`, optional
    :param iterations: Number of iterations, defaults to 1
    :type iterations: :class:`int`, optional
    """

    arguments = {
        "size": Number(min_value=1, only_integer=True, only_odd=True, default=5),
        "iterations": Number(min_value=1, only_integer=True, default=1),
    }

    methods = ["opening", "closing", "tophat", "blackhat"]
    default_method = "opening"

    def process(self, image, **kwargs):
        kernel = np.ones((kwargs["size"], kwargs["size"]), np.uint8)
        return cv2.morphologyEx(
            image,
            morp_methods[kwargs["method"]],
            kernel,
            iterations=kwargs["iterations"],
        )
