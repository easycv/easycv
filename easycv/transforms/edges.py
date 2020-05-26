import cv2
import numpy as np

from easycv.validators import Number, Option
from easycv.transforms.base import Transform
from easycv.transforms.color import GrayScale


class Gradient(Transform):
    """
    Gradient is a transform that computes the gradient of an image.

    :param axis: Axis to compute, defaults to "both" (magnitude)
    :type axis: :class:`str`, optional
    :param method: Gradient calculation method, defaults to "sobel"
    :type method: :class:`str`, optional
    :param size: Kernel size, defaults to 5
    :type size: :class:`int`, optional
    """

    methods = {"sobel": {"arguments": ["axis", "size"]}, "laplace": {}}
    default_method = "sobel"

    arguments = {
        "axis": Option(["both", "x", "y"], default=0),
        "size": Number(
            min_value=1, max_value=31, only_integer=True, only_odd=True, default=5
        ),
    }

    def process(self, image, **kwargs):
        image = GrayScale().apply(image)
        if kwargs["method"] == "sobel":
            if kwargs["axis"] == "both":
                x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs["size"])
                y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs["size"])
                return (x ** 2 + y ** 2) ** 0.5
            if kwargs["axis"] == "x":
                return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs["size"])
            else:
                return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs["size"])
        else:
            return cv2.Laplacian(image, cv2.CV_64F)


class GradientAngle(Transform):
    """
    GradientAngle is a transform that computes the angles of the image gradient.

    :param size: Kernel size, defaults to 5
    :type size: :class:`int`, optional
    """

    arguments = {
        "size": Number(
            min_value=1, max_value=31, only_integer=True, only_odd=True, default=5
        )
    }

    def process(self, image, **kwargs):
        image = GrayScale().apply(image)
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs["size"])
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs["size"])
        return np.arctan2(x, y)


class Canny(Transform):
    """
    Canny is a transform that extracts the edges from the image using canny edge detection.

    :param low: Low threshold, defaults to 100
    :type low: :class:`int`, optional
    :param high: High threshold, defaults to 200
    :type high: :class:`int`, optional
    :param size: Kernel size, defaults to 5
    :type size: :class:`int`, optional
    """

    arguments = {
        "low": Number(min_value=1, max_value=255, only_integer=True, default=100),
        "high": Number(min_value=1, max_value=255, only_integer=True, default=200),
        "size": Number(
            min_value=3, max_value=7, only_integer=True, only_odd=True, default=3
        ),
    }

    def process(self, image, **kwargs):
        return cv2.Canny(image, kwargs["low"], kwargs["high"], kwargs["size"])
