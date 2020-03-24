import cv2
import numpy as np

from cv.validators import Option, Number
from cv.transforms.base import Transform
from cv.transforms.color import GrayScale


class Gradient(Transform):
    """
    Gradient is a transform that computes the gradient of an image

    :param axis: Axis to do gradient, defaults to "x"
    :type axis: :class:`str`, optional
    :param method: Method to use on gradients, defaults to "sobel"
    :type method: :class:`str`, optional
    :param size: Type of image abstraction, defaults to 5
    :type size: :class:`int`, optional
    """

    default_args = {
        "axis": Option(["x", "y"], default=0),
        "method": Option(["sobel", "laplace"], default=0),
        "size": Number(
            min_value=1, max_value=31, only_integer=True, only_odd=True, default=5
        ),
    }

    def apply(self, image, **kwargs):
        image = GrayScale().process(image)
        if kwargs["method"] == "sobel":
            if kwargs["axis"] == "x":
                return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs["size"])
            else:
                return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs["size"])
        else:
            return cv2.Laplacian(image, cv2.CV_64F)


class GradientMagnitude(Transform):
    """
    GradientMagnitude is a transform that computes the gradient magnitude of an image

    :param size: Size of operator, defaults to 5
    :type size: :class:`int`, optional
    """

    default_args = {
        "size": Number(
            min_value=1, max_value=31, only_integer=True, only_odd=True, default=5
        )
    }

    def apply(self, image, **kwargs):
        image = GrayScale().process(image)
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs["size"])
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs["size"])
        return (x ** 2 + y ** 2) ** 0.5


class GradientAngle(Transform):
    """
    GradientAngle is a transform that computes the angles of the gradients of an image

    :param size: Size of operator, defaults to 5
    :type size: :class:`int`, optional
    """

    default_args = {
        "size": Number(
            min_value=1, max_value=31, only_integer=True, only_odd=True, default=5
        )
    }

    def apply(self, image, **kwargs):
        image = GrayScale().process(image)
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs["size"])
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs["size"])
        return np.arctan2(x, y)


class Canny(Transform):
    """
    Canny is a transform that applies canny edge detection on an image

    :param low: Low threshold, defaults to 100
    :type low: :class:`int`, optional
    :param high: High threshold, defaults to 200
    :type high: :class:`int`, optional
    """

    default_args = {
        "low": Number(min_value=1, max_value=255, only_integer=True, default=100),
        "high": Number(min_value=1, max_value=255, only_integer=True, default=200),
    }

    def apply(self, image, **kwargs):
        return cv2.Canny(image, kwargs["low"], kwargs["high"])
