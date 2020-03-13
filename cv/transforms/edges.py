import cv2
import numpy as np

from cv.validators import Option, Number
from cv.transforms.base import Transform
from cv.transforms.color import GrayScale


class Gradient(Transform):
    default_args = {
        'axis': Option(['x', 'y'], default=0),
        'method': Option(['sobel', 'laplace'], default=0),
        'size': Number(min_value=1, max_value=31, only_integer=True, only_odd=True, default=5)
    }

    def apply(self, image, **kwargs):
        image = GrayScale().process(image)
        if kwargs['method'] == 'sobel':
            if kwargs['axis'] == 'x':
                return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs['size'])
            else:
                return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs['size'])
        else:
            return cv2.Laplacian(image, cv2.CV_64F)


class GradientMagnitude(Transform):
    default_args = {
        'size': Number(min_value=1, max_value=31, only_integer=True, only_odd=True, default=5)
    }

    def apply(self, image, **kwargs):
        image = GrayScale().process(image)
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs['size'])
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs['size'])
        return (x**2+y**2)**0.5


class GradientAngle(Transform):
    default_args = {
        'size': Number(min_value=1, max_value=31, only_integer=True, only_odd=True, default=5)
    }

    def apply(self, image, **kwargs):
        image = GrayScale().process(image)
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs['size'])
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs['size'])
        return np.arctan2(x, y)


class Canny(Transform):
    default_args = {
        'low': Number(min_value=1, max_value=255, only_integer=True, default=100),
        'high': Number(min_value=1, max_value=255, only_integer=True, default=200),
    }

    def apply(self, image, **kwargs):
        return cv2.Canny(image, kwargs['low'], kwargs['high'])
