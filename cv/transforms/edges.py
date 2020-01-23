import numpy as np

from cv.transforms.base import Transform
from cv.transforms.filter import Correlate
from cv.transforms.kernels import smooth_gradient_kernel


def normalize(x):
    return (x-x.min())/(x.max()-x.min())


class GradientMagnitude(Transform):
    arguments = {'size': 3, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image):
        gradient = smooth_gradient_kernel(size=self.arguments['size'],
                                          operator=self.arguments['operator'],
                                          sigma=self.arguments['sigma'])
        x = Correlate(kernel=gradient[0]).apply(image)
        y = Correlate(kernel=gradient[1]).apply(image)

        gradient = (x**2 + y**2)**0.5

        return normalize(gradient)*255


class GradientDirection(Transform):
    arguments = {'size': 3, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image):
        gradient = smooth_gradient_kernel(size=self.arguments['size'],
                                          operator=self.arguments['operator'],
                                          sigma=self.arguments['sigma'])
        x = Correlate(kernel=gradient[0]).apply(image)
        y = Correlate(kernel=gradient[1]).apply(image)

        gradient_direction = np.arctan(np.true_divide(y, x))

        return gradient_direction

