import numpy as np

from cv.transforms.base import Transform
from cv.transforms.filter import Correlate2d
from cv.transforms.kernels import smooth_gradient_kernel


def normalize(x):
    return (x-x.min())/(x.max()-x.min())


class GradientMagnitude(Transform):
    default_args = {'size': 1, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image, **kwargs):
        gradient = smooth_gradient_kernel(size=kwargs['size'],
                                          operator=kwargs['operator'],
                                          sigma=kwargs['sigma'])
        x = Correlate2d(kernel=gradient).process(image)
        y = Correlate2d(kernel=gradient[::-1]).process(image)

        gradient = (x**2 + y**2)**0.5

        return normalize(gradient)*255


class GradientDirection(Transform):
    default_args = {'size': 1, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image, **kwargs):
        gradient = smooth_gradient_kernel(size=kwargs['size'],
                                          operator=kwargs['operator'],
                                          sigma=kwargs['sigma'])
        x = Correlate2d(kernel=gradient).process(image)
        y = Correlate2d(kernel=gradient[::-1]).process(image)

        gradient_direction = np.arctan(np.true_divide(y, x))

        return gradient_direction


class CannyEdge(Transform):
    default_args = {'size': 1, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image, **kwargs):
        gradient_magnitude = GradientMagnitude(size=kwargs['size'],
                                               operator=kwargs['operator'],
                                               sigma=kwargs['sigma']).process(image)
        gradient_direction = GradientDirection(size=kwargs['size'],
                                               operator=kwargs['operator'],
                                               sigma=kwargs['sigma']).process(image)
        image_row, image_col = gradient_magnitude.shape

        output = np.zeros(gradient_magnitude.shape)

        pi = 180

        for row in range(1, image_row - 1):
            for col in range(1, image_col - 1):
                direction = gradient_direction[row, col]

                if (0 <= direction < pi / 8) or (15 * pi / 8 <= direction <= 2 * pi):
                    before_pixel = gradient_magnitude[row, col - 1]
                    after_pixel = gradient_magnitude[row, col + 1]

                elif (pi / 8 <= direction < 3 * pi / 8) or (9 * pi / 8 <= direction < 11 * pi / 8):
                    before_pixel = gradient_magnitude[row + 1, col - 1]
                    after_pixel = gradient_magnitude[row - 1, col + 1]

                elif (3 * pi / 8 <= direction < 5 * pi / 8) or (11 * pi / 8 <= direction < 13 * pi / 8):
                    before_pixel = gradient_magnitude[row - 1, col]
                    after_pixel = gradient_magnitude[row + 1, col]

                else:
                    before_pixel = gradient_magnitude[row - 1, col - 1]
                    after_pixel = gradient_magnitude[row + 1, col + 1]

                if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                    output[row, col] = gradient_magnitude[row, col]
        return output



