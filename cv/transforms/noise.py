import numpy as np

from cv.transforms.base import Transform
from cv.errors.transforms import InvalidMethodError


class SaltAndPepper(Transform):
    arguments = {'prob': 0.05}

    def apply(self, image):
        probabilities = np.random.rand(image.shape[0], image.shape[1])
        image[probabilities < self.arguments['prob']] = 0
        image[probabilities > 1 - self.arguments['prob']] = 255
        return image


class Impulse(Transform):
    arguments = {'prob': 0.05}

    def apply(self, image):
        probabilities = np.random.rand(image.shape[0], image.shape[1])
        image[probabilities < self.arguments['prob']] = 0
        return image


class Gaussian(Transform):
    arguments = {'mu': 0, 'sigma': 20, 'grayscale': False, 'method': 'clip'}

    def apply(self, image):
        noise_dim = 1 if self.arguments['grayscale'] else 3
        noise = np.random.normal(self.arguments['mu'], self.arguments['sigma'],
                                 (image.shape[0], image.shape[1], noise_dim))
        noisy_image = noise + image
        if self.arguments['method'] == 'clip':
            noisy_image[noisy_image < 0] = 0
            noisy_image[noisy_image > 255] = 255
        elif self.arguments['method'] == 'normalize':
            noisy_image = 255 * (noisy_image - noisy_image.min(axis=(0, 1))) / (
                        noisy_image.max(axis=(0, 1)) - noisy_image.min(axis=(0, 1)))
        else:
            raise InvalidMethodError(('clip', 'normalize'))
        return noisy_image.astype(np.uint8)
