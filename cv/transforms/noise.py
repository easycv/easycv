import numpy as np

from cv.errors.transforms import InvalidMethodError


def salt_and_pepper(image, prob=0.05):
    probabilities = np.random.rand(image.shape[0], image.shape[1])
    image[probabilities < prob] = 0
    image[probabilities > 1-prob] = 255
    return image


def impulse(image, prob=0.05):
    probabilities = np.random.rand(image.shape[0], image.shape[1])
    image[probabilities < prob] = 0
    return image


def gaussian(image, mu=0, sigma=1, grayscale=False, method='clip'):
    noise_dim = 1 if grayscale else 3
    noise = np.random.normal(mu, sigma, (image.shape[0], image.shape[1], noise_dim))
    noisy_image = noise + image
    if method == 'clip':
        noisy_image[noisy_image < 0] = 0
        noisy_image[noisy_image > 255] = 255
    elif method == 'normalize':
        noisy_image = 255 * (noisy_image - noisy_image.min(axis=(0, 1))) / (noisy_image.max(axis=(0, 1)) - noisy_image.min(axis=(0, 1)))
    else:
        raise InvalidMethodError(('clip', 'normalize'))
    return noisy_image.astype(np.uint8)
