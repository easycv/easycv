import numpy as np


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
        noisy_image[noisy_image > 255] = 255
        noisy_image[noisy_image < 0] = 0
    elif method == 'normalize':
        noisy_image[:, :, 0] = 255*(noisy_image[:, :, 0] - noisy_image[:, :, 0].min()) / (noisy_image[:, :, 0].max() - noisy_image[:, :, 0].min())
        noisy_image[:, :, 1] = 255*(noisy_image[:, :, 1] - noisy_image[:, :, 1].min()) / (noisy_image[:, :, 1].max() - noisy_image[:, :, 1].min())
        noisy_image[:, :, 2] = 255*(noisy_image[:, :, 2] - noisy_image[:, :, 2].min()) / (noisy_image[:, :, 2].max() - noisy_image[:, :, 2].min())
    return noisy_image.astype(np.uint8)
