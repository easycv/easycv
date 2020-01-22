import numpy as np


def salt_and_pepper(image, prob=0.05):
    probabilities = np.random.rand(image.shape[0], image.shape[1])
    image[probabilities < prob] = 0
    image[probabilities > 1-prob] = 255
    return image


def impulse_noise(image, prob=0.05):
    image = np.copy(image)
    probabilities = np.random.rand(image.shape[0], image.shape[1])
    image[probabilities < prob] = 0
    return image
