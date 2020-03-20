import numpy as np

from cv.transforms.base import Transform
from cv.errors.transforms import InvalidMethodError


class SaltAndPepper(Transform):
    default_args = {"prob": 0.05}

    def apply(self, image, **kwargs):
        probabilities = np.random.rand(image.shape[0], image.shape[1])
        image[probabilities < kwargs["prob"]] = 0
        image[probabilities > 1 - kwargs["prob"]] = 1
        return image


class Impulse(Transform):
    default_args = {"prob": 0.05}

    def apply(self, image, **kwargs):
        probabilities = np.random.rand(image.shape[0], image.shape[1])
        image[probabilities < kwargs["prob"]] = 0
        return image


class Gaussian(Transform):
    default_args = {"mu": 0, "sigma": 20, "grayscale": False, "method": "clip"}

    def apply(self, image, **kwargs):
        noise_dim = 1 if kwargs["grayscale"] else 3
        noise = np.random.normal(
            kwargs["mu"], kwargs["sigma"], (image.shape[0], image.shape[1], noise_dim)
        )
        noisy_image = noise + image
        if kwargs["method"] == "clip":
            noisy_image = np.clip(noisy_image, 0, 1)
        elif kwargs["method"] == "normalize":
            noisy_image = (noisy_image - noisy_image.min(axis=(0, 1))) / (
                noisy_image.max(axis=(0, 1)) - noisy_image.min(axis=(0, 1))
            )
        else:
            raise InvalidMethodError(("clip", "normalize"))
        return noisy_image
