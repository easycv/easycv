from cv.transforms.base import Transform

from cv.validators import Number, Type
from skimage.util import random_noise


class Noise(Transform):
    methods = {
        "gaussian": ["mean", "var"],
        "salt": ["amount"],
        "pepper": ["amount"],
        "s&p": ["amount", "salt_vs_pepper"],
        "poisson": [],
    }

    default_args = {
        "method": "gaussian",
        "seed": Number(min_value=0, max_value=2 ** 32 - 1, default=False),
        "clip": Type(bool, default=True),
        "mean": Type(object, default=0),
        "var": Number(min_value=0, max_value=1, default=0.01),
        "amount": Number(min_value=0, max_value=1, default=0.05),
        "salt_vs_pepper": Number(min_value=0, max_value=1, default=0.5),
    }

    def apply(self, image, **kwargs):
        kwargs["mode"] = kwargs.pop("method")
        kwargs["seed"] = kwargs["seed"] if kwargs["seed"] else None
        return random_noise(image, **kwargs)
