from cv.transforms.base import Transform

from cv.validators import Option, Number, Type
from skimage.util import random_noise


class Noise(Transform):
    default_args = {
        "method": Option(
            ["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"],
            default=0,
        ),
        "seed": Number(min_value=0, max_value=2 ** 32 - 1),
        "clip": Type(bool, default=True),
        "mean": Type(float, default=0),
        "var": Type(float, default=0.01),
    }

    def apply(self, image, **kwargs):
        if kwargs["method"] == "gaussian":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"],
                clip=kwargs["clip"],
                mean=kwargs["mean"],
                var=kwargs["var"],
            )
        elif kwargs["method"] == "localvar":
            return random_noise(image, mode=kwargs["method"], seed=kwargs["seed"])
        elif kwargs["method"] == "poisson":
            return random_noise(
                image, mode=kwargs["method"], seed=kwargs["seed"], clip=kwargs["clip"]
            )
        elif kwargs["method"] == "salt":
            return random_noise(image, mode=kwargs["method"], seed=kwargs["seed"])
        elif kwargs["method"] == "pepper":
            return random_noise(image, mode=kwargs["method"], seed=kwargs["seed"])
        elif kwargs["method"] == "s&p":
            return random_noise(image, mode=kwargs["method"], seed=kwargs["seed"])
        elif kwargs["method"] == "speckle":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"],
                clip=kwargs["clip"],
                mean=kwargs["mean"],
                var=kwargs["var"],
            )
