from cv.transforms.base import Transform

from cv.validators import Option, Number, Type
from skimage.util import random_noise


class Noise(Transform):
    default_args = {
        "method": Option(
            ["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"],
            default=0,
        ),
        "seed": Number(min_value=0, max_value=2 ** 32 - 1, default=False),
        "clip": Type(bool, default=True),
        "mean": Type(float, default=0),
        "var": Number(min_value=0, max_value=1, default=0.01),
        "amount": Number(min_value=0, max_value=1, default=0.05),
        "salt_vs_pepper": Number(min_value=0, max_value=1, default=0.5),
    }

    def apply(self, image, **kwargs):
        if kwargs["method"] == "gaussian":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"] if kwargs["seed"] else None,
                clip=kwargs["clip"],
                mean=kwargs["mean"],
                var=kwargs["var"],
            )
        elif kwargs["method"] == "localvar":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"] if kwargs["seed"] else None,
            )
        elif kwargs["method"] == "poisson":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"] if kwargs["seed"] else None,
                clip=kwargs["clip"],
            )
        elif kwargs["method"] == "salt":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"] if kwargs["seed"] else None,
                amount=kwargs["amount"],
            )
        elif kwargs["method"] == "pepper":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"] if kwargs["seed"] else None,
                amount=kwargs["amount"],
            )
        elif kwargs["method"] == "s&p":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"] if kwargs["seed"] else None,
                amount=kwargs["amount"],
                salt_vs_pepper=kwargs["salt_vs_pepper"],
            )
        elif kwargs["method"] == "speckle":
            return random_noise(
                image,
                mode=kwargs["method"],
                seed=kwargs["seed"] if kwargs["seed"] else None,
                clip=kwargs["clip"],
                mean=kwargs["mean"],
                var=kwargs["var"],
            )
