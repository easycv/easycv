from easycv.transforms.base import Transform

from easycv.validators import Number, Type
from skimage.util import random_noise


class Noise(Transform):
    """
        Noise is a transform that adds various types of noise to the image. Currently supported\
        types are:

        \t**∙ gaussian** - Gaussian-distributed additive noise\n
        \t**∙ poisson** - Poisson-distributed noise generated from the data\n
        \t**∙ salt** - Replaces random pixels with 255\n
        \t**∙ pepper** - Replaces random pixels with 0\n
        \t**∙ s&p** - Replaces random pixels with either 1 or 0.\n
        \t**∙ speckle** - Multiplicative noise using ``out = image + n * image``, where \
        n is uniform noise with specified mean & variance.\n

        :param method: Type of noise to add
        :type method: :class:`str`, optional
        :param seed: Seed for the random generator, by default generates random seed
        :type seed: :class:`int`, optional
        :param clip: If True the output will be clipped to [0, 255], defaults to True
        :type clip: :class:`bool`, optional
        :param mean: Mean of random distribution. Used in ‘gaussian’ and ‘speckle’, defaults to 0
        :type mean: :class:`float`, optional
        :param var: Variance of random distribution. Used in ‘gaussian’ and ‘speckle’, \
        defaults to 0.01
        :type var: :class:`float`, optional
        :param amount: Percentage of image pixels to replace with noise.Used in ‘salt’, \
        ‘pepper’, and ‘s&p’. Default : 0.05
        :type amount: :class:`float`, optional
        :param salt_vs_pepper: Proportion of salt vs. pepper noise for ‘s&p’ on range [0, 1].\
         Higher values represent more salt. Default : 0.5 (equal amounts)
        :type salt_vs_pepper: :class:`float`, optional
    """

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
        "var": Number(min_value=0, max_value=250, default=2.5),
        "amount": Number(min_value=0, max_value=1, default=0.05),
        "salt_vs_pepper": Number(min_value=0, max_value=1, default=0.5),
    }

    def apply(self, image, **kwargs):
        kwargs["mode"] = kwargs.pop("method")
        kwargs["seed"] = kwargs["seed"] if kwargs["seed"] else None
        if kwargs["mode"] == "gaussian":
            kwargs["var"] = kwargs["var"] / 255
        return random_noise(image, **kwargs) * 255
