import cv2
from skimage.filters import unsharp_mask

from easycv.transforms.base import Transform
from easycv.validators import Method, Number, Type


class Blur(Transform):
    """
    Blur is a transform that blurs an image.

    \t**∙ uniform** - Uniform Filter\n
    \t**∙ gaussian** - Gaussian-distributed additive noise\n
    \t**∙ median** - Median Filter\n
    \t**∙ bilateral** - Edge preserving blur\n

    :param method: Blur method to be used, defaults to "uniform"
    :type method: :class:`str`, optional
    :param size: Kernel size, defaults to auto
    :type size: :class:`int`, optional
    :param sigma: Sigma value, defaults to 0
    :type sigma: :class:`int`, optional
    :param sigma_color: Sigma for color space, defaults to 75
    :type sigma_color: :class:`int`, optional
    :param sigma_space: Sigma for coordinate space, defaults to 75
    :type sigma_space: :class:`int`, optional
    :param truncate: Truncate the filter at this many standard deviations., defaults to 4
    :type truncate: :class:`int`, optional
    """

    default_args = {
        "method": Method(
            {
                "uniform": [],
                "gaussian": ["sigma", "truncate"],
                "median": [],
                "bilateral": ["sigma_color", "sigma_space"],
            },
            default="gaussian",
        ),
        "size": Number(min_value=1, only_integer=True, only_odd=True, default="auto"),
        "sigma": Number(min_value=0, default=0),
        "sigma_color": Number(min_value=0, default=75),
        "sigma_space": Number(min_value=0, default=75),
        "truncate": Number(min_value=0, default=4),
    }

    def apply(self, image, **kwargs):
        if kwargs["method"] == "uniform":
            return cv2.blur(image, (kwargs["size"], kwargs["size"]))
        elif kwargs["method"] == "gaussian":
            if kwargs["size"] == "auto":
                kwargs["size"] = 2 * int(kwargs["sigma"] * kwargs["truncate"] + 0.5) + 1
            return cv2.GaussianBlur(
                image, (kwargs["size"], kwargs["size"]), kwargs["sigma"]
            )
        elif kwargs["method"] == "median":
            return cv2.medianBlur(image, kwargs["size"])
        else:
            if kwargs["size"] == "auto":
                kwargs["size"] = 5
            return cv2.bilateralFilter(
                image, kwargs["size"], kwargs["sigma_color"], kwargs["sigma_ssspace"]
            )


class Sharpen(Transform):
    """
    Sharpen is a transform that sharpens an image.

    :param sigma: Kernel sigma, defaults to 1
    :type sigma: :class:`float`, optional
    :param amount: Amount to sharpen, defaults to 1
    :type amount: :class:`float`, optional
    :param multichannel: `True` if diferent processing for each color layer `False` otherwise
    :type multichannel: :class:`bool`
    """

    default_args = {
        "sigma": Number(min_value=0, default=1),
        "amount": Number(default=1),
        "multichannel": Type(bool, default=False),
    }

    def apply(self, image, **kwargs):
        kwargs["radius"] = kwargs.pop("sigma")
        return unsharp_mask(image, preserve_range=True, **kwargs)
