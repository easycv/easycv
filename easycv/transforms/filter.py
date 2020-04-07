import cv2
from skimage.filters import unsharp_mask

from easycv.transforms.base import Transform
from easycv.validators import Option, Number, Type


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
        "method": Option(["uniform", "gaussian", "median", "bilateral"], default=1),
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
                kwargs["size"] = int(kwargs["sigma"] * 8 + 1)
            return cv2.GaussianBlur(
                image, (kwargs["size"], kwargs["size"]), kwargs["sigma"]
            )
        elif kwargs["method"] == "median":
            return cv2.medianBlur(image, kwargs["size"])
        else:
            return cv2.bilateralFilter(
                image, kwargs["size"], kwargs["sigma_color"], kwargs["sigma_space"]
            )


class Sharpen(Transform):
    """
    Sharpen is a transform that sharpens an image.
    enhanced image = original + amount * (original - blurred)

    :param radius: Blur method to be used, defaults to "uniform"
    :type radius: :class:`str`, optional
    :param amount: Kernel size, defaults to auto
    :type amount: :class:`int`, optional
    :param multichannel: True
    :type multichannel: :class:`int`, optional
    """

    default_args = {
        "radius": Number(min_value=0, only_integer=True, default=1),
        "sigma": Number(min_value=0, default=0),
        "amount": Number(default=1),
        "multichannel": Type(bool, default=True),
    }

    def apply(self, image, **kwargs):
        return unsharp_mask(image, **kwargs, preserve_range=True)
