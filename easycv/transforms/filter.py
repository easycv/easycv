import cv2
from skimage.filters import unsharp_mask

from easycv.transforms.base import Transform
from easycv.transforms.color import GrayScale
import easycv.transforms.edges
from easycv.validators import Number, Type


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

    methods = {
        "uniform": {"arguments": ["size"]},
        "gaussian": {"arguments": ["size", "sigma", "truncate"]},
        "median": {"arguments": ["size"]},
        "bilateral": {"arguments": ["size", "sigma_color", "sigma_space"]},
    }
    default_method = "gaussian"

    arguments = {
        "size": Number(min_value=1, only_integer=True, only_odd=True, default="auto"),
        "sigma": Number(min_value=0, default=0),
        "sigma_color": Number(min_value=0, default=75),
        "sigma_space": Number(min_value=0, default=75),
        "truncate": Number(min_value=0, default=4),
    }

    def process(self, image, **kwargs):
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
                image, kwargs["size"], kwargs["sigma_color"], kwargs["sigma_space"]
            )


class Sharpness(Transform):
    """
    Sharpness is a transform that measures how sharpen an image is. The sharpness metric is \
    calculated using the laplacian of the image. Images are classified as sharpen when above a \
    certain value of sharpness given by the threshold.

    :param threshold: Threshold to classify images as sharpen, defaults to 100
    :type threshold: :class:`int`/:class:`float`, optional
    """

    arguments = {"threshold": Number(min_value=0, default=100)}

    outputs = {"sharpness": Number(), "sharpen": Type(bool)}

    def process(self, image, **kwargs):
        grayscale = GrayScale().apply(image)
        variance = (
            easycv.transforms.edges.Gradient(method="laplace").apply(grayscale).var()
        )
        sharpen = variance >= kwargs["threshold"]
        return {"sharpness": variance, "sharpen": sharpen}


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

    arguments = {
        "sigma": Number(min_value=0, default=1),
        "amount": Number(default=1),
        "multichannel": Type(bool, default=False),
    }

    def process(self, image, **kwargs):
        kwargs["radius"] = kwargs.pop("sigma")
        return unsharp_mask(image, preserve_range=True, **kwargs)
