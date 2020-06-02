import cv2
import numpy as np

from color_transfer import color_transfer
from easycv.validators import Option, List, Number, Image
from easycv.transforms.base import Transform


class GrayScale(Transform):
    """
    GrayScale is a transform that turns an image into grayscale.
    """

    def process(self, image, **kwargs):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image


class Sepia(Transform):
    """
    Sepia is a transform that applies the sepia effect to an image
    """

    def process(self, image, **kwargs):
        gray = GrayScale().process(image)
        sepia = np.array([153 / 255 * gray, 204 / 255 * gray, gray])
        return sepia.transpose(1, 2, 0).astype("uint8")


class FilterChannels(Transform):
    """
    FilterChannels is a transform that removes color channel(s).

    :param channels: List of channels to remove
    :type channels: :class:`list`
    :param scheme: Image color scheme (rgb or bgr), defaults to "rgb"
    :type scheme: :class:`str`, optional
    """

    arguments = {
        "channels": List(Number(min_value=0, max_value=2, only_integer=True)),
        "scheme": Option(["rgb", "bgr"], default=0),
    }

    def process(self, image, **kwargs):
        channels = np.array(kwargs["channels"])
        if kwargs["scheme"] == "rgb":
            channels = 2 - channels
        if len(channels) > 0:
            image[:, :, channels] = 0
        return image


class GammaCorrection(Transform):
    """
    GammaCorrection is a transform that corrects the contrast of images and displays.

    :param gamma: Gamma value
    :type gamma: :class:`Float`
    """

    arguments = {
        "gamma": Number(min_value=1e-30, default=1),
    }

    def process(self, image, **kwargs):
        table = np.array(
            [((i / 255.0) ** (1.0 / kwargs["gamma"])) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)


class Negative(Transform):
    """
    Negative is a transform that inverts color and brightness in an image.
    """

    def process(self, image, **kwargs):
        return 255 - image


class Cartoon(Transform):
    """
    Cartoon is a transform that creates a stylized / cartoonized image.

    :param smoothing: Determines the amount of smoothing, defaults to 60.
    :type smoothing: :class:`float`, optional
    :param region_size: Determines the size of regions of constant color, defaults to 0.45.
    :type region_size: :class:`float`, optional
    """

    arguments = {
        "smoothing": Number(min_value=0, max_value=200, default=60),
        "region_size": Number(min_value=0, max_value=1, default=0.45),
    }

    def process(self, image, **kwargs):
        return cv2.stylization(
            image, sigma_s=kwargs["smoothing"], sigma_r=kwargs["region_size"]
        )


class PhotoSketch(Transform):
    """
    PhotoSketch is a transform that creates a black and white pencil-like drawing.
    """

    def process(self, image, **kwargs):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
        img_blend = cv2.divide(img_gray, img_blur, scale=256)
        return img_blend


class ColorTransfer(Transform):
    """
    ColorTransfer is a transform that transfers the color of an image to another.

    :param source: Source image from where the colors will be transferred from.
    :type source: :class:`~easycv.image.Image`
    """

    arguments = {
        "source": Image(),
    }

    def process(self, image, **kwargs):
        return color_transfer(kwargs["source"].array, image)
