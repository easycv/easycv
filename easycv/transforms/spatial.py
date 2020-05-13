import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Number, List, Type, Regex
from easycv.utils import interpolation_methods


class Resize(Transform):
    """
    Resize is a transform that resizes an image to a given width, height or a multiplier for \
    either. Currently supported interpolation methods:

    \t**∙ auto** - Automatically detect the best method\n
    \t**∙ nearest** - Nearest-neighbor interpolation\n
    \t**∙ linear** - Bilinear interpolation\n
    \t**∙ area** - Pixel area relation interpolation\n
    \t**∙ cubic** - Bicubic interpolation (4x4 pixel neighborhood)\n
    \t**∙ lanczos4** - Lanczos interpolation (8x8 pixel neighborhood)\n

    :param width: Output image width or scalar of width
    :type width: :class:`int`
    :param height: Output image height or scalar of width
    :type height: :class:`int`
    :param method: Resize method, defaults to auto
    :type method: :class:`str`, optional
    """

    methods = {
        "auto": {"arguments": []},
        "nearest": {"arguments": []},
        "linear": {"arguments": []},
        "area": {"arguments": []},
        "cubic": {"arguments": []},
        "lanczos4": {"arguments": []},
    }
    default_method = "auto"
    regex_description = (
        "float with a x at the the end or an int with or without a x at the end "
    )
    arguments = {
        "width": Regex(
            r"^(([\d]+([x])?)|(([\d]*[.])?[\d]+x))$", description=regex_description
        ),
        "height": Regex(
            r"^(([\d]+([x])?)|(([\d]*[.])?[\d]+x))$", description=regex_description
        ),
    }

    def process(self, image, **kwargs):
        if kwargs["method"] == "auto":
            if (
                image.shape[1] * image.shape[0] < kwargs["width"] * kwargs["height"]
                or kwargs["fx"] * kwargs["fy"] > 1
            ):
                kwargs["method"] = "cubic"
            else:
                kwargs["method"] = "area"
        if kwargs["width"][-1] == "x":
            kwargs["width"] = int(image.shape[1] * float(kwargs["width"][:-1]))

        if kwargs["height"][-1] == "x":
            kwargs["height"] = int(image.shape[1] * float(kwargs["height"][:-1]))

        return cv2.resize(
            image,
            (kwargs["width"], kwargs["height"]),
            interpolation=interpolation_methods[kwargs["method"]],
        )


class Rotate(Transform):
    """
            Rotate is a transform that rotates an image by certain degrees arround the provided \
            center. It can also be scaled.

            :param degrees: Degrees to rotate
            :type degrees: :class:`float`
            :param scale: Scale factor, defaults to 1
            :type scale: :class:`float`
            :param center: Center of rotation, defaults to the image center
            :type center: :class:`list`/:class:`tuple`, optional
            :param original: If True the image will be rescaled in order to keep it inside the \
             original size, defaults to True
            :type original: :class:`bool`, optional
        """

    arguments = {
        "degrees": Number(),
        "scale": Number(default=1),
        "center": List(
            Number(min_value=0, only_integer=True), length=2, default="auto"
        ),
        "original": Type(bool, default=True),
    }

    def process(self, image, **kwargs):
        (h, w) = image.shape[:2]
        if kwargs["center"] == "auto" or kwargs["original"]:
            kwargs["center"] = (w / 2, h / 2)

        matrix = cv2.getRotationMatrix2D(
            kwargs["center"], -kwargs["degrees"], kwargs["scale"]
        )

        if kwargs["original"]:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])

            n_w = int((h * sin) + (w * cos))
            h = int((h * cos) + (w * sin))

            matrix[0, 2] += (n_w / 2) - kwargs["center"][0]
            matrix[1, 2] += (h / 2) - kwargs["center"][1]

            w = n_w

        return cv2.warpAffine(image, matrix, (w, h))


class Crop(Transform):
    """
        Crop is a transform that crops a rectangular portion of an image, if original is True then
        the image size will be kept.

        :param box: A 4-tuple defining the left, r  ight, upper, and lower pixel coordinate.
        :type box: :class:`list`/:class:`tuple`
        :param original: True to keep original image size, False to resize to cropped area
        :type original: :class:`bool`, optional
    """

    arguments = {
        "rectangle": List(
            List(Number(min_value=0, only_integer=True), length=2), length=2
        ),
        "original": Type(bool, default=False),
    }

    def process(self, image, **kwargs):
        lx, rx, ty, by = (
            kwargs["rectangle"][0][0],
            kwargs["rectangle"][1][0],
            kwargs["rectangle"][0][1],
            kwargs["rectangle"][1][1],
        )

        #  crops the image keeping the original size
        if kwargs["original"]:

            output = np.zeros_like(image, dtype=np.uint8)
            output[:, :, -1] = 0

            # copy image to output
            output[ty:by, lx:rx] = image[ty:by, lx:rx]

            cv2.addWeighted(image, 0, output, 1, 0, output)

            return output

        # crops and resizes the image to match the cropped area
        else:
            return image[ty:by, lx:rx]


class Translate(Transform):
    """
        Translate is a transform that translates the image according to a vector xy

        :param x: x value, defaults to 0
        :type x: :class:`int`, optional
        :param y: y value, defaults to 0
        :type y: :class:`int`, optional
    """

    arguments = {
        "x": Number(min_value=0, only_integer=True, default=0),
        "y": Number(min_value=0, only_integer=True, default=0),
    }

    def process(self, image, **kwargs):
        height, width = image.shape[:2]

        matrix = np.float32([[1, 0, kwargs["x"]], [0, 1, kwargs["y"]]])

        return cv2.warpAffine(image, matrix, (width, height))
