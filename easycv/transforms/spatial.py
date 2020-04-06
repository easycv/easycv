import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Number, Option, List, Type
from easycv.utils import interpolation_methods


class Resize(Transform):
    """
    Resize is a transform that resizes an image to a given width and height. Currently supported \
    interpolation methods:

    \t**∙ auto** - Automatically detect the best method\n
    \t**∙ nearest** - Nearest-neighbor interpolation\n
    \t**∙ linear** - Bilinear interpolation\n
    \t**∙ area** - Pixel area relation interpolation\n
    \t**∙ cubic** - Bicubic interpolation (4x4 pixel neighborhood)\n
    \t**∙ lanczos4** - Lanczos interpolation (8x8 pixel neighborhood)\n

    :param width: Output image width
    :type width: :class:`int`
    :param height: Output image height
    :type height: :class:`int`
    :param method: Interpolation method, defaults to auto
    :type method: :class:`str`, optional
    """

    default_args = {
        "width": Number(min_value=0, only_integer=True),
        "height": Number(min_value=0, only_integer=True),
        "method": Option(
            ["auto", "nearest", "linear", "area", "cubic", "lanczos4"], default=0
        ),
    }

    def apply(self, image, **kwargs):
        if kwargs["method"] == "auto":
            if image.shape[1] * image.shape[0] < kwargs["width"] * kwargs["height"]:
                kwargs["method"] = "cubic"
            else:
                kwargs["method"] = "area"

        return cv2.resize(
            image,
            (kwargs["width"], kwargs["height"]),
            interpolation=interpolation_methods[kwargs["method"]],
        )


class Rescale(Transform):
    """
        Rescale is a transform that rescales an image by a scale factor for x and y. Currently \
        supported interpolation methods:

        \t**∙ auto** - Automatically detect the best method\n
        \t**∙ nearest** - Nearest-neighbor interpolation\n
        \t**∙ linear** - Bilinear interpolation\n
        \t**∙ area** - Pixel area relation interpolation\n
        \t**∙ cubic** - Bicubic interpolation (4x4 pixel neighborhood)\n
        \t**∙ lanczos4** - Lanczos interpolation (8x8 pixel neighborhood)\n

        :param fx: Scale factor along the horizontal axis
        :type fx: :class:`float`
        :param fy: Scale factor along the vertical axis
        :type fy: :class:`float`
        :param method: Interpolation method, defaults to auto
        :type method: :class:`str`, optional
    """

    default_args = {
        "fx": Number(min_value=0),
        "fy": Number(min_value=0),
        "method": Option(
            ["auto", "nearest", "linear", "area", "cubic", "lanczos4"], default=0
        ),
    }

    def apply(self, image, **kwargs):
        if kwargs["method"] == "auto":
            if kwargs["fx"] * kwargs["fy"] > 1:
                kwargs["method"] = "cubic"
            else:
                kwargs["method"] = "area"

        return cv2.resize(
            image,
            (0, 0),
            fx=kwargs["fx"],
            fy=kwargs["fy"],
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

    default_args = {
        "degrees": Number(),
        "scale": Number(default=1),
        "center": List(
            Number(min_value=0, only_integer=True), length=2, default="auto"
        ),
        "original": Type(bool, default=True),
    }

    def apply(self, image, **kwargs):
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

        :param box: A 4-tuple defining the left, right, upper, and lower pixel coordinate.
        :type box: :class:`list`/:class:`tuple`
        :param original: True to keep original image size, False to resize to cropped area
        :type original: :class:`bool`, optional
    """

    default_args = {
        "box": List(Number(min_value=0), length=4),
        "original": Type(bool, default=False),
    }

    def apply(self, image, **kwargs):
        lx, rx, ty, by = (
            kwargs["box"][0],
            kwargs["box"][1],
            kwargs["box"][2],
            kwargs["box"][3],
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

    default_args = {"x": Number(default=0), "y": Number(default=0)}

    def apply(self, image, **kwargs):
        height, width = image.shape[:2]

        matrix = np.float32([[1, 0, kwargs["x"]], [0, 1, kwargs["y"]]])

        return cv2.warpAffine(image, matrix, (width, height))
