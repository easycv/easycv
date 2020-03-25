import cv2
import numpy as np

from cv.transforms.base import Transform
from cv.validators import Number, Option, List, Type
from cv.utils import interpolation_methods


class Resize(Transform):
    """
    Resize is a transform that resizes a image given the desired widht an height, using a certain
    interpolation for the missing pixels.

    :param width: Output image width
    :type width: :class:`int`
    :param height: Output image height
    :type height: :class:`int`, optional
    :param method: Interpolation method
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
    default_args = {
        "degrees": Number(),
        "scale": Number(default=1),
        "center": List(Number(), length=2, default="auto"),
        "bounded": Type(bool, default=True),
    }

    def apply(self, image, **kwargs):
        (h, w) = image.shape[:2]
        if kwargs["center"] == "auto" or kwargs["bounded"]:
            kwargs["center"] = (w / 2, h / 2)

        matrix = cv2.getRotationMatrix2D(
            kwargs["center"], -kwargs["degrees"], kwargs["scale"]
        )

        if kwargs["bounded"]:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])

            nW = int((h * sin) + (w * cos))
            h = int((h * cos) + (w * sin))

            matrix[0, 2] += (nW / 2) - kwargs["center"][0]
            matrix[1, 2] += (h / 2) - kwargs["center"][1]

            w = nW

        return cv2.warpAffine(image, matrix, (w, h))


class Crop(Transform):
    default_args = {
        "x": List(Number(min_value=0), length=2),
        "y": List(Number(min_value=0), length=2),
    }

    def apply(self, image, **kwargs):
        return image[kwargs["x"][0] : kwargs["x"][1], kwargs["y"][0] : kwargs["y"][1]]


class Translate(Transform):
    default_args = {
        "x": Number(default=0),
        "y": Number(default=0)
    }

    def apply(self, image, **kwargs):
        height, width = image.shape[:2]

        matrix = np.float32([[1, 0, kwargs["xShift"]], [0, 1, kwargs["yShift"]]])

        return cv2.warpAffine(image, matrix, (width, height))
