import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Number, Option, List, Type
from easycv.utils import interpolation_methods


class Resize(Transform):
    """
    Resize is a transform that resizes an image to a given width and height. Currently supported \
    methods:

    \t**∙ auto** - Automatically detect the best method\n
    \t**∙ nearest** - Poisson-distributed noise generated from the data\n
    \t**∙ linear** - Replaces random pixels with 255\n
    \t**∙ area** - Replaces random pixels with 0\n
    \t**∙ cubic** - Replaces random pixels with either 1 or 0.\n
    \t**∙ lanczos4** - Multiplicative noise using ``out = image + n * image``, where \
    n is uniform noise with specified mean & variance.\n

    :param width: Output image width
    :type width: :class:`int`
    :param height: Output image height
    :type height: :class:`int`
    :param  method: Interpolation method, defaults to "cubic" if the image is to be upscaled and \
    to "area"  if downscaled
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
        Rescale is a transform that rescales an image to a scale factor for x and y.
        The interpolation method can be specified by the method parameter.

        :param fx: Scale factor along the horizontal axis
        :type fx: : class:`int`
        :param fy: Scale factor along the vertical axis
        :type fy: : class:`int`
        :param method: Interpolation method, defaults to "cubic" if the image is to be \
        upscaled and to "area"  if downscaled
        :type method: : class:`str`, optional
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
            Rotate is a transform that rotates an image by degrees according to the provided
            center. It can also be scaled. The user can decide whether to adjust the output  image\
            size to contain the all image, if needed.

            :param degrees: Degrees to rotate the image
            :type degrees: : class:`int`
            :param scale: Scale factor, defaults to 1
            :type scale: : class:`int`
            :param center: Center of rotation, defaults to the image center
            :type center: : class:'list'/'tuple', optional
            :param bounded: Flag to decide whether the image is to be bounded, defaults to True
            :type bounded: : class:'bool', optional
        """

    default_args = {
        "degrees": Number(),
        "scale": Number(default=1),
        "center": List(
            Number(min_value=0, only_integer=True), length=2, default="auto"
        ),
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

            n_w = int((h * sin) + (w * cos))
            h = int((h * cos) + (w * sin))

            matrix[0, 2] += (n_w / 2) - kwargs["center"][0]
            matrix[1, 2] += (h / 2) - kwargs["center"][1]

            w = n_w

        return cv2.warpAffine(image, matrix, (w, h))


class Crop(Transform):
    """
        Crop is a transform that crops a rectangular portion of an image , if original is True then
        the image size will be kept.

        :param box: A 4-tuple defining the left, right, upper, and lower pixel coordinate.
        :type box: :class:'list'/'tuple'
        :param original: True to keep original image size, False to resize to cropped area
        :type original: :class:'bool', optional
    """

    default_args = {
        "box": List(Number(min_value=0), length=4),
        "original": Type(bool, default=True),
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
        :type x: :class:'int'
        :param y: y value, defaults to 0
        :type y: :class: 'int'
    """

    default_args = {"x": Number(default=0), "y": Number(default=0)}

    def apply(self, image, **kwargs):
        height, width = image.shape[:2]

        matrix = np.float32([[1, 0, kwargs["x"]], [0, 1, kwargs["y"]]])

        return cv2.warpAffine(image, matrix, (width, height))
