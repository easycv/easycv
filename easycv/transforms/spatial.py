import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Number, Option, List, Type
from easycv.utils import interpolation_methods


class Resize(Transform):
    """
    Resize is a transform that resizes an image to a given width and height. The interpolation
    method can be specified by the method parameter.

    :param width: Output image width
    :type width: :class:`int`
    :param height: Output image height
    :type height: :class:`int`
    :param  : Interpolation method, defaults to "cubic" if the image is to be upscaled and \
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

        :param fx: scale factor along the horizontal axis
        :type fx: :class:`int`
        :param fy: scale factor along the vertical axis
        :type fy: :class:`int`
        :param method: Interpolation method, defaults to "cubic" if the image is to be \
        upscaled and to "area"  if downscaled
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
            Rotate is a transform that rotates an image by degrees according to the provided
            center. It can also be scaled. The user can decide whether to adjust the image size to
            contain the all image,if needed.

            :param degrees: degrees to rotate the image
            :type degrees: :class:`int`
            :param scale: scale factor, defaults to 1
            :type scale: :class:`int`
            :param center: center by which the image is rotated, defaults to the image center
            :type center: :class:'list'/'tuple', optional
            :param bounded: adjusts the image size to contain the all image,if needed. \
            Defaults to True
            :type bounded: :class:'bool', optional
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

            nW = int((h * sin) + (w * cos))
            h = int((h * cos) + (w * sin))

            matrix[0, 2] += (nW / 2) - kwargs["center"][0]
            matrix[1, 2] += (h / 2) - kwargs["center"][1]

            w = nW

        return cv2.warpAffine(image, matrix, (w, h))


class Crop(Transform):
    """
        Crop is a transform that crops an area given by x and y , if original is True then
        the image size will be kept and the cropped area blacked

        :param x: x beginning and end, given by a tuple or list (beginning,end)
        :type x: :class:'list'/'tuple'
        :param y: y beginning and end, given by a tuple or list (beginning,end)
        :type y: :class: 'list'/'tuple'
        :param original: center by which the image is rotated, defaults to the image center
        :type original: :class:'list'/'tuple', optional
    """

    default_args = {
        "x": List(Number(min_value=0), length=2, default=(0, 0)),
        "y": List(Number(min_value=0), length=2, default=(0, 0)),
        "original": Type(bool, default=True),
    }

    def apply(self, image, **kwargs):
        lx, rx, ty, by = kwargs["x"][0], kwargs["x"][1], kwargs["y"][0], kwargs["y"][1]

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

        :param x: x values of the vector
        :type x: :class:'int'
        :param y: y values of the vector
        :type y: :class: 'int'
    """

    default_args = {"x": Number(default=0), "y": Number(default=0)}

    def apply(self, image, **kwargs):
        height, width = image.shape[:2]

        matrix = np.float32([[1, 0, kwargs["x"]], [0, 1, kwargs["y"]]])

        return cv2.warpAffine(image, matrix, (width, height))
