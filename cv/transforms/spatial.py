import cv2

from cv.transforms.base import Transform
from cv.validators import Number, Option
from cv.utils import interpolation_methods


class Resize(Transform):
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
