import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Number
from easycv.utils import morp_methods


class Erode(Transform):
    arguments = {
        "size": Number(min_value=1, only_integer=True, default=5),
        "iterations": Number(min_value=1, only_integer=True, default=1),
    }

    def process(self, image, **kwargs):
        kernel = np.ones((kwargs["size"], kwargs["size"]), np.uint8)
        return cv2.erode(image, kernel, iterations=kwargs["iterations"])


class Dilate(Transform):
    arguments = {
        "size": Number(min_value=1, only_integer=True, default=5),
        "iterations": Number(min_value=1, only_integer=True, default=1),
    }

    def process(self, image, **kwargs):
        kernel = np.ones((kwargs["size"], kwargs["size"]), np.uint8)
        return cv2.dilate(image, kernel, iterations=kwargs["iterations"])


class Morphology(Transform):
    arguments = {
        "size": Number(min_value=1, only_integer=True, default=5),
        "iterations": Number(min_value=1, only_integer=True, default=1),
    }

    methods = ["opening", "closing", "tophat", "blackhat"]
    default_method = "opening"

    def process(self, image, **kwargs):
        kernel = np.ones((kwargs["size"], kwargs["size"]), np.uint8)
        return cv2.morphologyEx(
            image,
            morp_methods[kwargs["method"]],
            kernel,
            iterations=kwargs["iterations"],
        )
