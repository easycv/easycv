import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Type, List, Number
from easycv.transforms.color import GrayScale
from easycv.transforms.edges import Canny
import easycv.transforms.filter

try:
    from pyzbar import pyzbar
except ImportError:
    raise ImportError(
        "Error importing pyzbar. Make sure you have zbar installed on your system. "
        + "On linux you can simply run 'sudo apt-get install libzbar0'"
    )


class Scan(Transform):
    """
    Scan is a transform that scans and decodes all QR codes and barcodes in a image. The \
    transform returns the number of detections, the data encoded on each code and their bounding \
    boxes.
    """

    outputs = {
        "detections": Number(min_value=0, only_integer=True),
        "data": List(Type(str)),
        "rectangles": List(
            List(List(Number(min_value=0, only_integer=True), length=2), length=2)
        ),
    }

    def process(self, image, **kwargs):
        data = []
        rectangles = []
        decoded = pyzbar.decode(image)

        for code in decoded:
            data.append(code.data.decode("utf-8"))
            (x, y, width, height) = code.rect
            rectangles.append([(x, y), (x + width, y + height)])

        return {"detections": len(decoded), "data": data, "rectangles": rectangles}


class Lines(Transform):
    """
    Lines is a transform that detects lines in an image using the Hough Transform.

    :param low: Low canny threshold, defaults to 50
    :type low: :class:`int`, optional
    :param high: High canny threshold, defaults to 150
    :type high: :class:`int`, optional
    :param size: Gaussian kernel size (canny), defaults to 3
    :type size: :class:`int`, optional
    :param rho: Distance resolution of the accumulator in pixels, defaults to 1
    :type rho: :class:`int`/:class:`float`, optional
    :param theta: Angle resolution of the accumulator in radians, defaults to pi/180
    :type theta: :class:`int`/:class:`float`, optional
    :param threshold: Threshold of votes, defaults to 200
    :type threshold: :class:`int`, optional
    :param min_size: Minimum line length, defaults to 3
    :type min_size: :class:`int`/:class:`float`, optional
    :param max_gap: Maximum gap between lines to treat them as single line, defaults to 3
    :type max_gap: :class:`int`/:class:`float`, optional
    """

    methods = {
        "normal": {"arguments": ["low", "high", "size", "rho", "theta", "threshold"]},
        "probablistic": {
            "arguments": [
                "low",
                "high",
                "size",
                "rho",
                "theta",
                "threshold",
                "min_size",
                "max_gap",
            ]
        },
    }

    default_method = "normal"

    arguments = {
        "low": Number(min_value=1, max_value=255, only_integer=True, default=50),
        "high": Number(min_value=1, max_value=255, only_integer=True, default=150),
        "size": Number(
            min_value=3, max_value=7, only_integer=True, only_odd=True, default=3
        ),
        "rho": Number(min_value=0, default=1),
        "theta": Number(min_value=0, max_value=np.pi / 2, default=np.pi / 180),
        "threshold": Number(min_value=0, only_integer=True, default=200),
        "min_size": Number(min_value=0, default=3),
        "max_gap": Number(min_value=0, default=3),
    }

    outputs = {
        "lines": List(
            List(List(Number(min_value=0, only_integer=True), length=2), length=2)
        )
    }

    def process(self, image, **kwargs):
        gray = GrayScale().apply(image)
        edges = Canny(
            low=kwargs["low"], high=kwargs["high"], size=kwargs["size"]
        ).apply(gray)
        if kwargs["method"] == "normal":
            h_lines = cv2.HoughLines(
                edges, kwargs["rho"], kwargs["theta"], kwargs["threshold"]
            )
            lines = []
            if h_lines is not None:
                for rho, theta in h_lines[:, 0]:
                    x1 = int(rho * np.cos(theta))
                    y1 = int(rho * np.sin(theta))
                    if x1 > y1:
                        y1 = 0
                    elif y1 > x1:
                        x1 = 0
                    x2 = (rho - image.shape[0] * np.sin(theta)) / np.cos(theta)
                    y2 = (rho - image.shape[1] * np.cos(theta)) / np.sin(theta)
                    if 0 <= x2 <= image.shape[1] or np.sin(theta) == 0:
                        y2 = image.shape[0]
                    elif 0 <= y2 <= image.shape[0] or np.cos(theta) == 0:
                        x2 = image.shape[1]
                    lines.append([[int(x1), int(y1)], [int(x2), int(y2)]])
        else:
            lines = cv2.HoughLines(
                edges,
                kwargs["rho"],
                kwargs["theta"],
                kwargs["threshold"],
                kwargs["min_size"],
                kwargs["max_gap"],
            )
        return {"lines": lines}


class Circles(Transform):
    """
    Circles is a transform that can detect where there are circles in an image using the hough
    space

    :param size: Kernel size for media blur, defaults to 1
    :type size: :class:`int`, optional
    :param dp: Inverse ratio of the accumulator resolution to the image resolution, defaults to 1.
    :type dp: :class:`int`, optional
    :param min_dist: Minimal distance between centers of circles, defaults to 1.
    :type min_dist: :class:`int`/:class:`float`, optional
    :param min_radius: Minimum radius of a circle, defaults to 0
    :type min_radius: :class:`int`/:class:`float`, optional
    :param max_radius: Maximum radius of a circle, defaults to 0
    :type max_radius: :class:`int`/:class:`float`, optional
    :param high: High canny threshold, defaults to 200
    :type high: :class:`int`, optional
    :param threshold: Threshold of votes, defaults to 200
    :type threshold: :class:`int`, optional
    """

    arguments = {
        "size": Number(min_value=1, only_integer=True, only_odd=True, default=1),
        "dp": Number(min_value=1, only_integer=True, default=1),
        "min_dist": Number(min_value=0, default=1),
        "min_radius": Number(min_value=0, default=0),
        "max_radius": Number(min_value=0, default=0),
        "high": Number(min_value=0, only_integer=True, default=200),
        "threshold": Number(min_value=0, only_integer=True, default=200),
    }

    outputs = {"circles": List(List(Number(min_value=0, only_integer=True), length=3))}

    def process(self, image, **kwargs):
        blur = easycv.transforms.filter.Blur(
            method="median", size=kwargs["size"]
        ).apply(image)
        gray = GrayScale().apply(blur)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            kwargs["dp"],
            kwargs["min_dist"],
            param1=kwargs["high"],
            param2=kwargs["threshold"],
            minRadius=kwargs["min_radius"],
            maxRadius=kwargs["max_radius"],
        )
        return {"circles": circles[0]}
