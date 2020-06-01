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
    Lines is a transform that detects lines in an image using the hough space.

    :param low: Low threshold, defaults to 50
    :type low: :class:`int`, optional
    :param high: High threshold, defaults to 150
    :type high: :class:`int`, optional
    :param size: Kernel size, defaults to 3
    :type size: :class:`int`, optional
    :param rho: Distance resolution of the accumulator in pixels, defaults to 1
    :type rho: :class:`int`/:class:`float`, optional
    :param theta: Angle resolution of the accumulator in radians, defaults to pi/8
    :type theta: :class:`int`/:class:`float`, optional
    :param threshold: Threshold of votes, defaults to 200
    :type threshold: :class:`int`, optional
    :param minLineSize: Minimum line lenght, defaults to 3
    :type minLineSize: :class:`int`/:class:`float`, optional
    :param maxGapLine: Maximum gap between lines to treat them as single line, defaults to 3
    :type maxGapLine: :class:`int`/:class:`float`, optional
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
                "minLineSize",
                "maxGapLine",
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
        "minLineSize": Number(min_value=0, default=3),
        "maxGapLine": Number(min_value=0, default=3),
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
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    lines.append([[x1, y1], [x2, y2]])
        else:
            lines = cv2.HoughLines(
                edges,
                kwargs["rho"],
                kwargs["theta"],
                kwargs["threshold"],
                kwargs["minLineSize"],
                kwargs["maxGapLine"],
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
    :param minDist: Minimal distance between centers of circles, defaults to 1.
    :type minDist: :class:`int`:/:class:`float`, optional
    :param minRadius: Minimum radius of a circle, defaults to 0
    :type minRadius: :class:`int`/:class:`float`, optional
    :param maxRadius: Maximum radius of a circle, defaults to 0
    :type maxRadius: :class:`int`/:class:`float`, optional
    :param cannyThreshold: Threshold to be used in canny, defaults to 200
    :type cannyThreshold: :class:`int`, optional
    :param threshold: Threshold of votes, defaults to 200
    :type threshold: :class:`int`, optional
    """

    arguments = {
        "size": Number(min_value=1, only_integer=True, only_odd=True, default=1),
        "dp": Number(min_value=1, only_integer=True, default=1),
        "minDist": Number(min_value=0, default=1),
        "minRadius": Number(min_value=0, default=0),
        "maxRadius": Number(min_value=0, default=0),
        "cannyThreshold": Number(min_value=0, only_integer=True, default=200),
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
            kwargs["minDist"],
            param1=kwargs["cannyThreshold"],
            param2=kwargs["threshold"],
            minRadius=kwargs["minRadius"],
            maxRadius=kwargs["maxRadius"],
        )
        return {"circles": circles[0]}
