import cv2
import numpy as np

import sys
from math import ceil


def nearest_square_side(n):
    return ceil(n ** 0.5)


def _sort_by(points, axis):
    return sorted(points, key=lambda x: x[0 if axis == "x" else 1])


def order_corners(corners):
    sorted_by_x = _sort_by(corners, "x")
    left_points = sorted_by_x[:2]
    right_points = sorted_by_x[2:]

    (tl, bl) = _sort_by(left_points, "y")
    (tr, br) = _sort_by(right_points, "y")

    return tl, tr, br, bl


def distance(point1, point2):
    return int(np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)))


def running_on_notebook():
    if "IPython" in sys.modules:
        from IPython import get_ipython

        return "IPKernelApp" in get_ipython().config
    else:
        return False


interpolation_methods = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}

lines = {"line_AA": cv2.LINE_AA, "line_4": cv2.LINE_4, "line_8": cv2.LINE_8}

font = {
    "SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
    "COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}
