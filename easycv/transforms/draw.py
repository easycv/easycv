import cv2 as cv
import numpy as np

from easycv.transforms.base import Transform
from easycv.validators import Number, List, Type, Option
from easycv.utils import lines
from easycv.utils import font


class Draw(Transform):
    """
    The Draw transform provides a way to draw 2D shapes or text on a image. Currently supported\
    methods are:

    \t**∙ ellipse** - Draw an ellipse on an image\n
    \t**∙ line** - Draw a line on an image\n
    \t**∙ polygon** - Draw a polygon on an image\n
    \t**∙ rectangle** - Draw a rectangle on an image\n
    \t**∙ text** - Write text on an image\n

    :param ellipse: Tuple made up by: three arguments(center(int,int), axis1(int), axis2(int)) \
    regarding the ellipse.
    :type ellipse: :class:`tuple`
    :param color: color of the shape in BGR, defaults to "(0,0,0)" - black.
    :type color: :class:`list`/:class:`tuple`, optional
    :param end_angle: Ending angle of the elliptic arc in degrees, defaults to "360".
    :type end_angle: :class:`int`, optional
    :param font: Font to be used, defaults to "SIMPLEX".
    :type font: :class:`str`, optional
    :param filled: True if shape is to be filled, defaults to "false".
    :type filled: :class:`bool`, optional
    :param closed: If true, a line is drawn from the last vertex to the first
    :type closed: :class:`str`, optional
    :param line_type: Line type to be used when drawing a shape, defaults to "line_AA".
    :type line_type: :class:`str`, optional
    :param org: Bottom-left corner of the text in the image.
    :type org: :class:`tuple`
    :param pt1: First point to define a shape.
    :type pt1: :class:`tuple`
    :param pt2: Second point to define a shape.
    :type pt2: :class:`tuple`
    :param points: A list of point.
    :type points: :class:`list`/:class:`tuple`
    :param radius: Radius of the circle.
    :type radius: :class:`int`
    :param rectangle: Tuple made up by: two points(upper left corner(int,int), lower right \
    corner(int,int))regarding the rectangle.
    :type rectangle: :class:`tuple`
    :param rotation_angle: Angle of rotation.
    :type rotation_angle: :class:`int`
    :param size: Size of the text to be drawn, defaults to "5".
    :type size: :class:`int`, optional
    :param start_angle: Starting angle of the elliptic arc in degrees, defaults to "0".
    :type start_angle: :class:`int`, optional
    :param text: Text string to be drawn.
    :type text: :class:`str`
    :param thickness: Thickness of the line used, defaults to "5".
    :type thickness: :class:`int`, optional
    :param x_mirror: When true the text is mirrored in the x axis, defaults to "false".
    :type x_mirror: :class:`bool`
    """

    methods = {
        "ellipse": {
            "arguments": [
                "ellipse",
                "rotation_angle",
                "start_angle",
                "end_angle",
                "filled",
                "color",
                "thickness",
                "line_type",
            ]
        },
        "line": {"arguments": ["pt1", "pt2", "color", "thickness", "line_type"]},
        "polygon": {
            "arguments": [
                "points",
                "closed",
                "color",
                "thickness",
                "line_type",
                "filled",
            ]
        },
        "rectangle": {
            "arguments": ["rectangle", "color", "thickness", "line_type", "filled"]
        },
        "text": {
            "arguments": [
                "text",
                "org",
                "font",
                "size",
                "x_mirror",
                "color",
                "thickness",
                "line_type",
            ]
        },
    }

    arguments = {
        "ellipse": List(
            List(Number(only_integer=True, min_value=0), length=2),
            Number(min_value=0, only_integer=True),
            Number(min_value=0, only_integer=True),
        ),
        "rectangle": List(
            List(Number(min_value=0, only_integer=True), length=2), length=2
        ),
        "color": List(Number(min_value=0, max_value=255), length=3, default=(0, 0, 0)),
        "end_angle": Number(default=360),
        "font": Option(
            [
                "SIMPLEX",
                "PLAIN",
                "DUPLEX",
                "COMPLEX",
                "TRIPLEX",
                "COMPLEX_SMALL",
                "SCRIPT_SIMPLEX",
                "SCRIPT_COMPLEX",
            ],
            default=0,
        ),
        "filled": Type(bool, default=False),
        "closed": Type(bool, default=True),
        "line_type": Option(["line_AA", "line_4", "line_8"], default=0),
        "org": List(Number(min_value=0, only_integer=True), length=2),
        "pt1": List(Number(min_value=0, only_integer=True), length=2),
        "pt2": List(Number(min_value=0, only_integer=True), length=2),
        "points": List(List(Number(min_value=0, only_integer=True), length=2)),
        "radius": Number(min_value=0),
        "rotation_angle": Number(default=0),
        "size": Number(min_value=0, default=5),
        "start_angle": Number(default=0),
        "text": Type(str),
        "thickness": Number(min_value=0, default=5, only_integer=True),
        "x_mirror": Type(bool, default=False),
    }

    def process(self, image, **kwargs):
        if len(image.shape) < 3:
            kwargs["color"] = (
                (0.3 * kwargs["color"][0])
                + (0.59 * kwargs["color"][1])
                + (0.11 * kwargs["color"][2])
            )

        method = kwargs.pop("method")

        kwargs["lineType"] = lines[kwargs.pop("line_type")]

        if method == "line":
            kwargs["pt1"] = tuple(kwargs["pt1"])
            kwargs["pt2"] = tuple(kwargs["pt2"])
            return cv.line(image, **kwargs)

        if method == "polygon":
            kwargs["pts"] = kwargs.pop("points")
            kwargs["pts"] = np.array(kwargs.pop("pts"), np.int32)
            kwargs["pts"] = [kwargs["pts"].reshape((-1, 1, 2))]
            if kwargs.pop("filled"):
                kwargs.pop("thickness")
                kwargs.pop("closed")
                return cv.fillPoly(image, **kwargs)
            else:
                kwargs["isClosed"] = kwargs.pop("closed")
                return cv.polylines(image, **kwargs)

        if method == "text":
            kwargs["org"] = tuple(kwargs["org"])
            kwargs["font"] = font[kwargs["font"]]
            kwargs["fontFace"] = kwargs.pop("font")
            kwargs["fontScale"] = kwargs.pop("size")
            kwargs["bottomLeftOrigin"] = kwargs.pop("x_mirror")
            return cv.putText(image, **kwargs)

        # to make a circle/ellipse/line/rectangle filled thickness must be negative/cv.FILLED
        if kwargs.pop("filled", False):
            kwargs["thickness"] = cv.FILLED

        if method == "ellipse":
            return cv.ellipse(
                image,
                kwargs["ellipse"][0],
                (kwargs["ellipse"][1], kwargs["ellipse"][2]),
                kwargs["rotation_angle"],
                kwargs["start_angle"],
                kwargs["end_angle"],
                kwargs["color"],
                kwargs["thickness"],
                kwargs["lineType"],
            )

        if method == "rectangle":
            pts = kwargs.pop("rectangle")
            return cv.rectangle(image, pts[0], pts[1], **kwargs)
