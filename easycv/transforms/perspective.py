import cv2
import numpy as np

from easycv.utils import order_corners, distance
from easycv.validators import List, Number
from easycv.transforms.base import Transform


class Perspective(Transform):
    """
        Perspective is a transform that changes the perspective of the image to the one given by \
        the provided corners/points. The new image will have the given points as corners.

        :param points: Corners of the desired perspective
        :type points: :class:`list`
    """

    arguments = {
        "points": List(List(Number(min_value=0, only_integer=True), length=2)),
    }

    def process(self, image, **kwargs):
        if len(kwargs["points"]) != 4:
            raise ValueError("Must receive 4 points.")

        corners = order_corners(kwargs["points"])
        tl, tr, br, bl = corners
        corners = np.array(corners, dtype="float32")

        new_width = max(distance(br, bl), distance(tr, tl))
        new_height = max(distance(tr, br), distance(tl, bl))

        dst = np.array(
            [
                [0, 0],
                [new_width - 1, 0],
                [new_width - 1, new_height - 1],
                [0, new_height - 1],
            ],
            dtype="float32",
        )

        shift_matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, shift_matrix, (new_width, new_height))

        return warped
