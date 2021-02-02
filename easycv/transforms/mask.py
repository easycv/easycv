import cv2

from easycv.transforms.base import Transform
from easycv.validators import Number, List, Type, Image


class Mask(Transform):
    """
    Mask applies a mask to an image.

    :param mask: Mask to apply
    :type brush: :class:`Image`
    :param inverse: Inverts mask
    :type inverse: :class:`bool`
    :param fill_color: Color to fill
    :type fill_color: :class:`List`
    """

    arguments = {
        "mask": Image(),
        "inverse": Type(bool, default=False),
        "fill_color": List(
            Number(only_integer=True, min_value=0, max_value=255),
            length=3,
            default=(0, 0, 0),
        ),
    }

    def process(self, image, **kwargs):

        if kwargs["inverse"]:
            mask = cv2.bitwise_not(kwargs["mask"].array)
        else:
            mask = kwargs["mask"].array

        image = cv2.bitwise_and(image, image, mask=mask)
        image[mask == 0] = kwargs["fill_color"]

        return image


class Inpaint(Transform):
    """
    Inpaint applies an inpainting technique to an image.

    :param radius: Inpainting radius
    :type radius: :class:`int`
    :param mask: Mask to apply inpaint
    :type mask: :class:`Image`
    """

    methods = {
        "telea": {"arguments": ["radius", "mask"]},
        "ns": {"arguments": ["radius", "mask"]},
    }
    default_method = "telea"

    arguments = {
        "radius": Number(only_integer=True, min_value=0, default=3),
        "mask": Image(),
    }

    def process(self, image, **kwargs):
        flag = cv2.INPAINT_TELEA if kwargs["method"] == "telea" else cv2.INPAINT_NS

        return cv2.inpaint(image, kwargs["mask"].array, kwargs["radius"], flags=flag)
