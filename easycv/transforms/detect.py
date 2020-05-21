from easycv.transforms.base import Transform
from easycv.validators import Type, List, Number

try:
    from pyzbar import pyzbar
except ImportError:
    raise ImportError(
        "Error importing pyzbar. Make sure you have zbar installed on your system. "
        + "On linux you can simply run 'sudo apt-get install libzbar0'"
    )


class Scan(Transform):
    """
    Gradient is a transform that computes the gradient of an image.

    :param axis: Axis to compute, defaults to "both" (magnitude)
    :type axis: :class:`str`, optional
    :param method: Gradient calculation method, defaults to "sobel"
    :type method: :class:`str`, optional
    :param size: Kernel size, defaults to 5
    :type size: :class:`int`, optional
    """

    outputs = {
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

        return {"data": data, "rectangles": rectangles}
