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
