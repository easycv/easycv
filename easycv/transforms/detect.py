import cv2

from easycv.resources import get_resource
from easycv.transforms.base import Transform
from easycv.transforms.color import GrayScale
from easycv.transforms.spatial import Crop
from easycv.validators import Type, List, Number, File

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


class CascadeDetector(Transform):

    arguments = {
        "cascade": File(),
        "scale": Number(default=1.1),
        "min_neighbors": Number(min_value=0, only_integer=True, default=3),
        "min_size": Number(min_value=0, default="auto"),
        "max_size": Number(min_value=0, default="auto"),
    }

    outputs = {
        "rectangles": List(
            List(List(Number(min_value=0, only_integer=True), length=2), length=2)
        ),
    }

    def process(self, image, **kwargs):
        cascade = cv2.CascadeClassifier(kwargs["cascade"])
        gray = GrayScale().apply(image)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=kwargs["scale"],
            minNeighbors=kwargs["min_neighbors"],
            minSize=kwargs["min_size"] if kwargs["min_size"] != "auto" else None,
            maxSize=kwargs["max_size"] if kwargs["max_size"] != "auto" else None,
        )

        rectangles = []
        for x, y, w, h in detections:
            rectangles.append([(x, y), (x + w, y + h)])

        return {"rectangles": rectangles}


class Faces(Transform):

    outputs = {
        "rectangles": List(
            List(List(Number(min_value=0, only_integer=True), length=2), length=2)
        ),
    }

    def process(self, image, **kwargs):
        cascade_file = get_resource(
            "haar-face-cascade", "haarcascade_frontalface_default.xml"
        )
        return CascadeDetector(
            cascade=str(cascade_file), scale=1.3, min_neighbors=5
        ).apply(image)


class Eyes(Transform):

    outputs = {
        "rectangles": List(
            List(List(Number(min_value=0, only_integer=True), length=2), length=2)
        ),
    }

    def process(self, image, **kwargs):
        cascade_file = get_resource("haar-eye-cascade", "haarcascade_eye.xml")

        rectangles = []
        for face in Faces().apply(image)["rectangles"]:
            face_image = Crop(rectangle=face).apply(image)
            eyes = CascadeDetector(cascade=str(cascade_file)).apply(face_image)[
                "rectangles"
            ]
            for eye in eyes:
                adjusted = []
                for i in range(len(eye)):
                    adjusted.append([eye[i][0] + face[0][0], eye[i][1] + face[0][1]])
                rectangles.append(adjusted)

        return {"rectangles": rectangles}
