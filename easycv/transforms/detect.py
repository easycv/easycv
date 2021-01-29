import cv2
import numpy as np

from easycv.transforms.base import Transform
from easycv.transforms.color import GrayScale
from easycv.transforms.spatial import Crop
from easycv.transforms.edges import Canny
from easycv.resources import get_resource
import easycv.transforms.filter
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

    arguments = {
        "scale": Number(default=1.3),
        "min_neighbors": Number(min_value=0, only_integer=True, default=5),
    }

    outputs = {
        "rectangles": List(
            List(List(Number(min_value=0, only_integer=True), length=2), length=2)
        ),
    }

    def process(self, image, **kwargs):
        cascade_file = get_resource(
            "haar-face-cascade", "haarcascade_frontalface_default.xml"
        )
        return CascadeDetector(cascade=str(cascade_file), **kwargs).apply(image)


class Eyes(Transform):

    arguments = {
        "scale": Number(default=1.1),
        "min_neighbors": Number(min_value=0, only_integer=True, default=3),
    }

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
            eyes = CascadeDetector(cascade=str(cascade_file), **kwargs).apply(
                face_image
            )["rectangles"]
            for eye in eyes:
                adjusted = []
                for i in range(len(eye)):
                    adjusted.append((eye[i][0] + face[0][0], eye[i][1] + face[0][1]))
                rectangles.append(adjusted)

        return {"rectangles": rectangles}


class Smile(Transform):

    arguments = {
        "scale": Number(default=1.2),
        "min_neighbors": Number(min_value=0, only_integer=True, default=20),
    }

    outputs = {
        "rectangles": List(
            List(List(Number(min_value=0, only_integer=True), length=2), length=2)
        ),
    }

    def process(self, image, **kwargs):
        faces = Faces().apply(image)
        cascade_file = get_resource("haar-smile-cascade", "haarcascade_smile.xml")
        rectangles = []
        for face in faces["rectangles"]:
            face_image = Crop(rectangle=face).apply(image)
            smile = CascadeDetector(cascade=str(cascade_file), **kwargs).apply(
                face_image
            )["rectangles"]
            if smile:
                adjusted = []
                for i in range(len(smile[0])):
                    adjusted.append(
                        (smile[0][i][0] + face[0][0], smile[0][i][1] + face[0][1])
                    )
                rectangles.append(adjusted)
        return {"rectangles": rectangles}


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


class Detect(Transform):
    methods = {
        "yolo": {"arguments": ["confidence", "threshold"]},
        "ssd": {"arguments": ["confidence"]},
    }
    default_method = "yolo"

    arguments = {
        "confidence": Number(min_value=0, max_value=1, default=0.5),
        "threshold": Number(min_value=0, max_value=1, default=0.3),
    }

    outputs = {
        "boxes": List(
            List(
                List(List(Number(min_value=0, only_integer=True), length=2), length=2),
                List(Number(min_value=0, max_value=255, only_integer=True), length=3),
                Type(str),
            )
        )
    }

    @staticmethod
    def labels(model):
        if model == "yolo":
            labels_path = get_resource("yolov3", "coco.names")
            labels = open(str(labels_path)).read().strip().split("\n")
        else:
            labels = [
                "background",
                "aeroplane",
                "bicycle",
                "bird",
                "boat",
                "bottle",
                "bus",
                "car",
                "cat",
                "chair",
                "cow",
                "diningtable",
                "dog",
                "horse",
                "motorbike",
                "person",
                "pottedplant",
                "sheep",
                "sofa",
                "train",
                "tvmonitor",
            ]
        return labels

    def process(self, image, **kwargs):
        labels = self.labels(kwargs["method"])
        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

        if kwargs["method"] == "yolo":
            config = get_resource("yolov3", "yolov3.cfg")
            weights = get_resource("yolov3", "yolov3.weights")

            net = cv2.dnn.readNetFromDarknet(str(config), str(weights))

            layers = net.getLayerNames()
            layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            blob = cv2.dnn.blobFromImage(
                image, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )

            h, w = image.shape[:2]
            net.setInput(blob)
            outputs = net.forward(layers)

            rectangles = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > kwargs["confidence"]:
                        # scale the bounding box back
                        rectangle = detection[0:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = rectangle.astype("int")

                        # compute top-left corner
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        rectangles.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # apply non-maximum suppression
            indexes_to_keep = cv2.dnn.NMSBoxes(
                rectangles, confidences, kwargs["confidence"], kwargs["threshold"]
            )

            boxes = []
            if len(indexes_to_keep) > 0:
                for i in indexes_to_keep.flatten():
                    (x, y) = (rectangles[i][0], rectangles[i][1])
                    (w, h) = (rectangles[i][2], rectangles[i][3])
                    color = [int(c) for c in colors[class_ids[i]]]
                    label = "{}: {:.4f}".format(
                        labels[int(class_ids[i])], confidences[i]
                    )
                    boxes.append([[(x, y), (w, h)], color, label])

            return {"boxes": boxes}
        else:
            prototxt = get_resource("ssd-mobilenet", "MobileNetSSD_deploy.prototxt")
            model = get_resource("ssd-mobilenet", "MobileNetSSD_deploy.caffemodel")
            net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            net.setInput(blob)
            detections = net.forward()

            boxes = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > kwargs["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    width, height = int(endX - startX), int(endY - startY)
                    label = "{}: {:.4f}".format(labels[idx], confidence)
                    color = [int(c) for c in colors[idx]]

                    boxes.append([[(startX, startY), (width, height)], color, label])

            return {"boxes": boxes}
