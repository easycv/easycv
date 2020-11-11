import cv2

from easycv.transforms.base import Transform
from easycv.transforms.spatial import Resize
from easycv.resources.resources import get_resource
import easycv.image as Image


class StyleTransfer(Transform):

    methods = [
        "candy",
        "la_muse",
        "mosaic",
        "feathers",
        "the_scream",
        "udnie",
        "the_wave",
        "starry_night",
        "la_muse2",
        "composition_vii",
    ]

    def process(self, image, **kwargs):
        net = get_resource("style_transfer", self._args["method"] + ".t7")
        net = cv2.dnn.readNetFromTorch(str(net))

        r = 600 / float(image.shape[1])
        dim = (image.shape[1], int(image.shape[0] * r))
        image = Resize(width=dim[0], height=dim[1]).apply(image)

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
        )
        net.setInput(blob)

        output = net.forward()
        output = output.reshape((3, output.shape[2], output.shape[3]))
        output[0] += 103.939
        output[1] += 116.779
        output[2] += 123.680
        output /= 255.0
        output = output.transpose(1, 2, 0)

        return output

    def style(self):
        method = self._args["method"]
        if method == "la_muse2":
            method = "la_muse"
        return Image.Image(
            "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/"
            + method
            + ".jpg?raw=true"
        )
