import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from color_transfer import color_transfer
from easycv.validators import Option, List, Number, Image
from easycv.transforms.base import Transform
from easycv.transforms.selectors import Select
from easycv.transforms.spatial import Crop
from easycv.resources import get_resource


class GrayScale(Transform):
    """
    GrayScale is a transform that turns an image into grayscale.
    """

    def process(self, image, **kwargs):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image


class Sepia(Transform):
    """
    Sepia is a transform that applies the sepia effect to an image
    """

    def process(self, image, **kwargs):
        gray = GrayScale().process(image)
        sepia = np.array([153 / 255 * gray, 204 / 255 * gray, gray])
        return sepia.transpose(1, 2, 0).astype("uint8")


class FilterChannels(Transform):
    """
    FilterChannels is a transform that removes color channel(s).

    :param channels: List of channels to remove
    :type channels: :class:`list`
    :param scheme: Image color scheme (rgb or bgr), defaults to "rgb"
    :type scheme: :class:`str`, optional
    """

    arguments = {
        "channels": List(Number(min_value=0, max_value=2, only_integer=True)),
        "scheme": Option(["rgb", "bgr"], default=0),
    }

    def process(self, image, **kwargs):
        channels = np.array(kwargs["channels"])
        if kwargs["scheme"] == "rgb":
            channels = 2 - channels
        if len(channels) > 0:
            image[:, :, channels] = 0
        return image


class GammaCorrection(Transform):
    """
    GammaCorrection is a transform that corrects the contrast of images and displays.

    :param gamma: Gamma value
    :type gamma: :class:`Float`
    """

    arguments = {
        "gamma": Number(min_value=1e-30, default=1),
    }

    def process(self, image, **kwargs):
        table = np.array(
            [((i / 255.0) ** (1.0 / kwargs["gamma"])) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)


class Negative(Transform):
    """
    Negative is a transform that inverts color and brightness in an image.
    """

    def process(self, image, **kwargs):
        return 255 - image


class Cartoon(Transform):
    """
    Cartoon is a transform that creates a stylized / cartoonized image.

    :param smoothing: Determines the amount of smoothing, defaults to 60.
    :type smoothing: :class:`float`, optional
    :param region_size: Determines the size of regions of constant color, defaults to 0.45.
    :type region_size: :class:`float`, optional
    """

    arguments = {
        "smoothing": Number(min_value=0, max_value=200, default=60),
        "region_size": Number(min_value=0, max_value=1, default=0.45),
    }

    def process(self, image, **kwargs):
        return cv2.stylization(
            image, sigma_s=kwargs["smoothing"], sigma_r=kwargs["region_size"]
        )


class PhotoSketch(Transform):
    """
    PhotoSketch is a transform that creates a black and white pencil-like drawing.
    """

    def process(self, image, **kwargs):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
        img_blend = cv2.divide(img_gray, img_blur, scale=256)
        return img_blend


class ColorTransfer(Transform):
    """
    ColorTransfer is a transform that transfers the color of an image to another.

    :param source: Source image from where the colors will be transferred from.
    :type source: :class:`~easycv.image.Image`
    """

    arguments = {
        "source": Image(),
    }

    def process(self, image, **kwargs):
        return color_transfer(kwargs["source"].array, image)


class Hue(Transform):
    """
    Hue is a transform that changes the image hue

    :param value: Value of Hue to Add
    :type value: :class:`int`
    """

    arguments = {
        "value": Number(only_integer=True),
    }

    def process(self, image, **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 0] = (image[:, :, 0] + kwargs["value"]) % 180
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


class Contrast(Transform):
    """
    Contrast is a transform that changes the image contrast

    :param alpha: Value of contrast to Add
    :type alpha: :class:`float`
    """

    arguments = {
        "alpha": Number(only_integer=False),
    }

    def process(self, image, **kwargs):
        image = cv2.addWeighted(image, kwargs["alpha"], image, 0, 0)
        return image


class Brightness(Transform):
    """
    Brightness is a transform that changes the image brightness

    :param beta: Value of brightness to Add
    :type beta: :class:`int`
    """

    arguments = {
        "beta": Number(only_integer=True),
    }

    def process(self, image, **kwargs):
        image = cv2.addWeighted(image, 1, image, 0, kwargs["beta"])
        return image


class Hsv(Transform):
    """
    Hsv is a transform that turns an image to hsv
    """

    def process(self, image, **kwargs):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


class ColorPick(Transform):
    """
    ColorPick is a transform that returns the color of a selected point or the \
    average color of a selected rectangle. Returns the color in RGB.

    :param method: Method to be used, defaults to "point"
    :type method: :class:`str`, optional
    """

    methods = ["point", "rectangle"]
    default_method = "point"

    outputs = {
        "color": List(Number(min_value=0, max_value=255, only_integer=True), length=3)
    }

    def process(self, image, **kwargs):
        if kwargs["method"] == "point":
            point = Select(method="point", n=1).apply(image)["points"][0]
            return {"color": list(image[point[1]][point[0]][::-1])}
        if kwargs["method"] == "rectangle":
            rectangle = Select(method="rectangle").apply(image)["rectangle"]
            cropped = Crop(rectangle=rectangle).apply(image)
            return {
                "color": list(cropped.mean(axis=(1, 0)).round().astype("uint8"))[::-1]
            }


class Colorize(Transform):
    """
    Colorize is a transform that puts the color in a grayscale image
    """

    def process(self, image, **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        proto = get_resource("colorization_zhang", "colorization_deploy_v2.prototxt")
        model = get_resource("colorization_zhang", "colorization_release_v2.caffemodel")
        pts = np.load(str(get_resource("colorization_zhang", "pts_in_hull.npy")))

        net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0] - 50
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]

        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        return (255 * np.clip(colorized, 0, 1)).astype("uint8")


class Quantitization(Transform):
    """
    Quantitization is a Transform that reduces the number of colors to the on give

    :param clusters: Number of colors that the image will have
    :type clusters: :class:`int`, required
    """

    arguments = {
        "clusters": Number(min_value=1, only_integer=True),
    }

    def process(self, image, **kwargs):
        (h, w) = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        clt = MiniBatchKMeans(n_clusters=kwargs["clusters"])
        labels = clt.fit_predict(image)

        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

        return quant
