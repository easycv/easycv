import io
import os
import base64
import json

import numpy as np

from easycv.collection import Collection, auto_compute
from easycv.errors.io import InvalidImageInputSource
from easycv.io import save, valid_image_source, get_image_array, show, random_dog_image
from easycv.output import Output
from easycv.transforms.base import Transform


class Image(Collection):
    """
    This class represents an image.
    Images can be created from a NumPy array containing the **image** data, a path to a local file
    or a link to an **image** on the web. :doc:`Transforms <transforms/index>` and \
    :doc:`Pipelines <pipeline>` can easily be applied to any **image**.
    If the image is lazy, computations will be delayed until needed or until the image is \
    computed. This can facilitate large scale processing and distributed computation.

    :param source: Image data source. An array representing the image or a path/link to a file \
    containing the image
    :type source: :class:`str`/:class:`~numpy:numpy.ndarray`
    :param pipeline: Pipeline to be applied to the image at creation time, defaults to None
    :type pipeline: :class:`~easycv.pipeline.Pipeline`, optional
    :param lazy: `True` if the image is lazy (computations are delayed until needed), defaults to \
    False
    :type lazy: :class:`boolean`, optional
    """

    def __init__(self, source, pipeline=None, lazy=False):
        if not valid_image_source(source):
            raise InvalidImageInputSource()

        super().__init__(pending=pipeline, lazy=lazy)

        if self._lazy:
            self._source = source
            self._img = None
        else:
            self._img = self._pending(get_image_array(source))["image"]
            self._pending.clear()

    @classmethod
    def random(cls, lazy=False):
        """
        Get a random image. Currently all images are from `DogApi <https://dog.ceo/dog-api/>`_.

        :return: Random Image
        :rtype: :class:`Image`
        """
        path = random_dog_image()
        return cls(path, lazy=lazy)

    @property
    def loaded(self):
        """
        Check if **image** is loaded or if it still needs to be downloaded/decoded.

        :return: `True` if loaded, `False` otherwise
        :rtype: :class:`bool`
        """
        return self._img is not None

    @property
    @auto_compute
    def height(self):
        """
        Returns image height.

        :return: Image height
        :rtype: :class:`int`
        """
        return self._img.shape[0]

    @property
    @auto_compute
    def width(self):
        """
        Returns **image** width.

        :return: Image width
        :rtype: :class:`int`
        """
        return self._img.shape[1]

    @property
    @auto_compute
    def channels(self):
        """
        Returns **image** number of channels.

        :return: Image numeber of channels
        :rtype: :class:`int`
        """

        if len(self._img.shape) == 2:
            return 1
        else:
            return self._img.shape[2]

    @property
    @auto_compute
    def array(self):
        """
        Returns a NumPy array that represents the **image**.

        :return: Image as NumPy array
        :rtype: :class:`~numpy:numpy.ndarray`
        """
        return self._img

    def load(self):
        """
        Loads the **image** if it isn't already loaded
        """
        if not self.loaded:
            self._img = get_image_array(self._source)

    def apply(self, transform, in_place=False):
        """
        Returns a new **image** with the :doc:`transform <transforms/index>` or \
        :doc:`pipeline <pipeline>` applied.
        If the image is lazy the transform/pipeline will be stored as a pending operation
        (no computation is done).
        If `in_place` is *True* the operation will change the **current image** instead of \
        returning a new Image.

        :param transform: Transform/Pipeline to be applied
        :type transform: :class:`~easycv.transforms.base.Transform`/\
        :class:`~easycv.pipeline.Pipeline`
        :param in_place: `True` to change the current **image**, `False` to return a new one with \
        the transform applied, defaults to `False`
        :type in_place: :class:`bool`, optional
        :return: The new **image** if `in_place` is *False*
        :rtype: :class:`~eascv.image.Image`
        """

        if isinstance(transform, Transform):
            transform.initialize()
        outputs = transform.outputs

        if self._lazy:
            if outputs == {}:  # If transform outputs an image
                if in_place:
                    self._pending.add_transform(transform)
                else:
                    new_source = self._img if self.loaded else self._source
                    new_image = Image(new_source, pipeline=self._pending, lazy=True)
                    new_image.apply(transform, in_place=True)
                    return new_image
            else:
                self.load()
                return Output(self._img, pending=transform)
        else:
            self.load()
            if outputs == {}:  # If transform outputs an image
                if in_place:
                    self._img = transform(self._img)["image"]
                else:
                    new_image = transform(self._img.copy())["image"]
                    return Image(new_image)
            else:
                return transform(self._img)

    def compute(self, in_place=True):
        """
        Returns a new **image** with all the pending operations applied.
        If `in_place` is *True* the pending operations will be applied
        to the current **image** instead.

        :param in_place: `True` to change the current **image**, `False` to return a new one with \
         the pending transforms applied, defaults to `True`
        :type in_place: :class:`bool`, optional
        :return: The new **image** if `in_place` is *False*
        :rtype: :class:`~eascv.image.Image`
        """
        self.load()
        if in_place:
            self._img = self._pending(self._img)["image"]
            self._pending.clear()
            return self
        else:
            result = Image(self._pending(self._img)["image"], lazy=self._lazy)
            return result

    @auto_compute
    def encode(self):
        """
        Returns a encoded version of the **image**

        :return: Encoded image
        :rtype: :class:`str`
        """
        image_data = base64.b64encode(self.array.copy(order="C")).decode("utf-8")
        encoded = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": str(self.array.dtype),
            "data": image_data,
        }
        return json.dumps(encoded)

    @classmethod
    def decode(cls, encoded):
        """
        Creates an image by decoding a previously encoded encoded image.

        :return: Decoded image
        :rtype: :class:`~eascv.image.Image`
        """
        encoded = json.loads(encoded)
        shape = (encoded["height"], encoded["width"], encoded["channels"])
        image_data = bytes(encoded["data"], encoding="utf-8")
        image_array = np.frombuffer(
            base64.decodebytes(image_data), dtype=encoded["dtype"]
        )
        return cls(image_array.reshape(shape))

    @auto_compute
    def show(self, name="Image"):
        """
        Opens a popup window with the **image** displayed on it. This window is resizable and\
        supports zoom/pan. If its impossible to open a popup window this method will return the
        image instead.

        :param name: Window name, defaults to "Image"
        :type name: :class:`str`, optional
        """

        if "DISPLAY" in os.environ:
            show(self._img, name=name)
        else:
            return self

    @auto_compute
    def save(self, filename):
        """
        Saves an **image** to a file under a given filename.

        :param filename: Filename to save
        :type filename: :class:`str`, optional
        """
        save(self.array, filename)

    @auto_compute
    def __eq__(self, other):
        return (
            isinstance(other, Image)
            and np.array_equal(other.array, self.array)
            and self.pending == other.pending
        )

    @auto_compute
    def __repr__(self):
        return "<image size={}x{} at 0x{}>".format(
            self.height, self.width, id(self._img)
        )

    def _repr_png_(self):
        b = io.BytesIO()
        save(self._img, b, "PNG")
        return b.getvalue()
