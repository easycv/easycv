import io

import numpy as np
from functools import wraps

from cv.pipeline import Pipeline
from cv.errors.io import InvalidImageInputSource
from cv.io import save, valid_image_source, get_image_array, show


def auto_compute(decorated):
    @wraps(decorated)
    def wrapper(image, *args):
        image.compute(in_place=True)
        return decorated(image, *args)

    return wrapper


def auto_compute_property(decorated):
    @property
    @wraps(decorated)
    def wrapper(image, *args):
        image.compute(in_place=True)
        return decorated(image, *args)

    return wrapper


class Image:
    """
    This class represents an **image**.

    Images can be created from a NumPy array containing the image data, a path to a local file
    or a link to an image on the web. Transforms and pipelines can easily be applied to any image.
    If the Image is lazy, computations will be delayed until needed or until the image is computed.
    This can facilitate large scale processing and distributed computation.

    :param source: Image data source. An array representing the image or a path/link to a file
    containing the image
    :type source: :class:`str`/:class:`~numpy:numpy.ndarray`
    :param pipeline: Pipeline to be applied to the image at creation time, defaults to None
    :type pipeline: :class:`~cv.pipeline.Pipeline`, optional
    :param lazy: `True` if the image is lazy (computations are delayed until needed),
     defaults to False
    :type lazy: :class:`boolean`, optional
    """

    def __init__(self, source, pipeline=None, lazy=False):
        self._lazy = lazy
        self._pending = Pipeline([]) if pipeline is None else pipeline.copy()

        if not valid_image_source(source):
            raise InvalidImageInputSource()

        if self._lazy:
            self._source = source
            self._img = None
        else:
            self._pending.clear()
            self._img = self._pending(get_image_array(source))

    @property
    def loaded(self):
        """
        Check if image is loaded or if it still needs to be downloaded/decoded.

        :return: `True` if loaded, `False` otherwise
        :rtype: :class:`bool`
        """
        return self._img is not None

    @property
    def pending(self):
        """
        Returns all pending transforms/pipelines.

        :return: Pipeline containing pending operations
        :rtype: :class:`~cv.pipeline.Pipeline`
        """
        return self._pending

    @auto_compute_property
    def height(self):
        """
        Returns image height.

        :return: Image height
        :rtype: :class:`int`
        """
        return self._img.shape[0]

    @auto_compute_property
    def width(self):
        """
        Returns image width.

        :return: Image width
        :rtype: :class:`int`
        """
        return self._img.shape[1]

    @auto_compute_property
    def array(self):
        """
        Returns a NumPy array that represents the image.

        :return: Image as NumPy array
        :rtype: :class:`~numpy:numpy.ndarray`
        """
        return self._img

    def load(self):
        """
        Loads the image if it isn't already loaded
        """
        if not self.loaded:
            self._img = get_image_array(self._source)

    def apply(self, transform, in_place=False):
        """
        Returns a new Image with the Transform or Pipeline applied.
        If the image is lazy the transform/pipeline will be stored as a pending operation
        (no computation is done).
        If `in_place` is *True* the operation will change the current image instead of returning
        a new Image.

        :param transform: Transform/Pipeline to be applied
        :type transform: :class:`~cv.transforms.base.Transform`/:class:`~cv.pipeline.Pipeline`
        :param in_place: `True` to change the current image, `False` to return a new one with
        the transform applied, defaults to `False`
        :type in_place: :class:`bool`, optional
        :return: The new Image if `in_place` is *False*
        :rtype: :class:`~cv.image.Image`
        """
        if in_place:
            if self._lazy:
                self._pending.add_transform(transform)
            else:
                self.load()
                self._img = transform(self._img)
        else:
            if self._lazy:
                new_pending = self._pending.copy()
                new_pending.add_transform(transform)
                return Image(self._source, pipeline=new_pending, lazy=True)
            else:
                self.load()
                return Image(transform(self._img))

    def compute(self, in_place=False):
        """
        Returns a new Image with all the pending operations applied.
        If `in_place` is *True* the pending operations will be applied
        to the current image instead.

        :param in_place: `True` to change the current image, `False` to return a new one with the
         pending transforms applied, defaults to `False`
        :type in_place: :class:`bool`, optional
        :return: The new Image if `in_place` is *False*
        :rtype: :class:`~cv.image.Image`
        """
        self.load()
        if in_place:
            self._img = self._pending(self._img)
            self._pending.clear()
        else:
            result = Image(self._pending(self._img), lazy=True)
            return result

    @auto_compute
    def show(self):
        show(self._img)

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
