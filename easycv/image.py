import io

import numpy as np
from functools import wraps

from easycv.pipeline import Pipeline
from easycv.errors.io import InvalidImageInputSource
from easycv.io import save, valid_image_source, get_image_array, show


def auto_compute(decorated):
    """ Decorator to auto-compute **image** before running method. Add this to all methods that \
    need the updated image array to function properly."""

    @wraps(decorated)
    def wrapper(image, *args):
        image.compute(in_place=True)
        return decorated(image, *args)

    return wrapper


class Image:
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
        self._lazy = lazy
        self._pending = Pipeline([], name='pending') if pipeline is None else pipeline.copy()

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
        Check if **image** is loaded or if it still needs to be downloaded/decoded.

        :return: `True` if loaded, `False` otherwise
        :rtype: :class:`bool`
        """
        return self._img is not None

    @property
    def pending(self):
        """
        Returns all pending transforms/pipelines.

        :return: Pipeline containing pending operations
        :rtype: :class:`~eascv.pipeline.Pipeline`
        """
        return self._pending

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
                new_source = self._img if self.loaded else self._source
                return Image(new_source, pipeline=new_pending, lazy=True)
            else:
                self.load()
                return Image(transform(self._img))

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
            self._img = self._pending(self._img)
            self._pending.clear()
            return self
        else:
            result = Image(self._pending(self._img), lazy=True)
            return result

    @auto_compute
    def show(self, name="Image"):
        """
        Opens a popup window with the **image** displayed on it. This window is resizable and\
        supports zoom/pan.

        :param name: Window name, defaults to "Image"
        :type name: :class:`str`, optional
        """
        show(self._img, name=name)

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
