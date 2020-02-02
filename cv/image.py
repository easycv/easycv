import io

import numpy as np

from cv.pipeline import Pipeline
from cv.io import open_image, save
from cv.errors.io import InvalidImageInputSource


def valid_image_array(image_array):
    source_is_grayscale = len(image_array.shape) == 2
    source_is_color = len(image_array.shape) == 3 and image_array.shape[2] == 3
    return source_is_grayscale or source_is_color


def valid_image_source(source):
    source_is_str = isinstance(source, str)
    source_is_array = isinstance(source, np.ndarray)
    return source_is_str or (source_is_array and valid_image_array)


def get_image_array(image_source):
    if isinstance(image_source, str):
        return open_image(image_source)
    else:
        image = np.copy(image_source)
        np.clip(image, 0, 1, out=image).astype(np.float32)
        return image


def auto_compute(decorated, *args):
    def wrapper(image):
        image.compute(in_place=True)
        return decorated(image, *args)
    return wrapper


def auto_compute_property(decorated, *args):
    @property
    def wrapper(image):
        image.compute(in_place=True)
        return decorated(image, *args)
    return wrapper


class Image:
    def __init__(self, source, pipeline=None, lazy=False, loaded=False):
        self._lazy = lazy
        self._pending = Pipeline([]) if pipeline is None else pipeline.copy()

        if not valid_image_source(source):
            raise InvalidImageInputSource()

        if self._lazy:
            self._loaded = loaded
            self._img = source
        else:
            self._loaded = True
            self._pending.clear()
            self._img = self._pending(get_image_array(source))

    @property
    def loaded(self):
        return self._loaded

    @property
    def pending(self):
        return self._pending

    @auto_compute_property
    def height(self):
        return self._img.shape[0]

    @auto_compute_property
    def width(self):
        return self._img.shape[1]

    @auto_compute_property
    def array(self):
        return self._img

    def load(self):
        if not self._loaded:
            self._loaded = True
            self._img = get_image_array(self._img)

    def apply(self, transform, in_place=False):
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
                return Image(self._img, pipeline=new_pending, lazy=True, loaded=self._loaded)
            else:
                self.load()
                return Image(transform(self._img))

    def compute(self, in_place=False):
        self.load()
        if in_place:
            self._img = self._pending(self._img)
            self._pending.clear()
        else:
            result = Image(self._pending(self._img), lazy=True)
            return result

    @auto_compute
    def __eq__(self, other):
        return isinstance(other, Image) and other.array() == self.array() and self.pending() == other.pending()

    @auto_compute
    def __repr__(self):
        return f"<image size={self.height}x{self.width} at 0x{id(self._img)}>"

    def _repr_png_(self):
        b = io.BytesIO()
        img = (self._img*255).astype(np.uint8)
        save(img, b, 'PNG')
        return b.getvalue()
