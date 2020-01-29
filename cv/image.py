import io

import numpy as np

from cv.io import open_image, save
from cv.errors.io import InvalidImageInputSource


class Image:
    def __init__(self, source):
        if type(source) == str:
            self._img = open_image(source)
        elif type(source) == np.ndarray:
            self._img = np.clip(source, 0, 1).astype(np.float32)
        else:
            raise InvalidImageInputSource()

    @property
    def height(self):
        return self._img.shape[0]

    @property
    def width(self):
        return self._img.shape[1]

    def apply(self, transform, **kwargs):
        return Image(transform(np.copy(self._img), **kwargs))

    def __repr__(self):
        return f"<image size={self.height}x{self.width} at 0x{id(self._img)}>"

    def _repr_png_(self):
        b = io.BytesIO()
        img = (self._img*255).astype(np.uint8)
        save(img, b, 'PNG')
        return b.getvalue()
