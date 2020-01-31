import io

import numpy as np

from cv.pipeline import Pipeline
from cv.io import open_image, save
from cv.errors.io import InvalidImageInputSource


class Image:
    def __init__(self, source, pipeline=Pipeline([]), lazy=False):
        self._lazy = lazy
        self._pending = pipeline

        if type(source) == str:
            self._img = open_image(source)
        elif type(source) == np.ndarray:
            self._img = np.copy(source)
            np.clip(self._img, 0, 1, out=self._img).astype(np.float32)
        else:
            raise InvalidImageInputSource()

        if not lazy:
            self.apply(pipeline, in_place=True)

    @property
    def height(self):
        return self._img.shape[0]

    @property
    def width(self):
        return self._img.shape[1]

    def array(self):
        return self._img

    def pending(self):
        return self._pending

    def apply(self, transform, in_place=False):
        if in_place:
            if self._lazy:
                self._pending.add_transform(transform)
            else:
                self._img = transform(self._img)
        else:
            if self._lazy:
                new_pending = self._pending.copy()
                new_pending.add_transform(transform)
                return Image(self._img, pipeline=new_pending, lazy=True)
            else:
                return Image(transform(self._img))

    def compute(self, in_place=False):
        if in_place:
            self._img = self._pending(self._img)
            self._pending.clear()
        else:
            result = Image(self._pending(self._img), lazy=True)
            return result

    def __eq__(self, other):
        return isinstance(other, Image) and other.array() == self.array() and self.pending() == other.pending()

    def __repr__(self):
        return f"<image size={self.height}x{self.width} at 0x{id(self._img)}>"

    def _repr_png_(self):
        b = io.BytesIO()
        img = (self._img*255).astype(np.uint8)
        save(img, b, 'PNG')
        return b.getvalue()
