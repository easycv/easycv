import io

from cv.io import open, save


class Image:
    def __init__(self, path):
        self._img = open(path)

    def height(self):
        return self._img.shape[0]

    def width(self):
        return self._img.shape[1]

    def __repr__(self):
        return f"<image size={self.height()}x{self.width()} at 0x{id(self._img)}>"

    def _repr_png_(self):
        b = io.BytesIO()
        save(self._img, b, 'PNG')
        return b.getvalue()
