from cv.io import open


class Image:
    def __init__(self, path):
        self._data = open(path)
