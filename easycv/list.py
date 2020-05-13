import easycv.image
from easycv.errors.list import InvalidListInputSource
from easycv.io import show_grid
import ray


class List:
    def __init__(self, images):
        if isinstance(images, list) and all(
            isinstance(i, easycv.image.Image) for i in images
        ):
            self._images = images
        else:
            raise InvalidListInputSource()

    @classmethod
    def random(cls, length, lazy=False):
        images = [easycv.image.Image.random(lazy=lazy) for _ in range(length)]
        return cls(images)

    @staticmethod
    @ray.remote
    def _process(operation, image):
        return operation.apply(image)

    def apply(self, operation, in_place=False, parallel=False):
        if not parallel:
            if in_place:
                for image in self._images:
                    image.apply(operation, in_place=True)
            else:
                result = []
                for image in self._images:
                    result.append(image.apply(operation))
                return List(result)
        else:
            operation = ray.put(operation)
            images = [ray.put(i) for i in self._images]
            return ray.get([self._process.remote(operation, i) for i in images])

    def show(self, size=(10, 10), shape="auto"):
        show_grid(self._images, size=size, shape=shape)
