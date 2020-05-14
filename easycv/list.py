import easycv.image
from easycv.errors.list import InvalidListInputSource
from easycv.io import show_grid
from easycv.collection import auto_compute
from easycv.transforms.base import Transform
from copy import deepcopy
import ray


class List:
    def __init__(self, images):
        if isinstance(images, list) and all(
            isinstance(i, easycv.image.Image) for i in images
        ):
            self._images = images
        else:
            raise InvalidListInputSource()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._images[key]
        elif isinstance(key, slice):
            return List(self._images[key])
        else:
            raise TypeError("Unsupported type to access List.")

    @classmethod
    def random(cls, length, lazy=False):
        images = [easycv.image.Image.random(lazy=lazy) for _ in range(length)]
        return cls(images)

    @staticmethod
    @ray.remote
    def _process_image(operation, image):
        image.load()
        return operation.apply(image)

    @staticmethod
    @ray.remote
    def _compute_image(image):
        return image.compute(image, in_place=False)

    @staticmethod
    def start():
        if not ray.is_initialized():
            ray.init(logging_level=40)

    def __len__(self):
        return len(self._images)

    @staticmethod
    def shutdown():
        ray.shutdown()

    def apply(self, operation, in_place=False, parallel=False):
        if parallel:
            self.start()

        if isinstance(operation, Transform):
            operation.initialize()
        outputs = operation.outputs

        if parallel:
            operation = ray.put(operation)
            operation_outputs = ray.get(
                [self._process_image.remote(operation, i) for i in self._images]
            )
        else:
            operation_outputs = [operation.apply(i) for i in self._images]

        if outputs == {}:
            if in_place:
                self._images = operation_outputs
            else:
                return List(operation_outputs)
        else:
            return operation_outputs

    def compute(self, in_place=True, parallel=False):
        if parallel:
            self.start()

        if parallel:
            images = ray.get([self._compute_image.remote(i) for i in self._images])
        else:
            images = [i.compute(in_place=False) for i in self._images]

        if in_place:
            self._images = images
        else:
            return List(images)

    def copy(self):
        return deepcopy(self)

    @auto_compute
    def show(self, size=(10, 10), shape="auto"):
        show_grid(self._images, size=size, shape=shape)
