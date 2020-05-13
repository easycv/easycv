import easycv.image
from easycv.errors.list import InvalidListInputSource
from easycv.io import show_grid
from easycv.collection import Collection, auto_compute
from easycv.output import Output
from easycv.transforms.base import Transform
from copy import deepcopy
import ray


class List(Collection):
    def __init__(self, images, pipeline=None, lazy=False):
        if isinstance(images, list) and all(
            isinstance(i, easycv.image.Image) for i in images
        ):
            self._images = images
        else:
            raise InvalidListInputSource()

        super().__init__(pending=pipeline, lazy=lazy)

    @classmethod
    def random(cls, length, lazy=False):
        images = [easycv.image.Image.random(lazy=lazy) for _ in range(length)]
        return cls(images, lazy=lazy)

    @staticmethod
    @ray.remote
    def _process(operation, image):
        image.load()
        return operation.apply(image)

    def apply(self, operation, in_place=False, parallel=False):
        if parallel and not ray.is_initialized():
            ray.init(logging_level=40)

        if isinstance(operation, Transform):
            operation.initialize()
        outputs = operation.outputs

        if self._lazy:
            if outputs == {}:  # If transform outputs an image
                if in_place:
                    self._pending.add_transform(operation)
                else:
                    new_list = List(
                        deepcopy(self._images), pipeline=self._pending, lazy=True
                    )
                    new_list.apply(operation, in_place=True)
                    return new_list
            else:
                result = []
                for img in self._images:
                    img.load()
                    result.append(Output(img.array, pending=operation))
                return result
        else:
            if outputs == {}:  # If transform outputs an image
                if in_place:
                    if parallel:
                        self._images = ray.get(
                            [self._process.remote(operation, i) for i in self._images]
                        )
                    else:
                        for image in self._images:
                            image.apply(operation, in_place=True)
                else:
                    if parallel:
                        operation = ray.put(operation)
                        result = ray.get(
                            [self._process.remote(operation, i) for i in self._images]
                        )
                    else:
                        result = []
                        for image in self._images:
                            result.append(image.apply(operation))
                    return List(result)
            else:
                if parallel:
                    operation = ray.put(operation)
                    return ray.get(
                        [self._process.remote(operation, i) for i in self._images]
                    )
                else:
                    result = []
                    for img in self._images:
                        img.load()
                        result.append(img.apply(operation))
                    return result

    def compute(self, in_place=True, parallel=False):
        if parallel and not ray.is_initialized():
            ray.init(logging_level=40)

        if in_place:
            if parallel:
                operation = ray.put(self._pending)
                self._images = ray.get(
                    [self._process.remote(operation, i) for i in self._images]
                )
            else:
                result = []
                for img in self._images:
                    result.append(img.apply(self._pending))
                self._pending.clear()
        else:
            if parallel:
                operation = ray.put(self._pending)
                return List(
                    ray.get([self._process.remote(operation, i) for i in self._images])
                )
            else:
                result = []
                for img in self._images:
                    img = img.apply(self._pending)
                    img.compute()
                    result.append(img)
                return List(result)

    @auto_compute
    def show(self, size=(10, 10), shape="auto"):
        show_grid(self._images, size=size, shape=shape)
