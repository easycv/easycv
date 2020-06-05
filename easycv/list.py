from copy import deepcopy

import ray

import easycv.image
from easycv.io import show_grid
from easycv.collection import auto_compute
from easycv.transforms.base import Transform
from easycv.errors.list import InvalidListInputSource


class List:
    """
    This class represents a list of Images.
    Lists can be created from a list of image objects or by asking for a random list of images.
    Images inside the List can be lazy (delayed computation) or normal (everything runs in \
    the moment). Lists support parallel processing, locally or in a distributed cluster.

    :param images: List of the images to include
    :type images: :class:`list`
    """

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

    def __len__(self):
        return len(self._images)

    @staticmethod
    def start():
        """
        Starts the local cluster for parallel processing if it isn't already running.
        """
        if not ray.is_initialized():
            ray.init(logging_level=40)

    @staticmethod
    def shutdown():
        """
        Shutdowns the local cluster for parallel processing if it is running.
        """
        if ray.is_initialized():
            ray.shutdown()

    @classmethod
    def random(cls, length, lazy=False):
        """
        Get a random list of images. Currently all images are from \
        `DogApi <https://dog.ceo/dog-api/>`_.

        :param length: Number of images to include in the list
        :type length: :class:`int`
        :param lazy: `True` to create a List of lazy images, `False` otherwise, defaults to `False`
        :type lazy: :class:`bool`, optional
        :return: Random list of Images
        :rtype: :class:`~easycv.list.List`
        """
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

    def apply(self, operation, in_place=False, parallel=False):
        """
        Returns a new **image** with the :doc:`transform <transforms/index>` or \
        :doc:`pipeline <pipeline>` applied.
        If the image is lazy the transform/pipeline will be stored as a pending operation
        (no computation is done).
        If `in_place` is *True* the operation will change the **current image** instead of \
        returning a new Image.

        :param operation: Operation to be applied
        :type operation: :class:`~easycv.transforms.operation.Operation`
        :param in_place: `True` to change the current **list**, `False` to return a new one with \
        the transform applied, defaults to `False`
        :type in_place: :class:`bool`, optional
        :param parallel: `True` to apply the transform in parallel `False` otherwise, defaults to \
        `False`
        :type parallel: :class:`bool`, optional
        :return: The new **list** if `in_place` is *False*
        :rtype: :class:`~eascv.list.List`
        """
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
        """
        Returns a new **list** with all the pending operations applied.
        If `in_place` is *True* the pending operations will be applied
        to the current **list** instead.

        :param in_place: `True` to change the current **list**, `False` to return a new one with \
         the pending transforms applied, defaults to `list`
        :type in_place: :class:`bool`, optional
        :param parallel: `True` to compute in parallel `False` otherwise, defaults to `False`
        :type parallel: :class:`bool`, optional
        :return: The new **list** if `in_place` is *False*
        :rtype: :class:`~eascv.list.List`
        """
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
        """
        Returns a copy of the current List.

        :return: The new copy
        :rtype: :class:`~eascv.list.List`
        """
        return deepcopy(self)

    @auto_compute
    def show(self, size=(10, 10), shape="auto"):
        """
        Display list images on a grid

        :param size: Size of grid
        :type size: :class:`tuple`, optional
        :param shape: Shape of grid
        :type shape: :class:`tuple`, optional
        """
        show_grid(self._images, size=size, shape=shape)
