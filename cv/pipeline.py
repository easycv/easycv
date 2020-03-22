import os
import pickle

from cv.transforms.base import Transform
from cv.errors.io import InvalidPipelineInputSource


class Pipeline:
    """
    This class represents an **Pipeline**.

    Pipelines can be created from a list of Transforms or from a previously saved Pipeline.
    A Pipeline is simply a series of Transforms to be applied sequentially. It also supports nested
    Pipelines (Pipelines inside Pipelines). A Pipeline can be applied to an image exactly like a
    Transform.

    :param source: Pipeline data source. A list of Transforms or a path to a saved Pipeline
    :type source: :class:`list`/:class:`str`
    :param name: Name of the pipeline, "pipeline" if no name is specified
    :type name: :class:`str`, optional
    """

    def __init__(self, source, name=None):
        if isinstance(source, list) and all(
            [isinstance(x, (Transform, Pipeline)) for x in source]
        ):
            self._name = name if name else "pipeline"
            self._transforms = source
        elif isinstance(source, str) and os.path.isfile(source):
            try:
                with open(source, "rb") as f:
                    saved = pickle.load(f)
                    if isinstance(saved, Pipeline):
                        self._name = name if name else saved.name
                        self._transforms = saved.transforms()
                    else:
                        raise InvalidPipelineInputSource()
            except pickle.UnpicklingError:
                raise InvalidPipelineInputSource() from None
        else:
            raise InvalidPipelineInputSource()

    @property
    def name(self):
        """
        Returns the name of the Pipeline.

        :return: Pipeline name
        :rtype: :class:`str`
        """
        return self._name

    def description(self, level=0, start=1):
        """
        Returns Pipeline description. Nested Pipelines/Transforms are indented.
        This method calls itself recursively for nested Pipelines.

        :param level: Description indentation level, defaults to 0
        :type level: :class:`int`, optional
        :param start: Start of Transforms numeration, defaults to 1
        :type start: :class:`int`, optional
        :return: Pipeline description
        :rtype: :class:`str`
        """
        index = str(start) + ": " if (start > 1 or (start > 0 and level == 1)) else ""
        indent = "    " + "|    " * (level - 1) if level > 1 else "    " * level
        r = [
            indent
            + index
            + "Pipeline ({}) with {} transforms".format(
                self.name, self.num_transforms()
            )
        ]
        for i, t in enumerate(self._transforms):
            if isinstance(t, Pipeline):
                r.append(t.description(level=level + 1, start=i + 1))
            else:
                indent = "    " + "|    " * level
                r.append("{}{}: {}".format(indent, i + 1, str(t)))
        return "\n".join(r)

    def num_transforms(self):
        """
        Returns the total number of Transforms of the Pipeline. Nested Pipelines do not count as
        one Transform, they count as their own number of Transforms.

        :return: Total number of Transforms
        :rtype: :class:`int`
        """
        num = 0
        for t in self._transforms:
            if isinstance(t, Pipeline):
                num += t.num_transforms()
            else:
                num += 1
        return num

    def add_transform(self, transform):
        """
        Adds a Transform/Pipeline to the Pipeline. The new Transform/Pipeline is added in the end.

        :param transform: Transform/Pipeline to be added
        :type transform: :class:`~cv.transforms.base.Transform`/:class:`~cv.pipeline.Pipeline`
        """
        if isinstance(transform, (Transform, Pipeline)):
            self._transforms += [transform]
        else:
            raise ValueError("Pipelines can only contain Transforms or other pipelines")

    def transforms(self):
        """
        Returns a list with all the Transforms/Pipelines that make up the Pipeline.

        :return: Pipeline Transforms
        :rtype: :class:`list`
        """
        return self._transforms

    def copy(self):
        """
        Returns a copy of the Pipeline.

        :return: Pipeline copy
        :rtype: :class:`~cv.pipeline.Pipeline`
        """
        return Pipeline(self._transforms.copy(), name=self._name)

    def clear(self):
        """
        Clears the Pipeline (removes all Transforms/Pipelines).
        """
        self._transforms = []

    def save(self, filename=None):
        """
        Saves the Pipeline to a file.

        :param filename: Name of the saved file, if not specified Pipeline's name will be used
        :type filename: :class:`str`, optional
        """
        if not filename:
            filename = self._name + ".pipe"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def __eq__(self, other):
        return (
            isinstance(other, Pipeline)
            and self.name == other.name
            and self.num_transforms() == other.num_transforms()
            and all(t1 == t2 for t1, t2 in zip(self.transforms(), other.transforms()))
        )

    def __str__(self):
        return self.description()

    def __repr__(self):
        return str(self)

    def __call__(self, image):
        for transform in self._transforms:
            image = transform(image)
        return image
