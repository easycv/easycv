import os
import pickle
from copy import deepcopy

from easycv.transforms.base import Transform
from easycv.errors import InvalidPipelineInputSource


class Pipeline:

    """
    This class represents a **pipeline**.

    Pipelines can be created from a list of :doc:`transforms <transforms/index>` or from a \
    previously saved **pipeline**. A **pipeline** is simply a series of \
    :doc:`transforms <transforms/index>` to be applied sequentially. It also supports **nested
    pipelines** (pipelines inside pipelines). A **pipeline** can be applied to an image exactly \
    like a transform.

    :param source: Pipeline data source. A list of transforms/pipelines or a path to a \
    previously saved pipeline
    :type source: :class:`list`/:class:`str`
    :param name: Name of the **pipeline**, "pipeline" if no name is specified
    :type name: :class:`str`, optional
    """

    def __init__(self, source, name=None):
        if isinstance(source, list):
            self.forwards = Pipeline._calculate_forwards(source)

            self.arguments = source[0].arguments if source else {}
            self.outputs = source[-1].outputs if source else {}

            self._name = name if name else "pipeline"
            self._transforms = deepcopy(source)

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

    @staticmethod
    def _calculate_forwards(source):
        outputs = {}
        forwards = {}

        for i in range(len(source)):
            forwards[i] = {}

            if isinstance(source[i], Transform):
                for output_index in list(outputs):
                    for argument in list(outputs[output_index]):
                        if source[i].can_be_forwarded(
                            argument, outputs[output_index][argument]
                        ):
                            if argument not in forwards[i]:
                                forwards[i][argument] = output_index
                                outputs[output_index].pop(argument)
                                if not outputs[output_index]:
                                    outputs.pop(output_index)

                source[i].initialize(index=i, forwarded=forwards[i].keys())

            elif not isinstance(source[i], Pipeline):
                raise InvalidPipelineInputSource()

            if source[i].outputs:
                outputs[i] = source[i].outputs

        return forwards

    def __call__(self, image):
        if self._transforms:
            outputs = {}
            for i in range(len(self._transforms)):
                transform = self._transforms[i]
                forwarded = {
                    arg: outputs[self.forwards[i][arg]][arg] for arg in self.forwards[i]
                }
                if isinstance(transform, Transform):
                    output = transform(image, forwarded=forwarded)
                else:
                    output = transform(image)

                if "image" in output:
                    image = output["image"]

                outputs[i] = output

            return outputs[len(self._transforms) - 1]
        return {"image": image}

    @property
    def name(self):
        """
        Returns the name of the **pipeline**.

        :return: Pipeline name
        :rtype: :class:`str`
        """
        return self._name

    def description(self, level=0, start=1):
        """
        Returns **pipeline** description. Nested \
        :doc:`Transforms <transforms/index>`/:doc:`Pipelines <pipeline>` are indented.
        This method calls itself recursively for nested pipelines.

        :param level: Description indentation level, defaults to 0
        :type level: :class:`int`, optional
        :param start: Start of transforms numeration, defaults to 1
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
        Returns the total number of transforms of the **pipeline**. Nested pipelines do not \
        count as one transform, they count as their own number of transforms.

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

    def add_transform(self, transform, index=None):
        """
        Adds a transform/pipeline to the **pipeline**. The new transform/pipeline is added in the \
        end by default.

        :param transform: Transform/Pipeline to be added
        :type transform: :class:`~easycv.transforms.base.Transform`/\
        :class:`~easycv.pipeline.Pipeline`
        :param index: Index to add the transform, end of the list by default
        :type index: :class:`int`, optional
        """
        if isinstance(transform, (Transform, Pipeline)):
            if index is not None:
                self._transforms.insert(index, transform.copy())
            else:
                self._transforms.append(transform.copy())
            self.forwards = Pipeline._calculate_forwards(self._transforms)
        else:
            raise ValueError("Pipelines can only contain Transforms or other pipelines")

    def transforms(self):
        """
        Returns a list with all the transforms/pipelines that make up the **pipeline**.

        :return: Pipeline Transforms
        :rtype: :class:`list`
        """
        return self._transforms

    def copy(self):
        """
        Returns a copy of the **pipeline**.

        :return: Pipeline copy
        :rtype: :class:`~cv.pipeline.Pipeline`
        """
        return deepcopy(self)

    def clear(self):
        """
        Clears the **pipeline** (removes all transforms/pipelines).
        """
        self._transforms = []

    def save(self, filename=None):
        """
        Saves the **pipeline** to a file.

        :param filename: Name of the saved file, if not specified pipeline's name will be used
        :type filename: :class:`str`, optional
        """
        if not filename:
            filename = "_".join(self._name.lower().split()) + ".pipe"
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
