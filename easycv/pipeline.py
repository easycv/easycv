import os
import pickle
from copy import deepcopy

from easycv.transforms.base import Transform
from easycv.errors import InvalidPipelineInputSource
from easycv.operation import Operation
from easycv.errors import MissingArgumentError


class Pipeline(Operation):

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
            self.forwards = {}

            self.arguments = source[0].arguments if source else {}
            self.outputs = source[-1].outputs if source else {}

            self._name = name if name else "pipeline"
            self._transforms = deepcopy(source)
            self.required = {}
            self.initialize()

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

    def initialize(self, index=None, forwarded=(), nested=True):
        outputs = {}
        for i, transform in enumerate(self._transforms):
            used_args = []
            outputs[i] = {}
            self.forwards[i] = {}
            for idx in outputs:
                for arg in list(transform.required):
                    true_arg = arg
                    if isinstance(transform, Transform) and arg in transform.renamed:
                        true_arg = transform.renamed[arg]
                    for val in list(transform.required[arg]):
                        if true_arg in outputs[idx]:
                            if outputs[idx][true_arg][0].accepts(val):
                                transform.required[arg].pop()
                                if not transform.required[arg]:
                                    transform.required.pop(arg)
                                outputs[idx][true_arg].pop()
                                if not outputs[idx][true_arg]:
                                    outputs[idx].pop(true_arg)
                                used_args.append(true_arg)
                                if arg not in self.forwards[i]:
                                    self.forwards[i][arg] = []
                                self.forwards[i][arg].append(idx)
            if transform.required:
                for item in list(transform.required):
                    if item not in self.required:
                        self.required[item] = []
                    self.required[item] += transform.required[item]
                    for _ in list(transform.required[item]):
                        if item not in self.forwards[i]:
                            self.forwards[i][item] = []
                        self.forwards[i][item].append("in")
                        transform.required[item].pop()
                        if not transform.required[item]:
                            transform.required.pop(item)
            if isinstance(transform, Transform):
                transform.initialize(index=i, forwarded=used_args, nested=nested)
                if transform.outputs:
                    for arg in transform.outputs:
                        temp_arg = arg
                        if (
                            isinstance(transform, Transform)
                            and arg in transform.renamed
                        ):
                            temp_arg = transform.renamed[arg]
                        outputs[i][temp_arg] = [transform.outputs[arg]]
            else:
                if transform.outputs:
                    for arg in transform.outputs:
                        if arg not in outputs[i]:
                            outputs[i][arg] = []
                        for out in transform.outputs[arg]:
                            outputs[i][arg].append(out)
        self.outputs = {}
        for idx in outputs:
            for arg in outputs[idx]:
                if arg not in self.outputs:
                    self.outputs[arg] = []
                for outs in outputs[idx][arg]:
                    self.outputs[arg].append(outs)

    def __call__(self, image, forwarded=()):
        if not self.required:
            forwards = deepcopy(self.forwards)
            if self._transforms:
                outputs = {} if not forwarded else {"in": forwarded}
                for i, transform in enumerate(self._transforms):
                    forwarded = {
                        arg: outputs[forwards[i][arg].pop()][arg] for arg in forwards[i]
                    }
                    output = transform(image, forwarded=forwarded)

                    if "image" in output:
                        image = output["image"]
                    outputs[i] = output

                return outputs[len(self._transforms) - 1]
            return {"image": image}
        raise MissingArgumentError(1, index=1)  # TODO GIVE CORRECT ERROR MESSAGE

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
