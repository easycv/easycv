import os
import pickle
from copy import deepcopy

from easycv.transforms.base import Transform
from easycv.errors import InvalidPipelineInputSource
from easycv.operation import Operation
from easycv.utils import dict_lookup, inverse_dict_lookup


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
            self.arguments = {}  # source[0].arguments if source else {}
            # self.outputs = {}  # source[-1].outputs if source else {}
            self.forwarded = {}
            self.forwards, self.outputs = self._calculate_forwards(source)
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

    def _calculate_forwards(self, source):
        outputs = {}
        forwards = {}
        for i in range(len(source)):
            if not isinstance(source[i], Pipeline) and not isinstance(
                source[i], Transform
            ):
                raise InvalidPipelineInputSource()
            forwards[i] = {}
            outputs[i] = {}

            # calculate forwards
            for output_index in list(outputs):
                for argument in list(outputs[output_index]):
                    for pos_arg in list(outputs[output_index][argument]):
                        if source[i].can_be_forwarded(argument, pos_arg):
                            if argument not in forwards[i]:
                                forwards[i][argument] = []
                            forwards[i][argument].append(output_index)
                            outputs[output_index][argument].pop(0)
                            if not outputs[output_index][argument]:
                                outputs[output_index].pop(argument)
                                if not outputs[output_index]:
                                    outputs.pop(output_index)

            missing_args = source[i].initialize(
                index=i, forwarded=forwards[i].keys(), nested=True
            )

            # Add the missing args for forwarding
            for arg in missing_args:
                if arg not in self.arguments:
                    self.arguments[arg] = []
                if isinstance(source[i], Pipeline):
                    self.arguments[arg] += missing_args[arg]
                else:
                    self.arguments[arg].append(missing_args[arg])
                if arg not in forwards[i]:
                    forwards[i][arg] = []
                forwards[i][arg].append("in")
            source[i].forwarded = {}

            # Get the outputs of the source
            if source[i].outputs:
                for arg in source[i].outputs:
                    if isinstance(source[i], Transform):
                        tmp_arg = dict_lookup(source[i].rename_out, arg)
                    else:
                        tmp_arg = arg
                    if tmp_arg not in outputs[i]:
                        outputs[i][tmp_arg] = []
                    if isinstance(source[i].outputs[arg], list):
                        outputs[i][tmp_arg] += source[i].outputs[arg]
                    else:
                        outputs[i][tmp_arg].append(source[i].outputs[arg])
        if source:
            for arg in source[0].arguments:
                if isinstance(source[0], Transform):
                    forwards[0][arg] = ["in"]
                    if arg not in self.arguments:
                        self.arguments[arg] = []
                    self.arguments[arg] += [source[0].arguments[arg]]
                else:
                    forwards[0][arg] = ["in"] * len(source[0].arguments[arg])
                    if arg not in self.arguments:
                        self.arguments[arg] = []
                    self.arguments[arg].append(source[0].arguments[arg])
        return forwards, self.format_outputs(outputs)

    def format_outputs(self, outputs):
        f_outs = {}
        for idx in outputs:
            for arg in outputs[idx]:
                if arg not in f_outs:
                    f_outs[arg] = []
                if arg == "image":
                    if isinstance(outputs[idx][arg], list):
                        f_outs[arg] = outputs[idx][arg]
                    else:
                        f_outs[arg] = [outputs[idx][arg]]
                else:
                    if isinstance(outputs[idx][arg], list):
                        f_outs[arg] += outputs[idx][arg]
                    else:
                        f_outs[arg].append(outputs[idx][arg])
        return f_outs

    def can_be_forwarded(self, arg_name, validator):
        if arg_name in self.arguments:
            if (
                arg_name in self.forwarded
                and len(self.arguments[arg_name]) <= self.forwarded[arg_name]
            ):
                return False
            idx = 0 if arg_name not in self.forwarded else self.forwarded[arg_name]
            if self.arguments[arg_name][idx].accepts(validator):
                if arg_name not in self.forwarded:
                    self.forwarded[arg_name] = 0
                self.forwarded[arg_name] += 1
                return True
        return False

    def initialize(self, index=None, forwarded=(), nested=False):
        missing_args = {}
        aux_1 = deepcopy(self.arguments)  # this is a problem
        aux_2 = list(forwarded)

        for arg in self.arguments:
            if arg in forwarded:
                aux_1[arg].pop()
                aux_2.remove(arg)
                if not aux_1:
                    aux_1.pop(arg)
            else:
                if arg not in missing_args:
                    missing_args[arg] = []
                missing_args[arg] += self.arguments[arg]
        return missing_args

    def __call__(self, image, forward=()):
        if self._transforms and (
            list(self.arguments.keys()) == ["image"]
            or (isinstance(forward, dict) and len(forward.keys()) > 0)
        ):
            forwards = deepcopy(self.forwards)
            outputs = {"in": {"image": [image]}}
            if isinstance(forward, dict):
                outputs["in"].update(forward)
            for i in range(len(self._transforms)):
                transform = self._transforms[i]
                outputs[i] = {}
                forwarded = {}
                for arg in forwards[i]:
                    for _ in list(forwards[i][arg]):
                        ford = outputs[forwards[i][arg].pop(0)][arg].pop(0)
                        if isinstance(self._transforms[i], Transform):
                            arg = inverse_dict_lookup(
                                self._transforms[i].rename_in, arg
                            )
                            forwarded[arg] = ford
                        else:
                            if arg not in forwarded:
                                forwarded[arg] = []
                            forwarded[arg].append(ford)
                if isinstance(transform, Transform):
                    output = transform(image, forwarded=forwarded)
                    if "image" in output:
                        image = output["image"]
                else:
                    output = transform(image, forward=forwarded)
                    if "image" in output:
                        image = output["image"][0]
                for arg in output:
                    if isinstance(self._transforms[i], Pipeline):
                        if arg not in outputs[i]:
                            outputs[i][arg] = []
                        outputs[i][arg] += output[arg]
                    else:
                        tmp_arg = dict_lookup(self._transforms[i].rename_out, arg)
                        if tmp_arg not in outputs[i]:
                            outputs[i][tmp_arg] = []
                        outputs[i][tmp_arg].append(output[arg])
            real_outs = {}
            for idx in outputs:
                for arg in outputs[idx]:
                    if outputs[idx][arg]:
                        if arg not in real_outs:
                            real_outs[arg] = []
                        real_outs[arg] += outputs[idx][arg]
            return real_outs
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


def pprint(dct):
    for arg in dct:
        if arg != "image":
            print(arg, dct[arg])
        else:
            print("image", "(...)")
