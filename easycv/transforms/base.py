import cv2

from easycv.operation import Operation
from easycv.errors import (
    UnsupportedArgumentError,
    InvalidMethodError,
    ArgumentNotProvidedError,
    MissingArgumentError,
)
from easycv.utils import dict_lookup, inverse_dict_lookup
from easycv.validators import Image as ImageVal


class Metadata(type):
    exclude = {
        "run",
        "process",
        "arguments",
        "outputs",
        "method_name",
        "methods",
        "default_method",
    }

    def __dir__(cls):
        return list(set(cls.__dict__.keys()) - cls.exclude)


class Transform(Operation, metaclass=Metadata):
    methods = None
    default_method = None
    method_name = "method"

    def __init__(self, r_in=(), r_out=(), **kwargs):
        self._method = self._extract_method(kwargs)
        r_in = {} if not r_in else r_in
        r_out = {} if not r_out else r_out
        self.rename_in = {}
        self.rename_out = {}

        self.arguments = self._extract_attribute("arguments", self._method)
        self.outputs = self._extract_attribute("outputs", self._method)

        self.check_renames(r_in, r_out)

        self.forwarded = {}

        if any(arg not in self.arguments for arg in kwargs):
            raise UnsupportedArgumentError(self, method=self._method)

        if self._method is not None:
            self._args = {self.method_name: self._method}
        else:
            self._args = {}

        for arg in kwargs:
            validator = self.arguments[arg]
            validator.check(arg, kwargs[arg])
            self._args[arg] = kwargs[arg]

    def __call__(self, image, forwarded=None):
        output = self.run(image, forwarded=forwarded)

        if isinstance(output, dict):
            output.update({"image": image})  # maybe change
            return output
        else:
            if output.min() >= 0 and output.max() <= 255:
                if output.dtype.kind != "i":
                    if output.min() >= 0 and output.max() <= 1:
                        output = output * 255
                    output = output.astype("uint8")
            else:
                output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(
                    "uint8"
                )
            return {"image": output}

    def __eq__(self, other):
        return isinstance(other, Transform) and self.args == other.args

    def __repr__(self):
        self.initialize()
        return str(self)

    def __str__(self):
        args_str = []
        for arg in self._args:
            if arg != "image":
                value = (
                    self._args[arg]
                    if isinstance(self._args[arg], (str, int, float))
                    else "(...)"
                )
                if arg == self.method_name:
                    arg = "method"
                args_str.append("{}={}".format(arg, value))
        return "{} ({})".format(self.__class__.__name__, ", ".join(args_str))

    @property
    def contains_outputs(self):
        return self.outputs is not None

    @classmethod
    def contains_methods(cls):
        return cls.methods is not None

    @classmethod
    def get_methods(cls):
        return list(cls.methods) if cls.methods is not None else []

    @classmethod
    def get_default_values(cls, method=None):
        if method is None:
            if cls.default_method is not None:
                default_values = {"method": cls.default_method}
                args = cls._extract_attribute("arguments", cls.default_method)
            else:
                default_values = {}
                args = cls.arguments
        else:
            default_values = {}
            args = cls._extract_attribute("arguments", method)

        if cls.arguments is not None:
            for argument in args:
                default_values[argument] = args[argument].default

        return default_values

    @classmethod
    def _extract_attribute(cls, name, method):
        collection = cls.arguments if name == "arguments" else cls.outputs
        if collection is not None:
            to_keep = collection.keys()
            if cls.contains_methods() and isinstance(cls.methods, dict):
                if cls.methods[method] is None:
                    to_keep = []
                elif len(cls.methods[method]) > 0:
                    args = cls.methods[method].get(name)
                    if args is not None:
                        to_keep = args
            aux = {"image": ImageVal(default=None)}
            aux.update({key: collection[key] for key in to_keep})
            return aux
        else:
            return {"image": ImageVal(default=None)}

    def _extract_method(self, kwargs):
        if self.contains_methods():
            method = self.default_method
            if "method" in kwargs:
                method = kwargs.pop("method")
                if method not in self.methods:
                    raise InvalidMethodError(tuple(self.methods))
            if method is None:
                raise ArgumentNotProvidedError("method")
            return method
        else:
            return None

    def process(self, image, **kwargs):
        pass

    def run(self, image, forwarded=None):
        self.initialize()
        if forwarded is None:
            args = self._args
        else:
            args = self._args.copy()
            args.update(forwarded)
        args["image"] = image
        return self.process(**args)

    def check_renames(self, r_in, r_out):
        for arg in r_in:
            if arg in self.arguments:
                self.rename_in[arg] = r_in[
                    arg
                ]  # Maybe raise exception for invalid renames
        for arg in r_out:
            if arg in self.outputs:
                self.rename_out[arg] = r_out[arg]

    def can_be_forwarded(self, arg_name, validator):
        tmp_arg = inverse_dict_lookup(self.rename_in, arg_name)
        if tmp_arg in self.arguments and tmp_arg not in self._args:
            if dict_lookup(self.rename_in, tmp_arg) == arg_name:
                if tmp_arg not in self.forwarded and self.arguments[tmp_arg].accepts(
                    validator
                ):
                    self.forwarded[tmp_arg] = 1
                    return True
        return False

    def initialize(self, index=None, forwarded=(), nested=False):
        missing_args = {}
        for arg in self.arguments:
            temp_arg = dict_lookup(self.rename_in, arg)
            if arg not in self._args:
                if (
                    self.arguments[arg].default is None and temp_arg not in forwarded
                ) and arg != "image":
                    if nested:
                        missing_args[temp_arg] = self.arguments[arg]
                    else:
                        raise MissingArgumentError(arg, index=index)
                validator = self.arguments[arg]
                self._args[arg] = validator.default
        if nested:
            return missing_args
